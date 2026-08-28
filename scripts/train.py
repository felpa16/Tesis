#!/usr/bin/env python3
"""Phase-1 (online MERT) training of the disentanglement model.

Each step draws P aligned cover pairs + R plain track windows, runs frozen
MERT on the fly, and optimizes the CLAUDE.md objectives: plain reconstruction
(2a), content contrastive / MIL-NCE (1), cover-swap reconstruction (2b),
latent cycle-consistency (2c), and content-style decorrelation (3).

Layer-mix weights are logged every epoch together with their cosine
similarity to the previous epoch — the phase-1 freeze criterion. Once they
stabilize, rerun with --freeze-layer-weights (or move to phase-2 caching).

Examples:
    python scripts/train.py --train-split train --val-split val
    # local smoke run:
    python scripts/train.py --train-split val --val-split val \
        --window-seconds 5 --batch-pairs 2 --batch-tracks 2 \
        --max-steps 5 --device cpu --num-workers 0
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from shs100k_meta import DEFAULT_DATA_ROOT  # noqa: E402
from src.config import TrainConfig, load_config  # noqa: E402
from src.data import (  # noqa: E402
    AlignedPairDataset,
    TrackWindowDataset,
    WindowConfig,
    read_pairs,
    read_tracks,
)
from src.losses import pool_tokens  # noqa: E402
from src.models import DisentanglementModel, MertExtractor  # noqa: E402
from src.training import (  # noqa: E402
    compute_losses,
    extract_mixes,
    make_optimizer,
    make_scheduler,
    pick_device,
    pooled_targets,
    worker_init,
)


def make_window_config(config: TrainConfig, n_candidates: int) -> WindowConfig:
    return WindowConfig(
        window_seconds=config.data.window_seconds,
        sample_rate=config.mert.sample_rate,
        n_candidates=n_candidates,
    )


def build_train_loaders(
    config: TrainConfig, data_root: Path
) -> tuple[DataLoader, DataLoader | None]:
    split = config.data.train_split
    tracks = read_tracks(data_root, split)
    pairs = read_pairs(data_root, split)
    pair_dataset = AlignedPairDataset(
        pairs, tracks, data_root, make_window_config(config, config.data.n_candidates)
    )
    if len(pair_dataset) == 0:
        raise SystemExit(f"no usable aligned pairs in split {split!r}")
    pair_loader = DataLoader(
        pair_dataset,
        batch_size=config.data.batch_pairs,
        shuffle=True,
        num_workers=config.data.num_workers,
        worker_init_fn=worker_init,
        drop_last=len(pair_dataset) > config.data.batch_pairs,
    )
    track_loader = None
    if config.data.batch_tracks > 0:
        track_dataset = TrackWindowDataset(
            tracks, data_root, make_window_config(config, 1)
        )
        track_loader = DataLoader(
            track_dataset,
            batch_size=config.data.batch_tracks,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=worker_init,
            drop_last=False,
        )
    return pair_loader, track_loader


def build_val_loader(config: TrainConfig, data_root: Path) -> DataLoader | None:
    if config.data.val_split.lower() == "none":
        return None
    tracks = read_tracks(data_root, config.data.val_split)
    pairs = read_pairs(data_root, config.data.val_split)
    dataset = AlignedPairDataset(
        pairs, tracks, data_root, make_window_config(config, 1)
    )
    if len(dataset) == 0:
        return None
    # num_workers=0 so seeding `random` makes the sampled windows reproducible
    return DataLoader(dataset, batch_size=config.data.batch_pairs, num_workers=0)


def repeat_forever(loader: DataLoader):
    while True:
        yield from loader


def assemble_waves(
    pair_batch: dict, track_batch: dict | None, device: torch.device
) -> tuple[torch.Tensor, int, int]:
    """Concatenate [A-windows, flattened B-candidates, track windows]."""
    wave_a = pair_batch["wave_a"]
    waves_b = pair_batch["waves_b"]
    p, k, length = waves_b.shape
    parts = [wave_a, waves_b.reshape(p * k, length)]
    if track_batch is not None:
        parts.append(track_batch["wave"])
    return torch.cat(parts).to(device), p, k


@torch.no_grad()
def validate(
    mert: MertExtractor,
    model: DisentanglementModel,
    config: TrainConfig,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    rng_state = random.getstate()
    random.seed(config.seed)  # reproducible window sampling across epochs
    sums: dict[str, float] = defaultdict(float)
    n_batches = 0
    a_vectors, b_vectors = [], []
    for batch in loader:
        if config.data.val_max_batches and n_batches >= config.data.val_max_batches:
            break
        waves, p, k = assemble_waves(batch, None, device)
        content_mix, style_mix = extract_mixes(
            mert, model, waves, config.mert.micro_batch
        )
        content_target, style_target = pooled_targets(
            config.loss, content_mix, style_mix
        )
        content, style = model.encode_mixes(content_mix, style_mix)
        total, losses = compute_losses(
            model, config.loss, content, style, content_target, style_target, p, k
        )
        sums["total"] += float(total)
        for name, value in losses.items():
            sums[name] += float(value)
        n_batches += 1
        a_vectors.append(pool_tokens(content[:p]).cpu())
        b_vectors.append(pool_tokens(content[p : p + p * k : k]).cpu())
    random.setstate(rng_state)
    model.train()

    metrics = {f"val/{name}": s / max(n_batches, 1) for name, s in sums.items()}
    a = torch.cat(a_vectors)
    b = torch.cat(b_vectors)
    if len(a) >= 2:
        # content invariance: does c_a retrieve its own cover's window?
        similarity = a @ b.T
        hits = similarity.argmax(dim=1) == torch.arange(len(a))
        metrics["val/content_recall@1"] = float(hits.float().mean())
    return metrics


def log_layer_weights(
    writer: SummaryWriter,
    weights: dict[str, torch.Tensor],
    previous: dict[str, torch.Tensor] | None,
    epoch: int,
) -> None:
    for branch, w in weights.items():
        for layer, value in enumerate(w.tolist()):
            writer.add_scalar(f"layer_weights/{branch}/{layer:02d}", value, epoch)
        if previous is not None:
            cosine = float(F.cosine_similarity(w, previous[branch], dim=0))
            writer.add_scalar(f"layer_weights/{branch}_cosine_to_prev", cosine, epoch)
            print(f"  layer weights [{branch}] cosine to previous epoch: {cosine:.6f}")


def save_checkpoint(
    path: Path,
    model: DisentanglementModel,
    optimizer,
    scheduler,
    config: TrainConfig,
    epoch: int,
    step: int,
    prev_weights: dict[str, torch.Tensor] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config.to_dict(),
            "epoch": epoch,
            "step": step,
            "prev_layer_weights": prev_weights,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="JSON config overriding defaults")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--train-split")
    parser.add_argument("--val-split")
    parser.add_argument("--window-seconds", type=float)
    parser.add_argument("--batch-pairs", type=int)
    parser.add_argument("--batch-tracks", type=int)
    parser.add_argument("--n-candidates", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--val-max-batches", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--mert-micro-batch", type=int)
    parser.add_argument(
        "--recon-pool",
        type=int,
        help="temporal pooling factor for the reconstruction target (1 = off)",
    )
    parser.add_argument("--freeze-layer-weights", action="store_true")
    parser.add_argument("--resume", type=Path, help="checkpoint to resume from")
    return parser.parse_args()


def apply_overrides(config: TrainConfig, args: argparse.Namespace) -> None:
    data, direct = config.data, config
    mapping = [
        (args.data_root, lambda v: setattr(data, "data_root", str(v))),
        (args.train_split, lambda v: setattr(data, "train_split", v)),
        (args.val_split, lambda v: setattr(data, "val_split", v)),
        (args.window_seconds, lambda v: setattr(data, "window_seconds", v)),
        (args.batch_pairs, lambda v: setattr(data, "batch_pairs", v)),
        (args.batch_tracks, lambda v: setattr(data, "batch_tracks", v)),
        (args.n_candidates, lambda v: setattr(data, "n_candidates", v)),
        (args.num_workers, lambda v: setattr(data, "num_workers", v)),
        (args.val_max_batches, lambda v: setattr(data, "val_max_batches", v)),
        (args.epochs, lambda v: setattr(direct, "epochs", v)),
        (args.max_steps, lambda v: setattr(direct, "max_steps", v)),
        (args.device, lambda v: setattr(direct, "device", v)),
        (args.lr, lambda v: setattr(config.optim, "lr", v)),
        (args.seed, lambda v: setattr(direct, "seed", v)),
        (args.log_dir, lambda v: setattr(direct, "log_dir", v)),
        (args.checkpoint_dir, lambda v: setattr(direct, "checkpoint_dir", v)),
        (args.log_every, lambda v: setattr(direct, "log_every", v)),
        (args.checkpoint_every, lambda v: setattr(direct, "checkpoint_every", v)),
        (args.mert_micro_batch, lambda v: setattr(config.mert, "micro_batch", v)),
        (args.recon_pool, lambda v: setattr(config.loss, "recon_pool", v)),
    ]
    for value, setter in mapping:
        if value is not None:
            setter(value)
    if args.freeze_layer_weights:
        config.freeze_layer_weights = True


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = pick_device(config.device)
    data_root = (
        Path(config.data.data_root) if config.data.data_root else DEFAULT_DATA_ROOT
    )
    print(f"device={device.type}  data_root={data_root}")

    pair_loader, track_loader = build_train_loaders(config, data_root)
    val_loader = build_val_loader(config, data_root)
    steps_per_epoch = len(pair_loader)
    total_steps = config.max_steps or config.epochs * steps_per_epoch
    print(
        f"train[{config.data.train_split}]: {len(pair_loader.dataset)} pairs, "
        f"{steps_per_epoch} steps/epoch, {total_steps} total steps"
    )

    model = DisentanglementModel(config).to(device)
    if config.freeze_layer_weights:
        model.freeze_layer_weights()
        print("layer-mix weights frozen")
    mert = MertExtractor(config.mert).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable / 1e6:.1f}M")

    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config, total_steps)

    start_epoch, global_step = 0, 0
    prev_weights: dict[str, torch.Tensor] | None = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["step"]
        prev_weights = checkpoint.get("prev_layer_weights")
        print(f"resumed from {args.resume} (epoch {start_epoch}, step {global_step})")

    run_name = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(Path(config.log_dir) / run_name))
    writer.add_text("config", f"```json\n{config.to_dict()}\n```")
    checkpoint_dir = Path(config.checkpoint_dir)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    track_iter = repeat_forever(track_loader) if track_loader is not None else None
    model.train()
    done = False
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        for pair_batch in pair_loader:
            track_batch = next(track_iter) if track_iter is not None else None
            waves, p, k = assemble_waves(pair_batch, track_batch, device)

            with autocast:
                content_mix, style_mix = extract_mixes(
                    mert, model, waves, config.mert.micro_batch
                )
                content_target, style_target = pooled_targets(
                    config.loss, content_mix, style_mix
                )
                model.content_std.update(content_target)
                model.style_std.update(style_target)
                content, style = model.encode_mixes(content_mix, style_mix)
                total, losses = compute_losses(
                    model,
                    config.loss,
                    content,
                    style,
                    content_target,
                    style_target,
                    p,
                    k,
                )

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.optim.grad_clip
            )
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % config.log_every == 0 or global_step == 1:
                parts = "  ".join(
                    f"{name}={float(value):.4f}" for name, value in losses.items()
                )
                print(
                    f"epoch {epoch} step {global_step}/{total_steps}  "
                    f"total={float(total):.4f}  {parts}"
                )
                writer.add_scalar("train/total", float(total), global_step)
                for name, value in losses.items():
                    writer.add_scalar(f"train/{name}", float(value), global_step)
                writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                writer.add_scalar(
                    "train/lr", scheduler.get_last_lr()[0], global_step
                )

            if (
                config.checkpoint_every
                and global_step % config.checkpoint_every == 0
            ):
                save_checkpoint(
                    checkpoint_dir / "last.pt", model, optimizer, scheduler,
                    config, epoch, global_step, prev_weights,
                )
            if config.max_steps and global_step >= config.max_steps:
                done = True
                break

        print(f"epoch {epoch} finished in {time.time() - epoch_start:.0f}s")
        weights = model.layer_weight_summary()
        log_layer_weights(writer, weights, prev_weights, epoch)
        prev_weights = weights

        if val_loader is not None:
            metrics = validate(mert, model, config, val_loader, device)
            parts = "  ".join(f"{n}={v:.4f}" for n, v in metrics.items())
            print(f"  {parts}")
            for name, value in metrics.items():
                writer.add_scalar(name, value, global_step)

        save_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer, scheduler,
            config, epoch + 1, global_step, prev_weights,
        )
        if done:
            break

    writer.close()
    print(f"done at step {global_step}; checkpoint: {checkpoint_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
