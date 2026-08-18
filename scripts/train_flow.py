#!/usr/bin/env python3
"""Downstream detection stage: train the normalizing flows on frozen latents.

Loads a phase-1 checkpoint of the DisentanglementModel, freezes it and drops
the decoder (scaffolding only, per CLAUDE.md), then trains on human music
only, drawing plain track windows (batch size = data.batch_tracks):

  * ConditionalStyleFlow  log p(style | content)   the research object
  * MarginalFlow          log p(content)           factorized-score term
  * ConditionalGaussian   diag-Gaussian baseline   detection sanity check

The three objectives are parameter-disjoint and optimized jointly with one
AdamW (weight decay 0 — decay pulls LU/ActNorm toward singular transforms).
Before optimization, fixed whitening statistics are fitted on the first
--whiten-windows training windows and stored in the model buffers.

Examples:
    python scripts/train_flow.py --encoder-checkpoint checkpoints/last.pt
    # local smoke run:
    python scripts/train_flow.py --encoder-checkpoint checkpoints/last.pt \
        --train-split val --val-split none --window-seconds 5 \
        --batch-tracks 2 --whiten-windows 4 --max-steps 3 \
        --device cpu --num-workers 0
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from shs100k_meta import DEFAULT_DATA_ROOT  # noqa: E402
from src.config import TrainConfig, config_from_dict, load_config  # noqa: E402
from src.data import TrackWindowDataset, WindowConfig, read_tracks  # noqa: E402
from src.models import (  # noqa: E402
    ConditionalGaussian,
    DisentanglementModel,
    FlowDetector,
    MertExtractor,
)
from src.training import (  # noqa: E402
    extract_mixes,
    make_optimizer,
    make_scheduler,
    pick_device,
    worker_init,
)


def build_loader(
    config: TrainConfig, data_root: Path, split: str, train: bool
) -> DataLoader | None:
    if split.lower() == "none":
        return None
    window = WindowConfig(
        window_seconds=config.data.window_seconds,
        sample_rate=config.mert.sample_rate,
    )
    dataset = TrackWindowDataset(read_tracks(data_root, split), data_root, window)
    if len(dataset) == 0:
        return None
    if not train:
        # num_workers=0 so seeding `random` makes the sampled windows reproducible
        return DataLoader(dataset, batch_size=config.data.batch_tracks, num_workers=0)
    return DataLoader(
        dataset,
        batch_size=config.data.batch_tracks,
        shuffle=True,
        num_workers=config.data.num_workers,
        worker_init_fn=worker_init,
        drop_last=len(dataset) > config.data.batch_tracks,
    )


def load_frozen_encoder(
    path: Path, device: torch.device
) -> tuple[DisentanglementModel, TrainConfig]:
    """Rebuild the phase-1 model from its checkpoint config and freeze it."""
    checkpoint = torch.load(path, map_location=device)
    encoder_config = config_from_dict(checkpoint["config"])
    model = DisentanglementModel(encoder_config)
    model.load_state_dict(checkpoint["model"])
    del model.decoder  # discarded after representation learning (CLAUDE.md)
    model.requires_grad_(False)
    model.eval()
    return model.to(device), encoder_config


@torch.no_grad()
def encode_windows(
    mert: MertExtractor,
    encoder: DisentanglementModel,
    waves: torch.Tensor,
    micro_batch: int,
    autocast,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Waveforms (B, L) -> content, style tokens (B, n_tokens, token_dim) fp32."""
    with autocast:
        content_mix, style_mix = extract_mixes(mert, encoder, waves, micro_batch)
        content, style = encoder.encode_mixes(content_mix, style_mix)
    return content.float(), style.float()  # flows want fp32, not bf16


def fit_whitening(
    loader: DataLoader,
    mert: MertExtractor,
    encoder: DisentanglementModel,
    detector: FlowDetector,
    baseline: ConditionalGaussian,
    config: TrainConfig,
    device: torch.device,
    autocast,
) -> None:
    n_windows = config.flow.whiten_windows
    contents, styles, count = [], [], 0
    for batch in loader:
        content, style = encode_windows(
            mert, encoder, batch["wave"].to(device), config.mert.micro_batch, autocast
        )
        contents.append(content.cpu())
        styles.append(style.cpu())
        count += content.shape[0]
        if count >= n_windows:
            break
    content = torch.cat(contents)[:n_windows]
    style = torch.cat(styles)[:n_windows]
    detector.fit_whitening(content, style)
    baseline.whitener.fit(style.flatten(1))
    print(f"whitening statistics fitted on {content.shape[0]} windows")


def compute_nlls(
    detector: FlowDetector,
    baseline: ConditionalGaussian,
    content: torch.Tensor,
    style: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "nll_style": -detector.style_flow.log_prob(content, style).mean(),
        "nll_content": -detector.content_flow.log_prob(content).mean(),
        "nll_gaussian": -baseline.log_prob(content, style).mean(),
    }


@torch.no_grad()
def validate(
    mert: MertExtractor,
    encoder: DisentanglementModel,
    detector: FlowDetector,
    baseline: ConditionalGaussian,
    config: TrainConfig,
    loader: DataLoader,
    device: torch.device,
    autocast,
) -> dict[str, float]:
    detector.eval()
    baseline.eval()
    rng_state = random.getstate()
    random.seed(config.seed)  # reproducible window sampling across epochs
    sums: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        if config.data.val_max_batches and n_batches >= config.data.val_max_batches:
            break
        content, style = encode_windows(
            mert, encoder, batch["wave"].to(device), config.mert.micro_batch, autocast
        )
        for name, value in compute_nlls(detector, baseline, content, style).items():
            sums[name] = sums.get(name, 0.0) + float(value)
        n_batches += 1
    random.setstate(rng_state)
    detector.train()
    baseline.train()
    return {f"val/{name}": s / max(n_batches, 1) for name, s in sums.items()}


def save_checkpoint(
    path: Path,
    detector: FlowDetector,
    baseline: ConditionalGaussian,
    optimizer,
    scheduler,
    config: TrainConfig,
    encoder_checkpoint: Path,
    epoch: int,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "detector": detector.state_dict(),
            "baseline": baseline.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config.to_dict(),
            "encoder_checkpoint": str(encoder_checkpoint),
            "epoch": epoch,
            "step": step,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--encoder-checkpoint", type=Path, required=True,
        help="phase-1 checkpoint providing the frozen encoders",
    )
    parser.add_argument("--config", type=Path, help="JSON config overriding defaults")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--train-split")
    parser.add_argument("--val-split")
    parser.add_argument("--window-seconds", type=float)
    parser.add_argument("--batch-tracks", type=int, help="windows per step")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--val-max-batches", type=int)
    parser.add_argument("--whiten-windows", type=int)
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
    parser.add_argument("--resume", type=Path, help="flow checkpoint to resume from")
    return parser.parse_args()


def apply_overrides(config: TrainConfig, args: argparse.Namespace) -> None:
    data, direct = config.data, config
    mapping = [
        (args.data_root, lambda v: setattr(data, "data_root", str(v))),
        (args.train_split, lambda v: setattr(data, "train_split", v)),
        (args.val_split, lambda v: setattr(data, "val_split", v)),
        (args.window_seconds, lambda v: setattr(data, "window_seconds", v)),
        (args.batch_tracks, lambda v: setattr(data, "batch_tracks", v)),
        (args.num_workers, lambda v: setattr(data, "num_workers", v)),
        (args.val_max_batches, lambda v: setattr(data, "val_max_batches", v)),
        (args.whiten_windows, lambda v: setattr(config.flow, "whiten_windows", v)),
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
    ]
    for value, setter in mapping:
        if value is not None:
            setter(value)


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

    encoder, encoder_config = load_frozen_encoder(args.encoder_checkpoint, device)
    mert = MertExtractor(encoder_config.mert).to(device)
    print(f"frozen encoder loaded from {args.encoder_checkpoint}")

    train_loader = build_loader(config, data_root, config.data.train_split, train=True)
    if train_loader is None:
        raise SystemExit(f"no usable tracks in split {config.data.train_split!r}")
    val_loader = build_loader(config, data_root, config.data.val_split, train=False)
    steps_per_epoch = len(train_loader)
    total_steps = config.max_steps or config.epochs * steps_per_epoch
    print(
        f"train[{config.data.train_split}]: {len(train_loader.dataset)} tracks, "
        f"{steps_per_epoch} steps/epoch, {total_steps} total steps"
    )

    # latents come from the checkpoint's bottleneck; flow sizes from this config
    detector = FlowDetector(encoder_config.bottleneck, config.flow).to(device)
    baseline = ConditionalGaussian(encoder_config.bottleneck, config.flow).to(device)
    trainable = nn.ModuleDict({"detector": detector, "baseline": baseline})
    n_params = sum(p.numel() for p in trainable.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params / 1e6:.1f}M")

    optimizer = make_optimizer(trainable, config, weight_decay=0.0)
    scheduler = make_scheduler(optimizer, config, total_steps)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    start_epoch, global_step = 0, 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        detector.load_state_dict(checkpoint["detector"])
        baseline.load_state_dict(checkpoint["baseline"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["step"]
        print(f"resumed from {args.resume} (epoch {start_epoch}, step {global_step})")
    else:
        fit_whitening(
            train_loader, mert, encoder, detector, baseline, config, device, autocast
        )

    run_name = time.strftime("flow-%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(Path(config.log_dir) / run_name))
    writer.add_text("config", f"```json\n{config.to_dict()}\n```")
    checkpoint_path = Path(config.checkpoint_dir) / "flow_last.pt"

    trainable.train()
    done = False
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        for batch in train_loader:
            content, style = encode_windows(
                mert, encoder, batch["wave"].to(device), config.mert.micro_batch,
                autocast,
            )
            losses = compute_nlls(detector, baseline, content, style)
            total = sum(losses.values())

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable.parameters(), config.optim.grad_clip
            )
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % config.log_every == 0 or global_step == 1:
                parts = "  ".join(
                    f"{name}={float(value):.2f}" for name, value in losses.items()
                )
                print(f"epoch {epoch} step {global_step}/{total_steps}  {parts}")
                for name, value in losses.items():
                    writer.add_scalar(f"train/{name}", float(value), global_step)
                writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            if (
                config.checkpoint_every
                and global_step % config.checkpoint_every == 0
            ):
                save_checkpoint(
                    checkpoint_path, detector, baseline, optimizer, scheduler,
                    config, args.encoder_checkpoint, epoch, global_step,
                )
            if config.max_steps and global_step >= config.max_steps:
                done = True
                break

        print(f"epoch {epoch} finished in {time.time() - epoch_start:.0f}s")
        if val_loader is not None:
            metrics = validate(
                mert, encoder, detector, baseline, config, val_loader, device, autocast
            )
            parts = "  ".join(f"{n}={v:.2f}" for n, v in metrics.items())
            print(f"  {parts}")
            for name, value in metrics.items():
                writer.add_scalar(name, value, global_step)

        save_checkpoint(
            checkpoint_path, detector, baseline, optimizer, scheduler,
            config, args.encoder_checkpoint, epoch + 1, global_step,
        )
        if done:
            break

    writer.close()
    print(f"done at step {global_step}; checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
