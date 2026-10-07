#!/usr/bin/env python3
"""Phase-1 (online MERT) training of the disentanglement model.

Each step draws P aligned cover pairs + R plain track windows, runs frozen
MERT on the fly, and optimizes the CLAUDE.md objectives: plain reconstruction
(2a), content contrastive / MIL-NCE (1), cover-swap reconstruction (2b),
latent cycle-consistency (2c), and content-style decorrelation (3).

Layer-mix weights are logged every epoch together with their cosine
similarity to the previous epoch — the phase-1 freeze criterion. Once they
stabilize, rerun with --freeze-layer-weights (or move to phase-2 caching).
--layer-weights starts both mixes from the file phase 1 froze, so a fresh run
with --freeze-layer-weights trains the encoders on the chosen layers, not on a
uniform average.

Validation runs every --val-every steps (default: each epoch end). Each
validation appends to {checkpoint_dir}/val_metrics.jsonl, together with the
mean training losses since the previous one, and best.pt keeps the checkpoint
with the best --select-metric. With --max-steps the run always reaches that
many steps, taking as many epochs as the training set needs, so runs on
subsets of different sizes get the same budget.

Examples:
    python scripts/train.py --train-split train --val-split val
    python scripts/train.py --train-split train_w25 --val-split val50 \
        --layer-weights phase1_layer_weights.pt --freeze-layer-weights \
        --max-steps 45000 --val-every 2500 --checkpoint-dir checkpoints/lc-w25
    # local smoke run:
    python scripts/train.py --train-split val --val-split val \
        --window-seconds 5 --batch-pairs 2 --batch-tracks 2 \
        --max-steps 5 --device cpu --num-workers 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
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
from src.losses import (  # noqa: E402
    ContentQueue,
    MomentumContentEncoder,
    pool_tokens,
)
from src.metrics import content_retrieval  # noqa: E402
from src.models import DisentanglementModel, MertExtractor  # noqa: E402
from src.training import (  # noqa: E402
    compute_losses,
    extract_mixes,
    LayerMixConvergence,
    load_layer_weights,
    save_layer_weights,
    make_optimizer,
    make_scheduler,
    pick_device,
    pooled_targets,
    worker_init,
)


def make_window_config(
    config: TrainConfig, n_candidates: int, augment: bool = False
) -> WindowConfig:
    """augment=True only for the *training* pair stream: validation must stay
    comparable across runs, and the track stream is pure reconstruction data."""
    return WindowConfig(
        window_seconds=config.data.window_seconds,
        sample_rate=config.mert.sample_rate,
        n_candidates=n_candidates,
        augment=config.data.augment if augment else None,
    )


def build_train_loaders(
    config: TrainConfig, data_root: Path
) -> tuple[DataLoader, DataLoader | None]:
    split = config.data.train_split
    tracks = read_tracks(data_root, split)
    pairs = read_pairs(data_root, split)
    # Capacity/optimization probe: keep only as much data as --overfit-batches
    # asks for, and stop shuffling, so every epoch replays the same batches.
    # `tracks` stays whole here: AlignedPairDataset uses it as a lookup table
    # and silently drops any pair whose two recordings are missing from it.
    overfit_tracks = tracks
    if config.overfit_batches:
        want = config.overfit_batches * config.data.batch_pairs
        # Stride, never a head slice: manifests list a work's pairs
        # contiguously, so pairs[:want] is one work, every in-batch negative is
        # masked as same-work, and the contrastive term reads exactly 0.0000 —
        # which is what the first version of this probe reported.
        pairs = pairs[:: max(len(pairs) // want, 1)][:want]
        overfit_tracks = tracks[:: max(len(tracks) // max(
            config.overfit_batches * config.data.batch_tracks, 1), 1)]
    pair_dataset = AlignedPairDataset(
        pairs,
        tracks,
        data_root,
        make_window_config(config, config.data.n_candidates, augment=True),
    )
    if len(pair_dataset) == 0:
        raise SystemExit(f"no usable aligned pairs in split {split!r}")
    overfit = bool(config.overfit_batches)
    pair_loader = DataLoader(
        pair_dataset,
        batch_size=config.data.batch_pairs,
        shuffle=not overfit,
        num_workers=0 if overfit else config.data.num_workers,
        worker_init_fn=worker_init,
        drop_last=len(pair_dataset) > config.data.batch_pairs,
    )
    track_loader = None
    if config.data.batch_tracks > 0:
        track_dataset = TrackWindowDataset(
            overfit_tracks, data_root, make_window_config(config, 1)
        )
        track_loader = DataLoader(
            track_dataset,
            batch_size=config.data.batch_tracks,
            shuffle=not overfit,
            num_workers=0 if overfit else config.data.num_workers,
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
    # Manifests list a work's pairs contiguously, so batches in file order hold
    # one work each and every in-batch negative would be a masked same-work
    # pair. A fixed seeded permutation mixes works and is identical at every
    # validation, and across runs with the same seed.
    order = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(config.seed)
    ).tolist()
    # num_workers=0 so seeding `random` makes the sampled windows reproducible
    return DataLoader(
        Subset(dataset, order), batch_size=config.data.batch_pairs, num_workers=0
    )


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


def pair_song_ids(pair_batch: dict, device: torch.device) -> torch.Tensor:
    """(P,) work id of each aligned pair in a batch, read off its track key."""
    return torch.tensor(
        [int(key.split("_")[0]) for key in pair_batch["key_a"]], device=device
    )


@torch.no_grad()
def validate(
    mert: MertExtractor,
    model: DisentanglementModel,
    config: TrainConfig,
    loader: DataLoader,
    device: torch.device,
    prefix: str = "val",
    return_queries: bool = False,
):
    """Mean losses plus content retrieval over a held-out pair split.

    Returns the metrics, and with return_queries also the per-query retrieval
    tensors (see src.metrics.content_retrieval) plus each query's work, for
    bootstrapping over works.
    """
    model.eval()
    rng_state = random.getstate()
    random.seed(config.seed)  # reproducible window sampling across epochs
    sums: dict[str, float] = defaultdict(float)
    n_batches = 0
    a_vectors, b_vectors = [], []
    keys_a: list[str] = []
    keys_b: list[str] = []
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
            model,
            config.loss,
            content,
            style,
            content_target,
            style_target,
            p,
            k,
            pair_song_ids(batch, device),
        )
        sums["total"] += float(total)
        for name, value in losses.items():
            sums[name] += float(value)
        n_batches += 1
        a_vectors.append(pool_tokens(content[:p]).float().cpu())
        b_vectors.append(pool_tokens(content[p : p + p * k : k]).float().cpu())
        keys_a += list(batch["key_a"])
        keys_b += list(batch["key_b"])
    random.setstate(rng_state)
    model.train()

    metrics = {f"{prefix}/{name}": s / max(n_batches, 1) for name, s in sums.items()}
    queries: dict[str, torch.Tensor] = {}
    if len(keys_a) >= 2:
        # content invariance: does c_a retrieve its own cover's window, and a
        # cover of its own composition?
        track_index = {key: i for i, key in enumerate(sorted(set(keys_a) | set(keys_b)))}
        works = torch.tensor([int(key.split("_")[0]) for key in keys_a])
        queries = content_retrieval(
            torch.cat(a_vectors),
            torch.cat(b_vectors),
            works,
            torch.tensor([track_index[key] for key in keys_a]),
            torch.tensor([track_index[key] for key in keys_b]),
        )
        metrics[f"{prefix}/content_recall@1"] = float(queries["pair_hit"].mean())
        metrics[f"{prefix}/content_work_recall@1"] = float(queries["work_hit"].mean())
        metrics[f"{prefix}/content_work_map"] = float(queries["work_ap"].mean())
        queries["work"] = works
    if return_queries:
        return metrics, queries
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
    best_metric: float | None = None,
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
            "best_metric": best_metric,
        },
        path,
    )


def run_info_dict(
    config: TrainConfig,
    pairs: list,
    tracks: list,
    total_steps: int,
    started: str | None = None,
) -> dict:
    """What a run trained on. Shared with scripts/rebuild_run_info.py, which
    reconstructs this file from a checkpoint when it is lost, so the two can
    never disagree about which fields a run is described by."""
    return {
        "train_split": config.data.train_split,
        "val_split": config.data.val_split,
        "n_pairs": len(pairs),
        "n_pair_works": len({pair.song_id for pair in pairs}),
        "n_tracks": len(tracks),
        "n_works": len({track.song_id for track in tracks} | {p.song_id for p in pairs}),
        "total_steps": total_steps,
        "batch_pairs": config.data.batch_pairs,
        "batch_tracks": config.data.batch_tracks,
        "n_candidates": config.data.n_candidates,
        "negative_queue": config.loss.negative_queue,
        "queue_momentum": config.loss.queue_momentum,
        "augment": config.data.augment.enabled,
        "layer_weights": config.layer_weights,
        "freeze_layer_weights": config.freeze_layer_weights,
        "seed": config.seed,
        "started": started or time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def write_run_info(
    checkpoint_dir: Path,
    config: TrainConfig,
    pair_loader: DataLoader,
    track_loader: DataLoader | None,
    total_steps: int,
) -> None:
    """Write run_info.json next to the checkpoints (read by learning_curve.py)."""
    info = run_info_dict(
        config,
        pair_loader.dataset.pairs,
        tracks=track_loader.dataset.tracks if track_loader is not None else [],
        total_steps=total_steps,
    )
    with open(checkpoint_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="JSON config overriding defaults")
    parser.add_argument(
        "--data-root",
        help="directory holding audio/, alignments/ and manifests/. Taken as a "
        "string, not a Path, so that an empty $DATA is detectable: Path('') "
        "collapses to '.' and would silently resolve to the current directory",
    )
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
    parser.add_argument(
        "--unfreeze-layer-weights",
        action="store_true",
        help="reset both MERT layer mixes to uniform and train them for the "
        "whole run. Overrides --layer-weights and --freeze-layer-weights, "
        "which scripts/launch_run.sh always passes, so this is how you opt "
        "out of them without bypassing that script's guards. The mix "
        "trajectory is reported every --converge-every steps so you can see "
        "afterwards whether it settled; use --converge-mert instead to stop "
        "the run once it has",
    )
    parser.add_argument(
        "--layer-weights",
        type=Path,
        help="phase-1 layer-weights file to start both mixes from "
        "(combine with --freeze-layer-weights)",
    )
    parser.add_argument(
        "--val-every", type=int, help="steps between validations (0 = epoch end)"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="style-only augmentation (EQ, bandwidth, reverb, saturation, "
        "noise, gain) of each pair's A-side window. Synthesises extra "
        "performances of a known work, which is what attacks work "
        "memorisation. Augmented windows are excluded from the reconstruction "
        "terms; pitch and tempo are not augmented",
    )
    parser.add_argument(
        "--negative-queue",
        type=int,
        help="extra contrastive negatives kept from earlier steps (default "
        "1000, 0 = off). Enlarges the denominator without enlarging the batch, "
        "which is capped by MERT activation memory",
    )
    parser.add_argument(
        "--queue-momentum",
        type=float,
        help="EMA rate of the key encoder that fills the queue (default 0.999). "
        "Lower moves the keys faster and makes them staler relative to each "
        "other; 1.0 freezes them at initialization",
    )
    parser.add_argument(
        "--overfit-batches",
        type=int,
        help="train on this many fixed batches, replayed every epoch, with "
        "validation off. If the losses do not collapse toward zero, the "
        "problem is capacity or optimization, not the data",
    )
    parser.add_argument(
        "--select-metric",
        help="validation metric that picks best.pt (default val/total; "
        "recall/map metrics are maximized, everything else minimized)",
    )
    parser.add_argument(
        "--converge-mert",
        action="store_true",
        help="reset both MERT layer mixes to uniform, train them, and stop "
        "when they stop moving. Overrides --layer-weights and "
        "--freeze-layer-weights. Writes the converged vectors to "
        "{checkpoint-dir}/phase1_layer_weights.pt for --layer-weights and "
        "scripts/extract_mert_features.py to consume",
    )
    parser.add_argument(
        "--converge-every",
        type=int,
        default=500,
        help="steps between layer-mix samples (default 500)",
    )
    parser.add_argument(
        "--converge-window",
        type=int,
        default=4,
        help="compare against the sample this many checks ago, so the lag is "
        "--converge-every x this (default 4 = 2000 steps)",
    )
    parser.add_argument(
        "--converge-tol",
        type=float,
        default=0.02,
        help="converged when ||w_now - w_then|| / ||w_now - uniform|| is below "
        "this for both branches (default 0.02, i.e. 2%% of the distance the "
        "vector has travelled from uniform). An *absolute* threshold cannot "
        "work: the whole journey from uniform is only ~0.033 in norm, so e.g. "
        "MSE < 1e-3 is satisfied at step 0",
    )
    parser.add_argument(
        "--converge-patience",
        type=int,
        default=3,
        help="consecutive passing checks required (default 3)",
    )
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
        (args.layer_weights, lambda v: setattr(direct, "layer_weights", str(v))),
        (args.val_every, lambda v: setattr(direct, "val_every", v)),
        (args.overfit_batches, lambda v: setattr(direct, "overfit_batches", v)),
        (args.negative_queue, lambda v: setattr(config.loss, "negative_queue", v)),
        (args.queue_momentum, lambda v: setattr(config.loss, "queue_momentum", v)),
        (args.augment or None, lambda v: setattr(config.data.augment, "enabled", v)),
        (args.select_metric, lambda v: setattr(direct, "select_metric", v)),
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
    if args.data_root is not None and not args.data_root.strip():
        raise SystemExit(
            "--data-root is empty. $DATA is probably unset in this shell: export it "
            "and, if tmux started before the export, run `tmux kill-server`."
        )
    defaulted = not config.data.data_root
    data_root = (
        DEFAULT_DATA_ROOT if defaulted else Path(config.data.data_root)
    )
    print(f"device={device.type}  data_root={data_root}")

    # Fail here, with the resolved root in the message, rather than inside
    # read_tracks with a bare FileNotFoundError. The usual cause is a shell that
    # never had $DATA, in which case --data-root was not passed at all and this
    # silently fell back to the repo's own data/ directory.
    manifest_dir = data_root / "manifests" / config.data.train_split
    if not (manifest_dir / "tracks.jsonl").exists():
        available = sorted(
            d.name
            for d in (data_root / "manifests").glob("*")
            if (d / "tracks.jsonl").exists()
        )
        raise SystemExit(
            f"no manifest at {manifest_dir}/tracks.jsonl\n"
            + (
                "  --data-root was not passed, so this is the built-in default; "
                "$DATA was probably unset or your $COMMON/launch shell state is "
                "gone (a new tmux pane does not inherit it).\n"
                if defaulted
                else ""
            )
            + f"  splits present under {data_root}/manifests: "
            + (", ".join(available) if available else "none")
        )

    pair_loader, track_loader = build_train_loaders(config, data_root)
    if config.overfit_batches:
        print(
            f"overfit probe: {config.overfit_batches} fixed batches, no shuffling, "
            "no validation. Every loss term should approach 0; whatever does not "
            "is a term the model cannot fit even with the data memorized."
        )
    val_loader = None if config.overfit_batches else build_val_loader(config, data_root)
    steps_per_epoch = len(pair_loader)
    total_steps = config.max_steps or config.epochs * steps_per_epoch
    # --max-steps is a budget, not just a cap: a small training subset takes
    # more epochs to spend it, so subsets of different sizes are trained for
    # the same number of steps under the same LR schedule.
    n_epochs = max(config.epochs, math.ceil(total_steps / steps_per_epoch))
    print(
        f"train[{config.data.train_split}]: {len(pair_loader.dataset)} pairs, "
        f"{steps_per_epoch} steps/epoch, {total_steps} total steps "
        f"(up to {n_epochs} epochs)"
    )

    model = DisentanglementModel(config).to(device)
    # --converge-mert owns the layer mixes: they start uniform, stay trainable,
    # and the run exists to stop when they settle. Honouring --layer-weights or
    # --freeze-layer-weights here would answer a different question, and
    # launch_run.sh passes both on every invocation, so override rather than
    # ask the caller to drop them.
    if args.unfreeze_layer_weights or args.converge_mert:
        flag = "--converge-mert" if args.converge_mert else "--unfreeze-layer-weights"
        if config.layer_weights or config.freeze_layer_weights:
            print(
                f"{flag}: ignoring --layer-weights / --freeze-layer-weights; "
                f"the mixes start uniform and train"
            )
        config.layer_weights = ""
        config.freeze_layer_weights = False
        with torch.no_grad():
            model.content_mix.weights.zero_()  # softmax(zeros) = uniform 1/25
            model.style_mix.weights.zero_()
        model.content_mix.weights.requires_grad_(True)
        model.style_mix.weights.requires_grad_(True)
        print(f"{flag}: both layer mixes reset to uniform and unfrozen")
    elif args.resume is None and config.layer_weights:
        load_layer_weights(model, Path(config.layer_weights), config.loss.recon_pool)
        print(f"layer-mix weights loaded from {config.layer_weights}")
    if config.freeze_layer_weights:
        model.freeze_layer_weights()
        print("layer-mix weights frozen")
        if args.resume is None and not config.layer_weights:
            print(
                "  warning: frozen at their initialization, i.e. a uniform "
                "average of all layers; pass --layer-weights to freeze the "
                "phase-1 mixes instead"
            )
    mert = MertExtractor(config.mert).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable / 1e6:.1f}M")

    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config, total_steps)

    start_epoch, global_step = 0, 0
    prev_weights: dict[str, torch.Tensor] | None = None
    best: float | None = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["step"]
        prev_weights = checkpoint.get("prev_layer_weights")
        best = checkpoint.get("best_metric")
        print(f"resumed from {args.resume} (epoch {start_epoch}, step {global_step})")

    run_name = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(Path(config.log_dir) / run_name))
    writer.add_text("config", f"```json\n{config.to_dict()}\n```")
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # val_metrics.jsonl is opened in append mode so a resumed run continues one
    # file. That makes a *fresh* run into a used directory silently interleave
    # two trajectories, and run_info.json/best.pt describe whichever ran last.
    # Refuse instead: the usual cause is a forgotten --checkpoint-dir, which is
    # how checkpoints/ itself collected three runs (timeline.md, 2026-10-03).
    metrics_path = checkpoint_dir / "val_metrics.jsonl"
    if args.resume is None and metrics_path.exists():
        raise SystemExit(
            f"{metrics_path} already exists, so this directory holds another "
            f"run. Pass --checkpoint-dir checkpoints/<run-name> (what "
            f"scripts/launch_run.sh does), --resume to continue that run, or "
            f"move the directory aside."
        )
    write_run_info(checkpoint_dir, config, pair_loader, track_loader, total_steps)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    higher_is_better = any(tag in config.select_metric for tag in ("recall", "map"))
    # train losses summed since the last validation, for the train/val gap
    interval: dict[str, torch.Tensor | float] = defaultdict(float)
    interval_steps = 0
    last_val_step = -1

    def run_validation(epoch: int) -> None:
        """Validate, log train-vs-val, and keep best.pt on the select metric."""
        nonlocal best, interval_steps, last_val_step
        metrics = validate(mert, model, config, val_loader, device)
        record = {
            "step": global_step,
            "epoch": epoch,
            **metrics,
            **{
                f"train/{name}": float(value) / max(interval_steps, 1)
                for name, value in interval.items()
            },
        }
        interval.clear()
        interval_steps = 0
        last_val_step = global_step
        print("  " + "  ".join(f"{n}={v:.4f}" for n, v in metrics.items()))
        for name, value in metrics.items():
            writer.add_scalar(name, value, global_step)
        with open(checkpoint_dir / "val_metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        value = metrics.get(config.select_metric)
        if value is None:
            print(f"  warning: {config.select_metric} not among the validation metrics")
            return
        if best is None or (value > best if higher_is_better else value < best):
            best = value
            save_checkpoint(
                checkpoint_dir / "best.pt", model, optimizer, scheduler,
                config, epoch, global_step, prev_weights, best,
            )
            with open(checkpoint_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump(record, f, indent=1)
            print(f"  new best {config.select_metric}={value:.4f} -> best.pt")

    convergence = (
        LayerMixConvergence(
            tol=args.converge_tol,
            window=args.converge_window,
            patience=args.converge_patience,
        )
        if args.converge_mert or args.unfreeze_layer_weights
        else None
    )
    mixes_saved = False

    queue, momentum_encoder = None, None
    if config.loss.negative_queue > 0 and config.loss.contrastive_weight > 0:
        queue = ContentQueue(
            config.loss.negative_queue, config.bottleneck.token_dim
        ).to(device)
        # Mandatory, not optional: a queue without it collapses the content
        # representation, because the positive and the queued negatives then
        # come from different encoders and can be told apart by age.
        momentum_encoder = MomentumContentEncoder(
            model, config.loss.queue_momentum
        ).to(device)
        print(
            f"contrastive negatives: {config.data.batch_pairs * config.data.n_candidates - 1}"
            f" in-batch + up to {config.loss.negative_queue} queued; keys from a"
            f" momentum encoder (m={config.loss.queue_momentum})"
        )

    track_iter = repeat_forever(track_loader) if track_loader is not None else None
    model.train()
    done = False
    epoch = start_epoch
    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()
        if config.overfit_batches:
            # windows.py draws a fresh random anchor on every __getitem__, so
            # without this the "fixed" batches would hold new audio each epoch
            # and memorizing them would be impossible by construction.
            random.seed(config.seed)
        for pair_batch in pair_loader:
            track_batch = next(track_iter) if track_iter is not None else None
            waves, p, k = assemble_waves(pair_batch, track_batch, device)
            song_ids = pair_song_ids(pair_batch, device)

            with autocast:
                content_mix, style_mix = extract_mixes(
                    mert, model, waves, config.mert.micro_batch
                )
                content_target, style_target = pooled_targets(
                    config.loss, content_mix, style_mix
                )
                # An augmented A window is a contrastive view, never a
                # reconstruction target: fitting the decoder and the style
                # branch to it would teach P(style | content) that lowpassed,
                # saturated audio is ordinary human style.
                recon_index = None
                if config.data.augment.enabled and p > 0:
                    recon_index = torch.arange(p, waves.shape[0], device=device)
                model.content_std.update(
                    content_target if recon_index is None else content_target[recon_index]
                )
                model.style_std.update(
                    style_target if recon_index is None else style_target[recon_index]
                )
                content, style = model.encode_mixes(content_mix, style_mix)
                keys = None
                if momentum_encoder is not None and p > 0:
                    keys = momentum_encoder(content_mix[p : p + p * k])
                total, losses = compute_losses(
                    model,
                    config.loss,
                    content,
                    style,
                    content_target,
                    style_target,
                    p,
                    k,
                    song_ids,
                    queue.negatives() if queue is not None else None,
                    recon_index,
                    keys,
                )
            if queue is not None and keys is not None:
                # queue exactly what served as the positive, so the denominator
                # and the numerator are drawn from one distribution
                queue.push(keys, song_ids.repeat_interleave(k))

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.optim.grad_clip
            )
            optimizer.step()
            scheduler.step()
            if momentum_encoder is not None:
                momentum_encoder.update(model)
            global_step += 1
            # detached tensors, not floats: float() would sync the GPU every step
            interval["total"] += total.detach()
            for name, value in losses.items():
                interval[name] += value.detach()
            interval_steps += 1

            if global_step % config.log_every == 0 or global_step == 1:
                # .detach() before float(): these tensors are still attached to
                # the (already-consumed) graph and float() on them warns. Read
                # each one once — every float() of a CUDA tensor is a GPU sync.
                step_total = float(total.detach())
                scalars = {n: float(v.detach()) for n, v in losses.items()}
                parts = "  ".join(f"{n}={v:.4f}" for n, v in scalars.items())
                print(
                    f"epoch {epoch} step {global_step}/{total_steps}  "
                    f"total={step_total:.4f}  {parts}"
                )
                writer.add_scalar("train/total", step_total, global_step)
                for name, value in scalars.items():
                    writer.add_scalar(f"train/{name}", value, global_step)
                writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                writer.add_scalar(
                    "train/lr", scheduler.get_last_lr()[0], global_step
                )

            if val_loader is not None and config.val_every and (
                global_step % config.val_every == 0
            ):
                run_validation(epoch)
            if (
                config.checkpoint_every
                and global_step % config.checkpoint_every == 0
            ):
                save_checkpoint(
                    checkpoint_dir / "last.pt", model, optimizer, scheduler,
                    config, epoch, global_step, prev_weights, best,
                )
            if convergence is not None and global_step % args.converge_every == 0:
                mixes = model.layer_weight_summary()
                settled, line = convergence.update(mixes)
                print(f"  [layer mix @ {global_step}] {line}")
                if settled and not mixes_saved:
                    mixes_saved = True
                    for warning in convergence.warnings(mixes):
                        print(f"  warning: {warning}")
                    save_layer_weights(
                        checkpoint_dir / "phase1_layer_weights.pt",
                        model, config, global_step,
                    )
                    print(
                        f"layer mixes converged at step {global_step} -> "
                        f"{checkpoint_dir / 'phase1_layer_weights.pt'}"
                    )
                    # --converge-mert exists only to produce that file, so it
                    # stops here. --unfreeze-layer-weights is a normal training
                    # run that happens to also train the mixes: it keeps going,
                    # and the saved file records where they settled.
                    if args.converge_mert:
                        done = True
                        break
            if config.max_steps and global_step >= config.max_steps:
                done = True
                break

        print(f"epoch {epoch} finished in {time.time() - epoch_start:.0f}s")
        weights = model.layer_weight_summary()
        log_layer_weights(writer, weights, prev_weights, epoch)
        prev_weights = weights

        if val_loader is not None and not config.val_every:
            run_validation(epoch)

        save_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer, scheduler,
            config, epoch + 1, global_step, prev_weights, best,
        )
        if done:
            break

    # the final weights always get a validation point, whatever the cadence
    if val_loader is not None and last_val_step != global_step:
        run_validation(epoch)
        save_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer, scheduler,
            config, epoch + 1, global_step, prev_weights, best,
        )

    writer.close()
    print(f"done at step {global_step}; checkpoint: {checkpoint_dir / 'last.pt'}")
    if best is not None:
        print(f"best {config.select_metric}={best:.4f}: {checkpoint_dir / 'best.pt'}")

if __name__ == "__main__":
    main()
