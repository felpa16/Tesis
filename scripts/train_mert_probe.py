#!/usr/bin/env python3
"""Supervised linear probe on pooled MERT, with one MoM generator held out.

A diagnostic, not a detector (experiments.md, section 13). The MERT-only flow
and its closed-form Gaussian land at the same AUROC, which points at the
features rather than the density model as the ceiling. This probe asks the
question directly: with labels, how well do time-pooled MERT features separate
human music from a generator never seen in training? See
src/models/mert_probe.py for the model: 25 trainable softmax layer weights over
per-layer standardized pooled hidden states, then one linear layer.

Data:
  human       --train-cache   SHS100K train windows (scripts/cache_pooled_mert.py)
  synthetic   --mom-cache     MoM windows (scripts/cache_pooled_mom.py): every
                              generator except --leave-out, split by song into
                              training and validation (--val-fraction)
  validation  --val-cache     val50 human windows, against the held-out songs of
                              the training generators

The best epoch is chosen on that validation set, so neither test50 nor the
left-out generator may enter it: test50 is refused as --val-cache, and the
left-out generator is never loaded. Batches are half human, half synthetic,
with the synthetic half balanced across the training generators, so neither a
class nor a generator dominates the gradient.

Each epoch appends one line to {checkpoint-dir}/history.jsonl (balanced BCE
and window-level AUROC on validation and on a fixed training subset of the
same size; per-generator validation AUROC; the 25 weights), and the run ends by
drawing loss_curves.png. best.pt is the epoch with the highest validation
AUROC; last.pt is the final one.

Then score test50 against the left-out generator:
    python scripts/evaluate_mert_probe.py --checkpoint {checkpoint-dir}/best.pt

Caveat that applies to every number this produces: all MoM audio is mp3 and
all SHS100K audio is YouTube opus/m4a. A supervised probe will use a codec
difference if one survives into MERT, and the left-out generator is mp3 too,
so a high held-out AUROC is an upper bound on generative artifacts plus
whatever codec signal survives, not on generative artifacts alone.

Example:
    python scripts/train_mert_probe.py --leave-out udio \\
        --train-cache $CACHE/train --val-cache $CACHE/val50 --mom-cache $CACHE/mom
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

import evaluate_mom as em  # noqa: E402
from plot_training_curves import plot_run  # noqa: E402
from src.data.pooled_cache import PooledCache  # noqa: E402
from src.models.mert_flow import STREAMS, load_mix_logits  # noqa: E402
from src.models.mert_probe import MertProbe  # noqa: E402
from src.training import pick_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--leave-out", required=True, choices=em.GENERATORS,
                        help="generator kept out of training, for evaluate_mert_probe.py")
    parser.add_argument("--train-cache", type=Path, required=True, help="human training windows")
    parser.add_argument("--val-cache", type=Path, required=True, help="human val50 windows")
    parser.add_argument("--mom-cache", type=Path, required=True, help="scripts/cache_pooled_mom.py")
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="default: checkpoints/mert-probe-no-{leave-out}")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=512, help="half human, half synthetic")
    parser.add_argument("--lr", type=float, default=1e-3, help="linear layer learning rate")
    parser.add_argument("--mix-lr", type=float, default=1e-2, help="layer-weight learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="on the linear weights only")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="share of each training generator's songs held out for validation")
    parser.add_argument("--val-max-windows", type=int, default=0, help="0 = whole val cache")
    parser.add_argument("--stats-windows", type=int, default=50000,
                        help="human training windows the per-layer standardization is fitted on")
    parser.add_argument("--mix-checkpoint", type=Path, default=None,
                        help="start the 25 weights from this checkpoint's --mix vector "
                             "(default: uniform)")
    parser.add_argument("--mix", choices=STREAMS, default="content")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def check_compatible(caches: dict[str, PooledCache]) -> None:
    """All three caches must hold the same kind of window."""
    first_name, first = next(iter(caches.items()))
    for name, cache in caches.items():
        for key in ("window_seconds", "model_name"):
            if cache.meta.get(key) != first.meta.get(key):
                raise SystemExit(f"{name} has {key}={cache.meta.get(key)!r} but "
                                 f"{first_name} has {first.meta.get(key)!r}")
        if (cache.n_layers, cache.dim) != (first.n_layers, first.dim):
            raise SystemExit(f"{name} rows are {(cache.n_layers, cache.dim)}, "
                             f"{first_name} rows are {(first.n_layers, first.dim)}")


def split_songs(
    rows: list[dict], generators: list[str], val_fraction: float, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Row indices per generator, split by song: all windows of a song land on one side."""
    by_song: dict[str, dict[str, list[int]]] = {g: {} for g in generators}
    for i, row in enumerate(rows):
        if row["source"] in by_song:
            by_song[row["source"]].setdefault(row["song"], []).append(i)
    rng = np.random.default_rng(seed)
    train, val = {}, {}
    for g in generators:
        songs = sorted(by_song[g])
        if len(songs) < 2:
            raise SystemExit(f"--mom-cache holds {len(songs)} {g} songs; need at least 2")
        order = rng.permutation(len(songs))
        n_val = min(max(1, round(val_fraction * len(songs))), len(songs) - 1)
        held = {songs[k] for k in order[:n_val]}
        train[g] = np.array([i for s in songs if s not in held for i in by_song[g][s]])
        val[g] = np.array([i for s in songs if s in held for i in by_song[g][s]])
    return train, val


@torch.no_grad()
def fit_layer_norm(
    cache: PooledCache, indices: np.ndarray, device: torch.device, chunk: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-layer, per-dimension mean and std over `indices`, accumulated in fp64."""
    total = torch.zeros(cache.n_layers, cache.dim, dtype=torch.float64, device=device)
    squares = torch.zeros_like(total)
    for start in range(0, len(indices), chunk):
        x = torch.from_numpy(cache.gather(indices[start : start + chunk])).to(device, torch.float64)
        total += x.sum(0)
        squares += (x**2).sum(0)
    n = len(indices)
    mean = total / n
    var = (squares / n - mean**2).clamp_min(0) * n / max(n - 1, 1)
    return mean.float(), var.sqrt().clamp_min(1e-6).float()


@torch.no_grad()
def logits_of(model: MertProbe, cache: PooledCache, indices: np.ndarray, device, batch: int = 2048) -> np.ndarray:
    was_training = model.training
    model.eval()
    out = [
        model(torch.from_numpy(cache.gather(indices[start : start + batch])).to(device)).float().cpu()
        for start in range(0, len(indices), batch)
    ]
    model.train(was_training)
    return torch.cat(out).numpy() if out else np.zeros(0, dtype=np.float32)


def separation(human: np.ndarray, synthetic: np.ndarray) -> dict[str, float]:
    """Class-balanced BCE and AUROC (human scores higher, as in evaluate_mom.py)."""
    bce = 0.5 * (np.logaddexp(0.0, human).mean() + np.logaddexp(0.0, -synthetic).mean())
    return {"bce": float(bce), "auroc": em.auroc(-human, -synthetic)}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    checkpoint_dir = args.checkpoint_dir or Path("checkpoints") / f"mert-probe-no-{args.leave_out}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    human = PooledCache(args.train_cache)
    val_human = PooledCache(args.val_cache)
    mom = PooledCache(args.mom_cache)
    if val_human.meta.get("split") == "test50":
        raise SystemExit("--val-cache is test50, the split evaluate_mert_probe.py reports on; "
                         "pass the val50 cache")
    check_compatible({"--train-cache": human, "--val-cache": val_human, "--mom-cache": mom})
    generators = [g for g in mom.meta["generators"] if g != args.leave_out]
    if not generators:
        raise SystemExit(f"--mom-cache holds no generator other than {args.leave_out}")
    if args.leave_out not in mom.meta["generators"]:
        print(f"note: {args.leave_out} is not in --mom-cache; every cached generator trains")

    ai_train, ai_val = split_songs(mom.rows(), generators, args.val_fraction, args.seed)
    val_human_rows = np.arange(min(args.val_max_windows or len(val_human), len(val_human)))
    ai_val_all = np.concatenate([ai_val[g] for g in generators])
    eval_rng = np.random.default_rng(args.seed + 1)
    train_eval_human = np.sort(eval_rng.choice(len(human), min(len(val_human_rows), len(human)), replace=False))
    ai_train_all = np.concatenate([ai_train[g] for g in generators])
    train_eval_ai = np.sort(eval_rng.choice(ai_train_all, min(len(ai_val_all), len(ai_train_all)), replace=False))
    print(f"human: {len(human)} training windows, {len(val_human_rows)} validation windows "
          f"({val_human.meta.get('split')})")
    for g in generators:
        print(f"{g}: {len(ai_train[g])} training windows, {len(ai_val[g])} validation windows")
    print(f"left out: {args.leave_out}")

    model = MertProbe(human.n_layers, human.dim).to(device)
    stats_rows = np.sort(rng.choice(len(human), min(args.stats_windows, len(human)), replace=False))
    mean, std = fit_layer_norm(human, stats_rows, device)
    with torch.no_grad():
        model.layer_mean.copy_(mean)
        model.layer_std.copy_(std)
        if args.mix_checkpoint is not None:
            model.logits.copy_(load_mix_logits(args.mix_checkpoint)[args.mix].to(device))
    print(f"layer standardization fitted on {len(stats_rows)} human windows; "
          f"mix starts {'uniform' if args.mix_checkpoint is None else f'at {args.mix_checkpoint} ({args.mix})'}")

    optimizer = torch.optim.AdamW([
        {"params": [model.head.weight], "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": [model.head.bias], "lr": args.lr, "weight_decay": 0.0},
        {"params": [model.logits], "lr": args.mix_lr, "weight_decay": 0.0},
    ])
    total_steps = args.epochs * args.steps_per_epoch
    warmup = min(100, total_steps // 10)

    def lr_lambda(step: int) -> float:  # linear warmup, cosine decay to 10 %
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = min((step - warmup) / max(total_steps - warmup, 1), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    half = args.batch_size // 2

    def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
        human_idx = np.sort(rng.integers(0, len(human), half))
        picks = rng.integers(0, len(generators), half)
        ai_idx = np.concatenate([
            ai_train[g][rng.integers(0, len(ai_train[g]), int((picks == k).sum()))]
            for k, g in enumerate(generators)
        ])
        x = np.concatenate([human.gather(human_idx), mom.gather(ai_idx)])
        y = np.concatenate([np.zeros(half), np.ones(len(ai_idx))])
        return torch.from_numpy(x).to(device), torch.from_numpy(y).float().to(device)

    history_path = checkpoint_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)

    def record(epoch: int, step: int, running: float | None) -> dict:
        h_val = logits_of(model, val_human, val_human_rows, device)
        s_val = {g: logits_of(model, mom, ai_val[g], device) for g in generators}
        rec: dict = {"epoch": epoch, "step": step}
        if running is not None:
            rec["train_running/bce"] = running
        train_sep = separation(logits_of(model, human, train_eval_human, device),
                               logits_of(model, mom, train_eval_ai, device))
        val_sep = separation(h_val, np.concatenate([s_val[g] for g in generators]))
        rec.update({f"train/{k}": v for k, v in train_sep.items()})
        rec.update({f"val/{k}": v for k, v in val_sep.items()})
        for g in generators:
            rec[f"val/auroc/{g}"] = em.auroc(-h_val, -s_val[g])
        weights = model.weights().detach().cpu()
        rec["mix/weights"] = [round(float(w), 6) for w in weights]
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        top = torch.argsort(weights, descending=True)[:3].tolist()
        per_gen = "  ".join(f"{g} {rec[f'val/auroc/{g}']:.4f}" for g in generators)
        print(f"epoch {epoch}: bce train {rec['train/bce']:.4f} val {rec['val/bce']:.4f}  "
              f"auroc train {rec['train/auroc']:.4f} val {rec['val/auroc']:.4f} ({per_gen})  "
              f"top layers {', '.join(f'{i}:{weights[i]:.3f}' for i in top)}", flush=True)
        return rec

    def save(path: Path, epoch: int, step: int, rec: dict) -> None:
        torch.save(
            {
                "kind": "mert_probe",
                "model": model.state_dict(),
                "n_layers": human.n_layers,
                "dim": human.dim,
                "leave_out": args.leave_out,
                "train_generators": generators,
                "epoch": epoch,
                "step": step,
                "val": {k: v for k, v in rec.items() if k.startswith("val/")},
                "window_seconds": float(human.meta["window_seconds"]),
                "model_name": human.meta["model_name"],
                "data_root": human.meta.get("data_root"),
                "train_cache": str(args.train_cache),
                "val_cache": str(args.val_cache),
                "mom_cache": str(args.mom_cache),
                "mom": {k: mom.meta.get(k) for k in ("bucket", "region", "mom_prefix", "mom_root",
                                                     "skip", "seed")},
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
            path,
        )

    step = 0
    best = -math.inf
    record(0, 0, None)
    model.train()
    for epoch in range(1, args.epochs + 1):
        started, loss_sum = time.time(), 0.0
        for _ in range(args.steps_per_epoch):
            x, y = sample_batch()
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach())
            step += 1
        rec = record(epoch, step, loss_sum / args.steps_per_epoch)
        save(checkpoint_dir / "last.pt", epoch, step, rec)
        if rec["val/auroc"] > best:
            best = rec["val/auroc"]
            save(checkpoint_dir / "best.pt", epoch, step, rec)
            print(f"  best so far -> {checkpoint_dir / 'best.pt'}")
        print(f"  epoch took {time.time() - started:.0f}s")
    plot_run(checkpoint_dir)
    print(f"done. Next: python scripts/evaluate_mert_probe.py --checkpoint {checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
