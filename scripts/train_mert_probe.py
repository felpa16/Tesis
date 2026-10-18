#!/usr/bin/env python3
"""Supervised linear probes on pooled MERT, one per MoM generator.

A diagnostic, not a detector (experiments.md, section 13). The MERT-only flow
and its closed-form Gaussian land at the same AUROC, which points at the
features rather than the density model as the ceiling. These probes ask how
well time-pooled MERT features separate human music from each generator when
labels are available. See src/models/mert_probe.py for the model: 25
trainable softmax layer weights over per-layer standardized pooled hidden
states, then one linear layer.

One probe per --train-generators entry. An entry is a generator, or a
comma-separated set trained as one probe: the default trains udio, riffusion,
diffrhythm and yue separately, which evaluate_mert_probe.py turns into a 4 x 4
matrix (rows: the probe's generator, columns: the generator scored). The
diagonal asks whether human and that generator are separable at all, on songs
the probe never saw; the off-diagonal cells ask whether one generator's cues
transfer to another. "riffusion,diffrhythm,yue" is the leave-udio-out probe and
"udio,riffusion,diffrhythm,yue" the joint one.

Data:
  human       --train-cache   SHS100K train windows (scripts/cache_pooled_mert.py)
  synthetic   --mom-cache     MoM windows (scripts/cache_pooled_mom.py), each
                              generator split by song into training and
                              validation (--val-fraction); the split of a
                              generator is the same in every probe
  validation  --val-cache     val50 human windows, against the held-out songs of
                              the probe's own generators

Each probe's best epoch is chosen on its own generators' validation songs
only, and test50 is refused as --val-cache. The other generators' validation
songs are scored every epoch as transfer/auroc/{g}, to watch how transfer moves
as the probe fits; they never choose an epoch. Batches are half human, half
synthetic, with the synthetic half balanced across the probe's generators.

Codec. All MoM audio is mp3, each generator with its own settings; all SHS100K
audio is YouTube opus/AAC. A probe on caches built without --codec can score
well by recognizing the mp3. Build all three caches with one --codec chain
(e.g. mp3:64k) to give both classes the same last codec: this script refuses
caches whose chains differ, records the chain in every checkpoint, and by
default puts the probes in checkpoints/mert-probe-{chain}. The counterfactual
check that asks a trained probe whether it reads the codec is in
evaluate_mert_probe.py (--codec-check).

Each probe gets {checkpoint-root}/{generators}/ with history.jsonl (balanced
BCE and window-level AUROC on validation and on a fixed training subset of the
same size; per-generator validation AUROC; transfer AUROC; the 25 weights),
loss_curves.png, best.pt (highest validation AUROC) and last.pt.

Examples:
    # original audio
    python scripts/train_mert_probe.py \\
        --train-cache $CACHE/train --val-cache $CACHE/val50 --mom-cache $CACHE/mom
    # common codec
    python scripts/train_mert_probe.py --train-cache $CACHE/train-mp3-64k \\
        --val-cache $CACHE/val50-mp3-64k --mom-cache $CACHE/mom-mp3-64k
    # then
    python scripts/evaluate_mert_probe.py --checkpoints checkpoints/mert-probe/*/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

import evaluate_mom as em  # noqa: E402
from plot_training_curves import plot_run  # noqa: E402
from src.data.codec import chain_slug, parse_chain  # noqa: E402
from src.data.pooled_cache import PooledCache  # noqa: E402
from src.models.mert_flow import STREAMS, load_mix_logits  # noqa: E402
from src.models.mert_probe import MertProbe  # noqa: E402
from src.training import pick_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train-generators", nargs="+", default=list(em.GENERATORS),
                        help="one probe per entry; an entry is a generator or a comma-separated "
                             "set (default: each generator alone)")
    parser.add_argument("--train-cache", type=Path, required=True, help="human training windows")
    parser.add_argument("--val-cache", type=Path, required=True, help="human val50 windows")
    parser.add_argument("--mom-cache", type=Path, required=True, help="scripts/cache_pooled_mom.py")
    parser.add_argument("--checkpoint-root", type=Path,
                        help="default: checkpoints/mert-probe, plus -{codec chain} if the caches have one")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=512, help="half human, half synthetic")
    parser.add_argument("--lr", type=float, default=1e-3, help="linear layer learning rate")
    parser.add_argument("--mix-lr", type=float, default=1e-2, help="layer-weight learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="on the linear weights only")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="share of each generator's songs held out for validation")
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
    """All three caches must hold the same kind of window, through the same codec."""
    first_name, first = next(iter(caches.items()))
    for name, cache in caches.items():
        for key in ("window_seconds", "model_name", "codec"):
            if cache.meta.get(key) != first.meta.get(key):
                raise SystemExit(f"{name} has {key}={cache.meta.get(key)!r} but "
                                 f"{first_name} has {first.meta.get(key)!r}")
        if (cache.n_layers, cache.dim) != (first.n_layers, first.dim):
            raise SystemExit(f"{name} rows are {(cache.n_layers, cache.dim)}, "
                             f"{first_name} rows are {(first.n_layers, first.dim)}")


def probe_specs(entries: list[str], available: list[str]) -> list[list[str]]:
    """--train-generators entries -> generator lists, in the cache's order."""
    specs: list[list[str]] = []
    for entry in entries:
        generators = [g.strip() for g in entry.split(",") if g.strip()]
        unknown = [g for g in generators if g not in available]
        if not generators or unknown:
            raise SystemExit(f"--train-generators {entry!r}: {', '.join(unknown) or 'empty'} not in "
                             f"--mom-cache ({', '.join(available)})")
        specs.append(sorted(set(generators), key=available.index))
    names = ["+".join(s) for s in specs]
    if len(set(names)) != len(names):
        raise SystemExit(f"--train-generators names a probe twice: {' '.join(entries)}")
    return specs


def split_songs(
    rows: list[dict], generators: list[str], val_fraction: float, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Row indices per generator, split by song: all windows of a song land on one
    side. Each generator is shuffled on its own, so its split is the same
    whichever probes are trained."""
    by_song: dict[str, dict[str, list[int]]] = {g: {} for g in generators}
    for i, row in enumerate(rows):
        if row["source"] in by_song:
            by_song[row["source"]].setdefault(row["song"], []).append(i)
    train, val = {}, {}
    for g in generators:
        songs = sorted(by_song[g])
        if len(songs) < 2:
            raise SystemExit(f"--mom-cache holds {len(songs)} {g} songs; need at least 2")
        random.Random(f"{seed}:{g}").shuffle(songs)
        n_val = min(max(1, round(val_fraction * len(songs))), len(songs) - 1)
        held = set(songs[:n_val])
        train[g] = np.array(sorted(i for s in songs if s not in held for i in by_song[g][s]))
        val[g] = np.array(sorted(i for s in held for i in by_song[g][s]))
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


@dataclass
class ProbeData:
    """Everything the probes share: caches, the song split, the layer statistics."""

    human: PooledCache
    val_human: PooledCache
    mom: PooledCache
    ai_train: dict[str, np.ndarray]
    ai_val: dict[str, np.ndarray]
    val_human_rows: np.ndarray
    layer_mean: torch.Tensor
    layer_std: torch.Tensor
    init_logits: torch.Tensor | None
    codec: str | None


def train_probe(
    generators: list[str], data: ProbeData, checkpoint_dir: Path, args: argparse.Namespace, device
) -> dict:
    """Train one probe on `generators`; returns its best epoch and validation AUROC."""
    name = checkpoint_dir.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    human, val_human, mom = data.human, data.val_human, data.mom
    others = [g for g in data.ai_val if g not in generators]
    torch.manual_seed(args.seed)
    rng = np.random.default_rng([args.seed, zlib.crc32(name.encode())])
    eval_rng = np.random.default_rng([args.seed + 1, zlib.crc32(name.encode())])

    ai_val_all = np.concatenate([data.ai_val[g] for g in generators])
    ai_train_all = np.concatenate([data.ai_train[g] for g in generators])
    n_eval = len(data.val_human_rows)
    train_eval_human = np.sort(eval_rng.choice(len(human), min(n_eval, len(human)), replace=False))
    train_eval_ai = np.sort(eval_rng.choice(ai_train_all, min(len(ai_val_all), len(ai_train_all)), replace=False))
    print(f"\n=== probe {name}: trains on {', '.join(generators)}"
          + (f"; transfer watched on {', '.join(others)}" if others else "") + " ===")
    for g in generators:
        print(f"{g}: {len(data.ai_train[g])} training windows, {len(data.ai_val[g])} validation windows")

    model = MertProbe(human.n_layers, human.dim).to(device)
    with torch.no_grad():
        model.layer_mean.copy_(data.layer_mean)
        model.layer_std.copy_(data.layer_std)
        if data.init_logits is not None:
            model.logits.copy_(data.init_logits.to(device))

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
            data.ai_train[g][rng.integers(0, len(data.ai_train[g]), int((picks == k).sum()))]
            for k, g in enumerate(generators)
        ])
        x = np.concatenate([human.gather(human_idx), mom.gather(ai_idx)])
        y = np.concatenate([np.zeros(half), np.ones(len(ai_idx))])
        return torch.from_numpy(x).to(device), torch.from_numpy(y).float().to(device)

    history_path = checkpoint_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)

    def record(epoch: int, step: int, running: float | None) -> dict:
        h_val = logits_of(model, val_human, data.val_human_rows, device)
        s_val = {g: logits_of(model, mom, data.ai_val[g], device) for g in data.ai_val}
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
        for g in others:
            rec[f"transfer/auroc/{g}"] = em.auroc(-h_val, -s_val[g])
        weights = model.weights().detach().cpu()
        rec["mix/weights"] = [round(float(w), 6) for w in weights]
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        top = torch.argsort(weights, descending=True)[:3].tolist()
        transfer = "  ".join(f"{g} {rec[f'transfer/auroc/{g}']:.3f}" for g in others)
        print(f"epoch {epoch}: bce train {rec['train/bce']:.4f} val {rec['val/bce']:.4f}  "
              f"auroc train {rec['train/auroc']:.4f} val {rec['val/auroc']:.4f}"
              + (f"  transfer: {transfer}" if others else "")
              + f"  top layers {', '.join(f'{i}:{weights[i]:.3f}' for i in top)}", flush=True)
        return rec

    def save(path: Path, epoch: int, step: int, rec: dict) -> None:
        torch.save(
            {
                "kind": "mert_probe",
                "name": name,
                "model": model.state_dict(),
                "n_layers": human.n_layers,
                "dim": human.dim,
                "train_generators": generators,
                "codec": data.codec,
                "epoch": epoch,
                "step": step,
                "val": {k: v for k, v in rec.items() if k.startswith(("val/", "transfer/"))},
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
    best, best_epoch = -math.inf, 0
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
            best, best_epoch = rec["val/auroc"], epoch
            save(checkpoint_dir / "best.pt", epoch, step, rec)
            print(f"  best so far -> {checkpoint_dir / 'best.pt'}")
        print(f"  epoch took {time.time() - started:.0f}s")
    plot_run(checkpoint_dir)
    return {"best_epoch": best_epoch, "best_val_auroc": best, "path": checkpoint_dir / "best.pt"}


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    human = PooledCache(args.train_cache)
    val_human = PooledCache(args.val_cache)
    mom = PooledCache(args.mom_cache)
    if val_human.meta.get("split") == "test50":
        raise SystemExit("--val-cache is test50, the split evaluate_mert_probe.py reports on; "
                         "pass the val50 cache")
    check_compatible({"--train-cache": human, "--val-cache": val_human, "--mom-cache": mom})
    codec = human.meta.get("codec")
    available = list(mom.meta["generators"])
    specs = probe_specs(args.train_generators, available)
    root = args.checkpoint_root or Path("checkpoints") / (
        "mert-probe" + (f"-{chain_slug(parse_chain(codec))}" if codec else ""))

    ai_train, ai_val = split_songs(mom.rows(), available, args.val_fraction, args.seed)
    val_human_rows = np.arange(min(args.val_max_windows or len(val_human), len(val_human)))
    print(f"human: {len(human)} training windows, {len(val_human_rows)} validation windows "
          f"({val_human.meta.get('split')}); codec {codec or 'none (original audio)'}")
    if codec is None:
        print("note: no codec chain, so a probe can learn each generator's mp3 settings; "
              "see --codec-check in evaluate_mert_probe.py")

    stats_rng = np.random.default_rng(args.seed)
    stats_rows = np.sort(stats_rng.choice(len(human), min(args.stats_windows, len(human)), replace=False))
    mean, std = fit_layer_norm(human, stats_rows, device)
    init_logits = load_mix_logits(args.mix_checkpoint)[args.mix] if args.mix_checkpoint is not None else None
    print(f"layer standardization fitted on {len(stats_rows)} human windows; "
          f"mix starts {'uniform' if init_logits is None else f'at {args.mix_checkpoint} ({args.mix})'}")
    data = ProbeData(human, val_human, mom, ai_train, ai_val, val_human_rows, mean, std, init_logits, codec)

    results = {}
    for generators in specs:
        name = "+".join(generators)
        results[name] = train_probe(generators, data, root / name, args, device)

    print(f"\n{'probe':<36} best epoch  val AUROC")
    for name, result in results.items():
        print(f"{name:<36} {result['best_epoch']:>10}  {result['best_val_auroc']:.4f}")
    paths = " ".join(str(r["path"]) for r in results.values())
    print(f"done. Next: python scripts/evaluate_mert_probe.py --checkpoints {paths}")


if __name__ == "__main__":
    main()
