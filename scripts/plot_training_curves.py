#!/usr/bin/env python3
"""Train-vs-val loss curves (and per-epoch detection metrics) of a training run.

Reads the per-validation history a training script leaves in its checkpoint
directory and draws one panel per loss term, train and val on the same axes:

    history.jsonl       scripts/train_mert_flow.py, one line per epoch
    val_metrics.jsonl   scripts/train.py, one line per validation

A term is drawn when the history holds both train/{term} and val/{term}.
train_mert_flow.py measures both sides the same way (eval mode, fixed window
sets). train.py's train side is the running mean of the training losses since
the previous validation, in train mode: a real gap is visible, but it is not a
like-for-like comparison. In particular its contrastive term sees the negative
queue in training and only in-batch negatives in validation, so train
contrastive sits *above* val by construction.

When the history also holds metrics/* entries (train_mert_flow.py
--track-metrics), a second figure shows the MoM AUROC per epoch.

Both training scripts call this at the end of a run; it can also be run by hand
on any finished (or still running) run:

    python scripts/plot_training_curves.py checkpoints/mert-flow-frozen
    python scripts/plot_training_curves.py checkpoints/mqa-w100
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Same chart surface and ink as scripts/evaluate_mom.py; series take the
# categorical slots in fixed order (train = 1, val = 2; generators 1-4).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
HISTORY_FILES = ("history.jsonl", "val_metrics.jsonl")
GENERATOR_ORDER = ("udio", "riffusion", "diffrhythm", "yue")  # evaluate_mom.GENERATORS


def read_history(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_history(run_dir: Path) -> Path:
    for name in HISTORY_FILES:
        if (run_dir / name).exists():
            return run_dir / name
    raise SystemExit(f"no {' or '.join(HISTORY_FILES)} in {run_dir}")


def paired_terms(records: list[dict]) -> list[str]:
    """Loss terms with both a train/ and a val/ value, total first."""
    keys = {k for r in records for k in r}
    terms = sorted(k[len("val/"):] for k in keys if k.startswith("val/") and f"train/{k[4:]}" in keys)
    return sorted(terms, key=lambda t: (t not in ("total", "nll"), t))


def _style(ax, x_key: str) -> None:
    from matplotlib.ticker import MaxNLocator

    ax.set_facecolor(SURFACE)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(x_key, color=INK_2, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=3)


def _series(records: list[dict], x_key: str, key: str) -> tuple[list[float], list[float]]:
    points = [(r[x_key], r[key]) for r in records if key in r and r[key] is not None and math.isfinite(r[key])]
    return [p[0] for p in points], [p[1] for p in points]


def plot_losses(records: list[dict], out: Path, x_key: str, title: str, dpi: int = 160) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")  # headless: this runs on an EC2 box with no display
    import matplotlib.pyplot as plt

    terms = paired_terms(records)
    if not terms:
        return None
    cols = min(3, len(terms))
    rows = math.ceil(len(terms) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.6 * cols, 3.2 * rows + 0.9), squeeze=False, facecolor=SURFACE
    )
    for ax, term in zip(axes.flat, terms):
        _style(ax, x_key)
        for side, color in (("train", SERIES[0]), ("val", SERIES[1])):
            xs, ys = _series(records, x_key, f"{side}/{term}")
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4, label=side)
        ax.set_title(term, color=INK, fontsize=11, loc="left")
    for ax in list(axes.flat)[len(terms):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2, fontsize=10,
               labelcolor=INK)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_metrics(records: list[dict], out: Path, x_key: str, title: str, dpi: int = 160) -> Path | None:
    """AUROC (human vs. each generator, and pooled) per epoch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = {k for r in records for k in r}
    found = {k.split("/")[2] for k in keys if k.startswith("metrics/auroc/") and k.endswith("/flow")} - {"all"}
    # fixed order, so a generator keeps its color from run to run
    generators = [g for g in GENERATOR_ORDER if g in found] + sorted(found - set(GENERATOR_ORDER))
    if not generators and "metrics/auroc/all/flow" not in keys:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE)
    _style(ax, x_key)
    for i, generator in enumerate(generators):
        xs, ys = _series(records, x_key, f"metrics/auroc/{generator}/flow")
        ax.plot(xs, ys, color=SERIES[i % len(SERIES)], linewidth=2, marker="o", markersize=4,
                label=generator)
    xs, ys = _series(records, x_key, "metrics/auroc/all/flow")
    ax.plot(xs, ys, color=INK, linewidth=2.6, marker="o", markersize=5, label="all AI (flow)")
    xs, ys = _series(records, x_key, "metrics/auroc/all/gaussian")
    if xs:
        ax.plot(xs, ys, color=INK_2, linewidth=1.5, linestyle=(0, (4, 3)),
                label="all AI (Gaussian)")
    ax.set_ylabel("AUROC (human scores higher)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="center left", bbox_to_anchor=(1.01, 0.5))
    human = next((r.get("metrics/human_split") for r in records if r.get("metrics/human_split")), None)
    subtitle = f"human = {human}, " if human else ""
    ax.set_title(f"{subtitle}min over windows; epoch 0 = before training",
                 color=INK_2, fontsize=9, loc="left")
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_run(run_dir: Path, x_key: str | None = None) -> list[Path]:
    """Draw every figure the run's history supports; returns the files written.

    Never raises on a plotting problem (a broken matplotlib must not cost a
    finished training run its exit status); it prints instead.
    """
    try:
        path = find_history(run_dir)
        records = read_history(path)
        if not records:
            print(f"plot: {path} is empty")
            return []
        x_key = x_key or ("epoch" if path.name == "history.jsonl" else "step")
        written = [
            plot_losses(records, run_dir / "loss_curves.png", x_key,
                        f"{run_dir.name}: train vs. val loss"),
            plot_metrics(records, run_dir / "metric_curves.png", x_key,
                         f"{run_dir.name}: detection AUROC during training"),
        ]
        written = [w for w in written if w is not None]
        for w in written:
            print(f"plot: {w}")
        return written
    except Exception as exc:  # noqa: BLE001
        print(f"plot: failed ({type(exc).__name__}: {exc}); "
              f"rerun with python scripts/plot_training_curves.py {run_dir}")
        return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path, help="checkpoint directory of the run")
    parser.add_argument("--x", choices=("epoch", "step"), help="x axis (default: per script)")
    args = parser.parse_args()
    if not plot_run(args.run_dir, args.x):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
