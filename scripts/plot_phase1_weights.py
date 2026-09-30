#!/usr/bin/env python3
"""Plot the two MERT layer-mix distributions learned in phase 1.

The matplotlib version of what `inspect_phase1.py` prints as ASCII bars, for
the thesis. Two panels — content and style — each a weight per MERT hidden
state, with the uniform weight drawn to scale.

Two things this shows that the ASCII cannot:

* **A shared y-axis.** The ASCII normalises each branch to its own maximum, so
  a branch with far less contrast than the other still fills the width. Side by
  side on one scale, the difference in amplitude is the point (`timeline.md`,
  run #2: style's deviation from uniform is smaller than content's).
* **The uniform line to scale.** Softmax over 25 states starts at 0.04 and a
  vector still sitting there has selected nothing — condition 2 of the phase-1
  freeze criterion in `CLAUDE.md`. How far the bars stand off that line *is*
  the measurement.

`--deviation` plots w − uniform instead, which is the sensitive view: the plain
softmax vector is dominated by its uniform component, so two very different
mixes look nearly identical until you subtract it (`timeline.md`, run #2's
methodological finding).

Accepts either `phase1_layer_weights.pt` or any training checkpoint.

Examples:
    python scripts/plot_phase1_weights.py --checkpoint phase1_layer_weights.pt
    python scripts/plot_phase1_weights.py --checkpoint checkpoints/q-w25/best.pt \\
        --deviation --format pdf
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this also runs on an EC2 box with no display

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parent))

BRANCHES = ("content", "style")

# Chart surface and ink, painted explicitly rather than inherited from a
# matplotlib style, so the figure is identical on any machine
# (same convention as scripts/plot_mert_features.py).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

# Categorical slots 1 and 2, in fixed order: content is blue, style is orange,
# in every figure. Validated for this pair: adjacent CVD dE 24.7 (protan),
# normal-vision dE 33.6, contrast >= 3:1 against the surface.
HUES = {"content": "#2a78d6", "style": "#eb6834"}


def read_weights(path: Path) -> dict[str, torch.Tensor]:
    """Softmax layer weights per branch, from a weights file or a checkpoint.

    Reads the tensors directly instead of building a DisentanglementModel: the
    model is 127.7 M parameters and none of them are needed to draw 50 numbers.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    weights = {}
    for branch in BRANCHES:
        for key in (f"{branch}_softmax", f"{branch}_logits", f"{branch}_mix.weights"):
            if key in state:
                value = state[key].detach().float().flatten()
                # a "_softmax" entry is already normalised; logits are not
                weights[branch] = value if key.endswith("_softmax") else value.softmax(0)
                break
        else:
            raise SystemExit(
                f"{path}: no layer-mix weights for the {branch} branch "
                f"(looked for {branch}_softmax / {branch}_logits / "
                f"{branch}_mix.weights). Keys: {sorted(state)[:8]}…"
            )
    return weights


def describe(w: torch.Tensor) -> dict[str, float]:
    """The numbers the phase-1 freeze criterion is judged on."""
    n = w.numel()
    uniform = torch.full((n,), 1.0 / n)
    return {
        "entropy": float(-(w * w.clamp_min(1e-12).log()).sum()),
        "uniform_entropy": math.log(n),
        "max": float(w.max()),
        "min": float(w.min()),
        "ratio": float(w.max() / w.min().clamp_min(1e-12)),
        "cos_uniform": float(F.cosine_similarity(w, uniform, dim=0)),
        "deviation": float((w - uniform).norm()),
        "peak": int(w.argmax()),
    }


def draw(
    weights: dict[str, torch.Tensor], deviation: bool, source: str
) -> plt.Figure:
    n = next(iter(weights.values())).numel()
    uniform = 1.0 / n
    stats = {b: describe(w) for b, w in weights.items()}
    series = {
        b: (w - uniform if deviation else w).tolist() for b, w in weights.items()
    }

    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.6), sharey=True, facecolor=SURFACE
    )
    span = max(max(map(abs, v)) for v in series.values())
    for ax, branch in zip(axes, BRANCHES):
        s = stats[branch]
        ax.set_facecolor(SURFACE)
        # horizontal grid only, behind the bars, recessive
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.xaxis.grid(False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

        # width 0.78 leaves a gap between adjacent bars, so 25 of them read as
        # 25 values rather than as one filled block
        ax.bar(
            range(n), series[branch], width=0.78, color=HUES[branch],
            linewidth=0, zorder=3,
        )
        if deviation:
            ax.axhline(0.0, color=INK_2, linewidth=1.0, zorder=4)
        else:
            ax.axhline(
                uniform, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=4
            )
            ax.annotate(
                f"uniform {uniform:.3f}",
                xy=(n - 0.4, uniform), xytext=(0, 4), textcoords="offset points",
                ha="right", va="bottom", fontsize=8.5, color=INK_2,
            )
        # only the peak layer is labelled: a number on every bar is noise
        ax.annotate(
            f"layer {s['peak']}",
            xy=(s["peak"], series[branch][s["peak"]]),
            xytext=(0, 5), textcoords="offset points",
            ha="center", fontsize=8.5, color=INK_2,
        )
        ax.set_title(branch, loc="left", fontsize=12, color=INK, pad=14, weight="bold")
        ax.set_xlabel("MERT hidden state (0 = embedding)", fontsize=9.5, color=INK_2)
        ax.set_xlim(-0.8, n - 0.2)
        ax.set_xticks(range(0, n, 2))
        ax.tick_params(labelsize=8.5, colors=INK_2, length=0)
        ax.annotate(
            f"cos to uniform {s['cos_uniform']:.4f}   max/min {s['ratio']:.2f}×\n"
            f"entropy {s['entropy']:.4f} of {s['uniform_entropy']:.4f} nats"
            f"   ‖w − uniform‖ {s['deviation']:.4f}",
            xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 6),
            textcoords="offset points", fontsize=8.5, color=INK_2, va="bottom",
        )

    axes[0].set_ylabel(
        "weight − uniform" if deviation else "softmax weight",
        fontsize=9.5, color=INK_2,
    )
    if deviation:
        axes[0].set_ylim(-span * 1.35, span * 1.35)
    else:
        axes[0].set_ylim(0, span * 1.18)

    cos = float(
        F.cosine_similarity(weights["content"], weights["style"], dim=0)
    )
    deviations = {b: weights[b] - uniform for b in BRANCHES}
    cos_dev = float(
        F.cosine_similarity(deviations["content"], deviations["style"], dim=0)
    )
    figure.suptitle(
        "Phase-1 MERT layer mixes", x=0.008, ha="left", fontsize=14,
        color=INK, weight="bold", y=0.985,
    )
    figure.text(
        0.008, 0.915,
        f"cos(content, style) = {cos:.4f}   ·   between their deviations from "
        f"uniform = {cos_dev:.4f}   ·   {source}",
        ha="left", fontsize=9, color=INK_2,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("phase1_layer_weights.pt")
    )
    parser.add_argument("--out", type=Path, help="default: next to the checkpoint")
    parser.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--deviation",
        action="store_true",
        help="plot w - uniform instead of w. The softmax vector is dominated "
        "by its uniform component, so this is the view in which two different "
        "mixes actually look different",
    )
    args = parser.parse_args()

    weights = read_weights(args.checkpoint)
    figure = draw(weights, args.deviation, source=str(args.checkpoint))
    suffix = "_deviation" if args.deviation else ""
    out = args.out or args.checkpoint.with_name(
        f"{args.checkpoint.stem}_layer_mix{suffix}.{args.format}"
    )
    figure.savefig(out, dpi=args.dpi, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")
    for branch, w in weights.items():
        s = describe(w)
        print(
            f"  {branch:8s} peak layer {s['peak']:2d}  max {s['max']:.4f}  "
            f"cos to uniform {s['cos_uniform']:.6f}  ‖dev‖ {s['deviation']:.4f}"
        )


if __name__ == "__main__":
    main()
