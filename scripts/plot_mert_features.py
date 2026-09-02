#!/usr/bin/env python3
"""Project cached MERT window features to 2D and plot them.

Reads the .npy files written by scripts/extract_mert_features.py -- downloading
anything not already in the local cache -- reduces them to two dimensions with
PCA, and plots them. Two modes:

    --mode combined   one plot, content and style in different colors
    --mode separate   two panels (content, style), colored by song, so covers
                      of the same composition share a color

The question the separate mode exists to answer is whether covers of the same
song land near each other. Because a 2D projection of a handful of points can
show clustering that is not really there, the script also prints the two
numbers that settle it in the full 1024-d space: within- versus between-song
cosine distance, and nearest-neighbour cover retrieval.

A cached file is a (N, 1024) fp16 *sequence* -- one row per MERT frame, ~1,499
of them for a 20 s window -- not a single vector. Each window is mean-pooled
over time into one 1024-d point, so one dot is one window of one cover.

PCA is deliberately run on centered but *unstandardized* features. Per-dimension
standardization is right for the reconstruction loss, where statistics come from
the whole training set, but here the std of each dimension would be estimated
from a few dozen windows, and MERT has near-constant dimensions that such a
divisor amplifies into pure noise (the effect inspect_phase1.py warns about).
Pass --standardize to see it the other way.

Examples:
    python scripts/plot_mert_features.py --bucket BUCKET --mode separate
    python scripts/plot_mert_features.py --bucket BUCKET --mode combined
    python scripts/plot_mert_features.py --bucket BUCKET --no-download   # cache only
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: this runs on an EC2 box with no display

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from src.s3 import S3  # noqa: E402

BRANCHES = ("content", "style")

# Chart surface and ink. Painted explicitly rather than inherited from a
# matplotlib style, so the figure looks the same on any machine.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

# Categorical hues in fixed order (never re-ordered, never generated).
# Slots 1-2 carry the combined plot's two branches and validate all-pairs
# cleanly. The song plot needs one per song, which no eight-hue palette can
# deliver in a scatter -- at eight slots the worst all-pairs normal-vision
# separation is dE 7.1, below the floor of 15 -- so song identity is carried
# redundantly by marker shape, a direct label and a spoke to the song's
# centroid, and hue is the least of the four cues rather than the only one.
HUES = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
SHAPES = ["o", "s", "^", "D", "v", "P"]

# Covers of one song land on top of each other exactly when the content branch
# is working, so their labels collide precisely in the interesting case. Placing
# each cover's label on a different side of its point keeps both readable.
LABEL_OFFSETS = [(9, 5, "left"), (9, -11, "left"), (-9, 5, "right"), (-9, -11, "right")]
BRANCH_COLOR = {"content": HUES[0], "style": HUES[1]}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def song_of(window_id: str) -> str:
    """'10154_10161_00007407' -> '10154' (the SHS work id all covers share)."""
    return window_id.split("_")[0]


def track_of(window_id: str) -> str:
    """'10154_10161_00007407' -> '10154_10161' (this particular cover)."""
    parts = window_id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 3 else window_id


def available_windows(s3: S3, prefix: str, split: str) -> list[str]:
    """Window ids present in *both* branches, sorted."""
    found = {
        branch: {
            Path(k).stem
            for k in s3.list_keys(f"{prefix.strip('/')}/{branch}/{split}/")
        }
        for branch in BRANCHES
    }
    both = found["content"] & found["style"]
    only = (found["content"] | found["style"]) - both
    if only:
        print(f"skipping {len(only)} windows cached in only one branch")
    return sorted(both)


def fetch(
    s3: S3, prefix: str, split: str, window_ids: list[str], cache: Path, workers: int
) -> None:
    """Download any (branch, window) file not already in the local cache."""
    jobs = []
    for branch in BRANCHES:
        (cache / branch).mkdir(parents=True, exist_ok=True)
        for window_id in window_ids:
            local = cache / branch / f"{window_id}.npy"
            if not local.exists():
                key = f"{prefix.strip('/')}/{branch}/{split}/{window_id}.npy"
                jobs.append((key, local))
    if not jobs:
        print(f"cache complete: {len(window_ids)} windows x {len(BRANCHES)} branches")
        return
    print(f"downloading {len(jobs)} files to {cache}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda job: s3.get_file(*job), jobs))


def load_matrix(cache: Path, branch: str, window_ids: list[str]) -> np.ndarray:
    """Mean-pool each window over time -> (n_windows, 1024) float32.

    The stored file is a sequence of frames; a point on the plot is a window,
    so the time axis has to go. The mean is the summary the downstream
    bottleneck is closest to, and it keeps every window comparable regardless
    of small differences in frame count.
    """
    rows = [
        np.load(cache / branch / f"{window_id}.npy").astype(np.float32).mean(axis=0)
        for window_id in window_ids
    ]
    return np.stack(rows)


# --------------------------------------------------------------------------
# Reduction and separation statistics
# --------------------------------------------------------------------------


def pca_2d(x: np.ndarray, standardize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Centered PCA via SVD -> (n, 2) coordinates and the explained-variance ratios.

    Exact and dependency-free; with far fewer samples than dimensions the SVD
    of the centered matrix is the whole computation.
    """
    centered = x - x.mean(axis=0)
    if standardize:
        centered = centered / (x.std(axis=0) + 1e-6)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ right[:2].T
    variance = singular**2
    return coords, variance[:2] / variance.sum()


def separation(x: np.ndarray, songs: list[str]) -> dict[str, float]:
    """Do covers of one song sit closer together than covers of different songs?

    Measured in the full 1024-d space, not in the projection, because two
    principal components of a few dozen points can easily show or hide
    structure that is not there.
    """
    unit = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    distance = 1.0 - unit @ unit.T
    same = np.equal.outer(np.array(songs), np.array(songs))
    off = ~np.eye(len(x), dtype=bool)
    within = distance[same & off]
    between = distance[~same]

    # Nearest-neighbour cover retrieval: for each window, is its closest
    # neighbour another cover of the same song? Chance is 1/(n-1) per window.
    masked = distance + np.eye(len(x)) * 1e9
    hits = same[np.arange(len(x)), masked.argmin(axis=1)]
    return {
        "within": float(within.mean()) if within.size else float("nan"),
        "between": float(between.mean()) if between.size else float("nan"),
        "retrieval": float(hits.mean()),
        "chance": 1.0 / max(len(x) - 1, 1),
        "n_pairs_within": int(within.size // 2),
    }


def report(name: str, stats: dict[str, float]) -> None:
    gap = stats["between"] - stats["within"]
    print(f"\n[{name}] cosine distance in the full 1024-d space")
    print(f"  within  song   {stats['within']:.4f}   ({stats['n_pairs_within']} pairs)")
    print(f"  between songs  {stats['between']:.4f}")
    print(f"  gap            {gap:+.4f}   <- positive means covers are closer")
    print(
        f"  cover retrieval {100 * stats['retrieval']:5.1f}%   "
        f"(chance {100 * stats['chance']:.1f}%)"
    )


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------


def style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)


def axis_label(index: int, ratios: np.ndarray) -> str:
    return f"PC{index + 1}  ({100 * ratios[index]:.1f}% of variance)"


def plot_by_song(
    ax, coords: np.ndarray, window_ids: list[str], songs: list[str],
    ratios: np.ndarray, title: str, annotate: bool,
) -> list[tuple[str, str, str]]:
    """One panel: a point per window, colored and shaped by song.

    Each point is joined to its song's centroid by a thin spoke, so a tight
    cluster of spokes reads as "these covers agree" without the reader having
    to match hues at all -- which matters because past a handful of songs no
    categorical palette can keep every pair distinguishable in a scatter.
    """
    order = sorted(set(songs))
    legend: list[tuple[str, str, str]] = []
    for slot, song in enumerate(order):
        color = HUES[slot % len(HUES)]
        shape = SHAPES[(slot // len(HUES)) % len(SHAPES)]
        legend.append((song, color, shape))
        rows = [i for i, s in enumerate(songs) if s == song]
        points = coords[rows]
        centroid = points.mean(axis=0)
        for point in points:  # spoke: grouping without relying on hue
            ax.plot(
                [centroid[0], point[0]], [centroid[1], point[1]],
                color=color, linewidth=1.0, alpha=0.45, zorder=2,
            )
        ax.scatter(
            points[:, 0], points[:, 1], s=110, c=color, marker=shape,
            edgecolors=SURFACE, linewidths=2.0, zorder=3,
        )
        if annotate:
            for slot_in_song, (row, point) in enumerate(zip(rows, points)):
                dx, dy, align = LABEL_OFFSETS[slot_in_song % len(LABEL_OFFSETS)]
                ax.annotate(
                    window_ids[row].split("_")[1],  # the cover's version id
                    point, textcoords="offset points", xytext=(dx, dy),
                    ha=align, fontsize=7, color=INK_2, zorder=4,
                )
    style_axes(ax, axis_label(0, ratios), axis_label(1, ratios), title)
    return legend


def figure_separate(
    coords: dict[str, np.ndarray], ratios: dict[str, np.ndarray],
    window_ids: list[str], songs: list[str], annotate: bool, out: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), facecolor=SURFACE)
    legend: list[tuple[str, str, str]] = []
    for ax, branch in zip(axes, BRANCHES):
        legend = plot_by_song(
            ax, coords[branch], window_ids, songs, ratios[branch],
            f"{branch.capitalize()} mix", annotate,
        )
    handles = [
        Line2D([], [], color=color, marker=shape, linestyle="none",
               markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, label=song)
        for song, color, shape in legend
    ]
    fig.legend(
        handles=handles, title="Song", loc="center left",
        bbox_to_anchor=(0.885, 0.5), frameon=False, fontsize=8,
        title_fontsize=9, labelcolor=INK_2,
    )
    fig.suptitle(
        "MERT window features by song — covers of one composition share a color and a spoke",
        color=INK, fontsize=13, fontweight="bold", x=0.008, ha="left", y=0.975,
    )
    fig.tight_layout(rect=(0, 0, 0.88, 0.94))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"\nwrote {out}")


def figure_combined(
    coords: np.ndarray, ratios: np.ndarray, window_ids: list[str], out: Path,
) -> None:
    """Both branches in one PCA space, joined per window so the offset is visible."""
    n = len(window_ids)
    fig, ax = plt.subplots(figsize=(9.2, 7.0), facecolor=SURFACE)
    content, style = coords[:n], coords[n:]
    for a, b in zip(content, style):  # same window, two branches
        ax.plot([a[0], b[0]], [a[1], b[1]], color=INK_2, linewidth=0.8,
                alpha=0.28, zorder=2)
    for branch, points in (("content", content), ("style", style)):
        ax.scatter(
            points[:, 0], points[:, 1], s=110, c=BRANCH_COLOR[branch],
            marker="o" if branch == "content" else "s",
            edgecolors=SURFACE, linewidths=2.0, zorder=3, label=branch.capitalize(),
        )
    style_axes(
        ax, axis_label(0, ratios), axis_label(1, ratios),
        "Content and style mixes in a shared PCA space",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="best")
    fig.text(
        0.5, 0.012,
        "Each grey line joins the two branches of the same window: short lines mean "
        "the two mixes carry the same information.",
        ha="center", color=INK_2, fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"\nwrote {out}")


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default=None)
    parser.add_argument("--prefix", default="mert-features")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--mode",
        choices=("separate", "combined"),
        default="separate",
        help="separate: one panel per branch, colored by song. "
        "combined: both branches in one plot, colored by branch.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("mert-cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("plots"))
    parser.add_argument(
        "--limit", type=int, default=0, help="use only the first N windows"
    )
    parser.add_argument(
        "--no-download", action="store_true", help="plot whatever is already cached"
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="divide each dimension by its std before PCA (see the module "
        "docstring for why this is off by default)",
    )
    parser.add_argument("--no-annotate", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    cache = args.cache_dir / args.split
    if args.no_download:
        ids = sorted(
            p.stem for p in (cache / "content").glob("*.npy")
            if (cache / "style" / p.name).exists()
        )
        if not ids:
            raise SystemExit(f"no cached windows under {cache}")
    else:
        s3 = S3(args.bucket, args.region)
        ids = available_windows(s3, args.prefix, args.split)
        if not ids:
            raise SystemExit(
                f"no windows under s3://{args.bucket}/{args.prefix}/"
                f"{{content,style}}/{args.split}/"
            )
    if args.limit:
        ids = ids[: args.limit]
    if not args.no_download:
        fetch(s3, args.prefix, args.split, ids, cache, args.workers)

    songs = [song_of(w) for w in ids]
    tracks = [track_of(w) for w in ids]
    per_song = defaultdict(list)
    for song, track in zip(songs, tracks):
        per_song[song].append(track)
    print(
        f"\n{len(ids)} windows, {len(set(tracks))} covers, {len(per_song)} songs "
        f"({np.mean([len(v) for v in per_song.values()]):.1f} windows per song)"
    )

    data = {branch: load_matrix(cache, branch, ids) for branch in BRANCHES}
    print(f"pooled features: {data['content'].shape} per branch")
    for branch in BRANCHES:
        report(branch, separation(data[branch], songs))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    annotate = not args.no_annotate and len(ids) <= 60
    if args.mode == "separate":
        coords, ratios = {}, {}
        for branch in BRANCHES:
            coords[branch], ratios[branch] = pca_2d(data[branch], args.standardize)
            print(
                f"[{branch}] PC1+PC2 explain "
                f"{100 * ratios[branch].sum():.1f}% of the variance"
            )
        figure_separate(
            coords, ratios, ids, songs, annotate,
            args.out_dir / f"mert_by_song_{args.split}.png",
        )
    else:
        stacked = np.vstack([data["content"], data["style"]])
        coords, ratios = pca_2d(stacked, args.standardize)
        print(f"[combined] PC1+PC2 explain {100 * ratios.sum():.1f}% of the variance")
        figure_combined(
            coords, ratios, ids, args.out_dir / f"mert_content_vs_style_{args.split}.png"
        )


if __name__ == "__main__":
    main()
