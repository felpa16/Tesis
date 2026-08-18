#!/usr/bin/env python3
"""Validate cover-alignment confidence scores.

Compares the Smith-Waterman normalized score distribution of
  * positives: within-song (true cover) pairs, read from data/alignments/
  * negatives: random cross-song pairs, aligned on the fly with the exact
    same procedure (OTI + Smith-Waterman on cached chroma)

and reports summary statistics, the overlap region, and the separation
(ROC-AUC) so a quality-filtering threshold can be chosen.

Example:
    python scripts/validate_alignment.py --split val --negatives 200
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from align_covers import optimal_transposition_index, smith_waterman_path  # noqa: E402
from shs100k_meta import DEFAULT_DATA_ROOT, alignment_dir, chroma_dir  # noqa: E402


def load_positive_scores(data_root: Path, split: str) -> list[float]:
    scores = []
    for path in sorted(alignment_dir(data_root, split).glob("*.npz")):
        scores.append(float(np.load(path)["score"]))
    return scores


def negative_scores(
    data_root: Path,
    split: str,
    n_pairs: int,
    match_quantile: float,
    gap: float,
    seed: int,
) -> list[float]:
    by_song: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(chroma_dir(data_root, split).glob("*.npz")):
        by_song[int(path.stem.split("_")[0])].append(path)
    songs = [s for s in by_song if by_song[s]]
    if len(songs) < 2:
        return []

    rng = random.Random(seed)
    scores = []
    for _ in range(n_pairs):
        song_a, song_b = rng.sample(songs, 2)
        path_a = rng.choice(by_song[song_a])
        path_b = rng.choice(by_song[song_b])
        chroma_a = np.load(path_a)["chroma"]
        chroma_b = np.load(path_b)["chroma"]
        oti = optimal_transposition_index(chroma_a, chroma_b)
        sim = chroma_a.T @ np.roll(chroma_b, oti, axis=0)
        _, score = smith_waterman_path(sim, match_quantile, gap)
        scores.append(score)
    return scores


def describe(name: str, scores: list[float]) -> None:
    arr = np.asarray(scores)
    quantiles = np.percentile(arr, [5, 25, 50, 75, 95])
    print(
        f"{name:9s} n={len(arr):4d}  mean={arr.mean():.3f}  "
        f"p5={quantiles[0]:.3f}  p25={quantiles[1]:.3f}  p50={quantiles[2]:.3f}  "
        f"p75={quantiles[3]:.3f}  p95={quantiles[4]:.3f}"
    )


def roc_auc(pos: list[float], neg: list[float]) -> float:
    """Probability that a random positive outscores a random negative."""
    pos_arr, neg_arr = np.asarray(pos), np.asarray(neg)
    greater = (pos_arr[:, None] > neg_arr[None, :]).sum()
    ties = (pos_arr[:, None] == neg_arr[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos_arr) * len(neg_arr)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--negatives", type=int, default=200)
    parser.add_argument("--match-quantile", type=float, default=0.9)
    parser.add_argument("--gap", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pos = load_positive_scores(args.data_root, args.split)
    if not pos:
        sys.exit("no alignments found; run align_covers.py first")
    neg = negative_scores(
        args.data_root, args.split, args.negatives,
        args.match_quantile, args.gap, args.seed,
    )

    describe("positive", pos)
    if neg:
        describe("negative", neg)
        print(f"ROC-AUC: {roc_auc(pos, neg):.3f}")
        neg_p95 = float(np.percentile(neg, 95))
        kept = float(np.mean(np.asarray(pos) > neg_p95))
        print(
            f"threshold at negative p95 = {neg_p95:.3f} "
            f"keeps {kept:.1%} of true cover pairs"
        )
    else:
        print("not enough distinct songs for negative pairs")


if __name__ == "__main__":
    main()
