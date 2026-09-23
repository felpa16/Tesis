#!/usr/bin/env python3
"""Nested, size-stratified work subsets of a manifest (for learning curves).

Reads manifests/{split}/ and writes one manifest per fraction, e.g.

    manifests/train_w25/  {tracks,pairs}.jsonl + subset.json
    manifests/train_w50/
    manifests/train_w100/

Every track and pair of a selected work is kept; nothing of an unselected work
is. The unit is the work (composition), because that is what the learning
curve varies: how many distinct compositions the model sees.

Two properties make the fractions comparable:

  * Nested: each work draws one number u in [0, 1) and a subset at fraction f
    is {u < f}. So w25 is contained in w50, which is contained in w100. A
    better score at 50% then cannot come from a lucky redraw of the works.
  * Stratified by size: works are sorted by (pairs, tracks) and cut into
    blocks of --block consecutive works. Within a block, a seeded permutation
    hands out ranks 0..block-1 and u = (rank + jitter) / block. At f = 0.25
    and block 4, exactly one work per block is chosen. The 25% subset
    therefore gets a quarter of the huge cliques as well as a quarter of the
    small ones, instead of however many the dice gave it. Fractions that are
    multiples of 1/block are exact; others are exact in expectation.

Manifest paths are relative to the data root, so a subset reads the same
audio and alignment files as its parent. Pass `--train-split train_w25` to
train.py.

Example:
    python scripts/subset_manifest.py --split train --fractions 0.25 0.5 1.0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from shs100k_meta import DEFAULT_DATA_ROOT  # noqa: E402
from src.data.manifest import (  # noqa: E402
    PairEntry,
    TrackEntry,
    manifest_dir,
    read_pairs,
    read_tracks,
    write_jsonl,
)


def work_draws(
    tracks: list[TrackEntry], pairs: list[PairEntry], block: int, seed: int
) -> dict[int, float]:
    """work id -> u in [0, 1); the subset at fraction f is {u < f}."""
    n_tracks: dict[int, int] = {}
    n_pairs: dict[int, int] = {}
    for track in tracks:
        n_tracks[track.song_id] = n_tracks.get(track.song_id, 0) + 1
    for pair in pairs:
        n_pairs[pair.song_id] = n_pairs.get(pair.song_id, 0) + 1
    ordered = sorted(n_tracks, key=lambda w: (n_pairs.get(w, 0), n_tracks[w], w))

    rng = random.Random(seed)
    draws: dict[int, float] = {}
    for start in range(0, len(ordered), block):
        chunk = ordered[start : start + block]
        ranks = list(range(len(chunk)))
        rng.shuffle(ranks)
        for work, rank in zip(chunk, ranks):
            draws[work] = (rank + rng.random()) / len(chunk)
    return draws


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="train", help="manifest to subset")
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0]
    )
    parser.add_argument(
        "--block",
        type=int,
        default=4,
        help="stratification block (consecutive works by size); fractions "
        "that are multiples of 1/block come out exact",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prefix",
        default=None,
        help="output manifest name prefix (default: '{split}_w', giving "
        "train_w25 for 0.25)",
    )
    args = parser.parse_args()

    if any(not 0.0 < f <= 1.0 for f in args.fractions):
        raise SystemExit("--fractions must lie in (0, 1]")
    tracks = read_tracks(args.data_root, args.split)
    pairs = read_pairs(args.data_root, args.split)
    draws = work_draws(tracks, pairs, args.block, args.seed)
    prefix = args.prefix if args.prefix is not None else f"{args.split}_w"
    print(
        f"[{args.split}] {len(draws)} works, {len(tracks)} tracks, "
        f"{len(pairs)} pairs"
    )

    for fraction in sorted(args.fractions):
        works = {w for w, u in draws.items() if u < fraction}
        name = f"{prefix}{round(fraction * 100)}"
        subset_tracks = [t for t in tracks if t.song_id in works]
        subset_pairs = [p for p in pairs if p.song_id in works]
        out_dir = manifest_dir(args.data_root, name)
        write_jsonl(out_dir / "tracks.jsonl", subset_tracks)
        write_jsonl(out_dir / "pairs.jsonl", subset_pairs)
        with open(out_dir / "subset.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "parent": args.split,
                    "fraction": fraction,
                    "block": args.block,
                    "seed": args.seed,
                    "n_works": len(works),
                    "n_tracks": len(subset_tracks),
                    "n_pairs": len(subset_pairs),
                    "works": sorted(works),
                },
                f,
                indent=1,
            )
        print(
            f"[{name}] {len(works)} works ({len(works) / len(draws):.1%}), "
            f"{len(subset_tracks)} tracks, {len(subset_pairs)} pairs -> {out_dir}"
        )


if __name__ == "__main__":
    main()
