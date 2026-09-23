#!/usr/bin/env python3
"""Designate the fixed works that make up the validation and test sets.

Writes splits/eval_works.json: two disjoint lists of works (by default
"val50" and "test50", 50 works each). The file is committed to git, because
the evaluation protocol must not change between the runs it compares.

Where the works come from
-------------------------
The official held-out splits (validate.csv, test.csv), never train. Train is
already de-contaminated against both (shs100k_meta.HELD_OUT_AGAINST), so the
designated works are guaranteed unseen, and training keeps every composition it
has. Compositions are the scarce resource, so carving eval works out of train
would cost exactly what the learning curve is trying to measure.

The official splits are not used as they are because their downloaded parts
are lopsided: val has 27 works, one of which holds 37% of its tracks, and test
has 77. Their union, restricted to works with enough downloaded tracks to form
cover pairs, is re-partitioned into two size-matched halves.

Selection
---------
1. Pool: works in val.csv or test.csv with >= --min-tracks downloaded tracks
   (per data/logs/{split}_downloaded_songs.csv). The 2 works listed in *both*
   official splits (5854, 186755) are sourced from one split only, the one
   holding more of their downloads. All 76 videos the two official splits
   share belong to those 2 works, so no video can land on both sides.
2. If the pool holds more than 2 x --count works, 2 x --count are drawn by
   size-stratified sampling: sort by track count, cut into 2 x --count
   contiguous bins, draw one work per bin.
3. The picks, sorted by size, are split in consecutive pairs, and a seeded
   coin sends one of each pair to val and the other to test. Both sets
   therefore span the same range of clique sizes.

Storage is untouched: a work's audio, chroma and alignments stay under its
official split. Only `build_manifest.py --split val50` gathers them.

Examples:
    python scripts/designate_eval_works.py            # writes splits/eval_works.json
    python scripts/designate_eval_works.py --dry-run  # print the designation only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from shs100k_meta import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    EVAL_WORKS_FILE,
    audio_from_csv,
    split_tracks,
)

SOURCES = ("val", "test")


def downloaded_pool(data_root: Path, min_tracks: int) -> dict[int, tuple[str, int]]:
    """work id -> (source split, downloaded tracks) for every eligible work.

    A work listed in both official splits is sourced from whichever holds
    more of its downloaded tracks, and only from that one.
    """
    best: dict[int, tuple[str, int]] = {}
    for source in SOURCES:
        downloaded = audio_from_csv(data_root, source)
        counts: dict[int, int] = {}
        for track in split_tracks(source):
            if track.key in downloaded:
                counts[track.song_id] = counts.get(track.song_id, 0) + 1
        for work, n in counts.items():
            if n > best.get(work, ("", 0))[1]:
                best[work] = (source, n)
    return {work: entry for work, entry in best.items() if entry[1] >= min_tracks}


def stratified_pick(
    pool: dict[int, tuple[str, int]], n: int, rng: random.Random
) -> list[int]:
    """n works spread evenly over the clique-size distribution of the pool."""
    ordered = sorted(pool, key=lambda w: (pool[w][1], w))
    if len(ordered) <= n:
        return ordered
    picks = []
    for i in range(n):
        lo = i * len(ordered) // n
        hi = (i + 1) * len(ordered) // n
        picks.append(rng.choice(ordered[lo:hi]))
    return picks


def split_matched(
    works: list[int], pool: dict[int, tuple[str, int]], rng: random.Random
) -> tuple[list[int], list[int]]:
    """Size-sorted consecutive pairs, one member of each to each side."""
    ordered = sorted(works, key=lambda w: (pool[w][1], w))
    first, second = [], []
    for i in range(0, len(ordered) - 1, 2):
        a, b = ordered[i], ordered[i + 1]
        if rng.random() < 0.5:
            a, b = b, a
        first.append(a)
        second.append(b)
    return first, second


def describe(name: str, works: list[int], pool: dict[int, tuple[str, int]]) -> None:
    sizes = sorted(pool[w][1] for w in works)
    total = sum(sizes)
    sources = {s: sum(1 for w in works if pool[w][0] == s) for s in SOURCES}
    print(
        f"  {name}: {len(works)} works, {total} downloaded tracks "
        f"(clique size min {sizes[0]}, median {sizes[len(sizes) // 2]}, "
        f"max {sizes[-1]}; largest work = {sizes[-1] / total:.0%} of tracks); "
        f"from " + ", ".join(f"{s} {n}" for s, n in sources.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--count", type=int, default=50, help="works per split")
    parser.add_argument(
        "--min-tracks",
        type=int,
        default=10,
        help="downloaded tracks a work needs to be eligible (10 tracks = 45 "
        "candidate pairs, enough to expect ~20 usable ones at the measured "
        "~42%% keep rate)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--names", nargs=2, default=("val50", "test50"))
    parser.add_argument("--out", type=Path, default=EVAL_WORKS_FILE)
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing designation"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not (args.force or args.dry_run):
        raise SystemExit(
            f"{args.out} already exists. The designation is part of the "
            f"evaluation protocol; pass --force only if you mean to change it "
            f"(and invalidate every result measured against the old one)."
        )

    pool = downloaded_pool(args.data_root, args.min_tracks)
    print(
        f"pool: {len(pool)} works in {'/'.join(SOURCES)} with >= "
        f"{args.min_tracks} downloaded tracks"
    )
    if len(pool) < 2 * args.count:
        raise SystemExit(
            f"need {2 * args.count} eligible works, found {len(pool)}. Lower "
            f"--min-tracks or --count, or download more of val/test."
        )

    rng = random.Random(args.seed)
    picks = stratified_pick(pool, 2 * args.count, rng)
    val_works, test_works = split_matched(picks, pool, rng)
    names = dict(zip(("val", "test"), args.names))
    print("designation:")
    describe(names["val"], val_works, pool)
    describe(names["test"], test_works, pool)

    unused = sorted(set(pool) - set(picks))
    if unused:
        print(f"  {len(unused)} eligible works left unused")
    if args.dry_run:
        return

    def entries(works: list[int]) -> dict[str, dict]:
        return {
            str(w): {"source": pool[w][0], "n_tracks": pool[w][1]}
            for w in sorted(works)
        }

    designation = {
        "description": (
            "Fixed evaluation works, drawn from the official SHS100K validate "
            "and test splits (never train) by scripts/designate_eval_works.py. "
            "n_tracks is the downloaded count at designation time."
        ),
        "created": time.strftime("%Y-%m-%d"),
        "seed": args.seed,
        "min_tracks": args.min_tracks,
        "pool": list(SOURCES),
        "splits": {
            names["val"]: {"works": entries(val_works)},
            names["test"]: {"works": entries(test_works)},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(designation, f, indent=1)
        f.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
