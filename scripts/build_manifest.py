#!/usr/bin/env python3
"""Build training manifests from preprocessed SHS100K data.

Scans a split's downloaded audio, probes durations, and cross-references the
alignment outputs to produce, under data/manifests/{split}/:

    tracks.jsonl  every usable track (downloaded, long enough)
    pairs.jsonl   every cover pair whose alignment score passes the quality
                  threshold (pair supervision only — tracks that fail pairing
                  still appear in tracks.jsonl and feed plain reconstruction)

--split also accepts the designated evaluation splits in splits/eval_works.json
(e.g. val50, test50; see designate_eval_works.py). Their works live under the
official split they came from, so the manifest gathers audio, chroma and
alignments from each work's source split. Manifest paths are relative to the
data root, so the result is an ordinary manifest that every loader reads as-is.

--kept-per-song K restricts pairs to the per-work selection written by
`align_covers.py --kept-per-song K`, which balances cover supervision across
works instead of letting the largest cliques dominate it.

--max-tracks-per-song M caps a work's tracks (pair tracks first, then a seeded
fill), so track-level metrics on an evaluation split are not dominated by one
huge clique.

Examples:
    python scripts/build_manifest.py --split val
    python scripts/build_manifest.py --split train --kept-per-song 20
    python scripts/build_manifest.py --split val50 --kept-per-song 20 \\
        --max-tracks-per-song 40
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from shs100k_meta import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    SPLITS,
    alignment_dir,
    chroma_dir,
    designated_splits,
    existing_audio,
    selection_path,
    split_tracks,
)
from src.data.manifest import (  # noqa: E402
    PairEntry,
    TrackEntry,
    manifest_dir,
    write_jsonl,
)


def probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(path),
    ]
    try:
        return float(subprocess.run(cmd, capture_output=True, check=True).stdout)
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def usable_audio(
    data_root: Path, source: str, works: set[int] | None, exclude: set[int]
) -> dict[str, Path]:
    """Audio on disk for a source split, de-contaminated and filtered by work."""
    audio = existing_audio(data_root, source)

    # The manifest, not the download, is what defines the training set, so the
    # de-contamination in shs100k_meta has to be re-applied to whatever is
    # actually on disk: audio fetched before the filter existed, or copied in
    # from elsewhere, would otherwise reach training.
    in_dataset = {t.key for t in split_tracks(source, drop_leaked=False)}
    allowed = {t.key for t in split_tracks(source)}
    leaked = {key for key in audio if key in in_dataset and key not in allowed}
    stale = {key for key in audio if key not in in_dataset}
    for key in leaked | stale:
        del audio[key]
    if leaked or stale:
        print(
            f"[{source}] excluded {len(leaked)} leaked and {len(stale)} stale "
            f"files on disk; {len(audio)} tracks usable"
        )

    def work(key: str) -> int:
        return int(key.split("_")[0])

    held_out = {key for key in audio if work(key) in exclude}
    if held_out:
        print(f"[{source}] excluded {len(held_out)} tracks of designated eval works")
    return {
        key: path
        for key, path in audio.items()
        if key not in held_out and (works is None or work(key) in works)
    }


def build_tracks(
    data_root: Path,
    source: str,
    audio: dict[str, Path],
    min_duration: float,
    workers: int,
) -> tuple[list[TrackEntry], int]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        durations = dict(zip(audio.keys(), pool.map(probe_duration, audio.values())))

    tracks: list[TrackEntry] = []
    too_short = 0
    for key, path in sorted(audio.items()):
        duration = durations[key]
        if duration < min_duration:
            too_short += 1
            continue
        song_id, version_id = (int(x) for x in key.split("_"))
        tracks.append(
            TrackEntry(
                key=key,
                song_id=song_id,
                version_id=version_id,
                audio=str(path.relative_to(data_root)),
                duration=duration,
                has_chroma=(chroma_dir(data_root, source) / f"{key}.npz").exists(),
            )
        )
    return tracks, too_short


def candidate_alignments(
    data_root: Path, source: str, kept_per_song: int, min_score: float
) -> list[Path]:
    """Alignment files to consider: all of them, or the per-work selection."""
    directory = alignment_dir(data_root, source)
    if kept_per_song <= 0:
        return sorted(directory.glob("*.npz"))
    path = selection_path(data_root, source, kept_per_song)
    if not path.exists():
        raise SystemExit(
            f"no pair selection at {path}. Write it first:\n"
            f"    python scripts/align_covers.py --stage align --split {source} "
            f"--kept-per-song {kept_per_song}"
        )
    with open(path, encoding="utf-8") as f:
        selection = json.load(f)
    if selection["min_score"] != min_score:
        print(
            f"[{source}] warning: the selection was made at min-score "
            f"{selection['min_score']}, this manifest filters at {min_score}"
        )
    return sorted(
        directory / f"{stem}.npz"
        for work in selection["works"].values()
        for stem in work["pairs"]
    )


def build_pairs(
    data_root: Path,
    source: str,
    kept_keys: set[str],
    min_score: float,
    kept_per_song: int,
    works: set[int] | None,
) -> tuple[list[PairEntry], dict[str, int]]:
    pairs: list[PairEntry] = []
    stats = {"low_score": 0, "missing_track": 0, "missing_file": 0}
    for path in candidate_alignments(data_root, source, kept_per_song, min_score):
        song_id, ver_a, ver_b = (int(x) for x in path.stem.split("_"))
        if works is not None and song_id not in works:
            continue
        key_a, key_b = f"{song_id}_{ver_a}", f"{song_id}_{ver_b}"
        if key_a not in kept_keys or key_b not in kept_keys:
            stats["missing_track"] += 1
            continue
        if not path.exists():
            stats["missing_file"] += 1
            continue
        data = np.load(path)
        score = float(data["score"])
        if score < min_score:
            stats["low_score"] += 1
            continue
        t_a, t_b = data["t_a"], data["t_b"]
        pairs.append(
            PairEntry(
                key_a=key_a,
                key_b=key_b,
                song_id=song_id,
                score=score,
                oti=int(data["oti"]),
                alignment=str(path.relative_to(data_root)),
                t_a_first=float(t_a[0]),
                t_a_last=float(t_a[-1]),
                t_b_first=float(t_b[0]),
                t_b_last=float(t_b[-1]),
                n_points=int(len(t_a)),
            )
        )
    return pairs, stats


def cap_tracks(
    tracks: list[TrackEntry], pairs: list[PairEntry], cap: int
) -> list[TrackEntry]:
    """At most `cap` tracks per work: every pair track, then a seeded fill.

    Pair tracks are always kept, even past the cap, since pairs.jsonl must
    only reference tracks listed in tracks.jsonl.
    """
    if cap <= 0:
        return tracks
    paired = {key for pair in pairs for key in (pair.key_a, pair.key_b)}
    by_song: dict[int, list[TrackEntry]] = {}
    for track in tracks:
        by_song.setdefault(track.song_id, []).append(track)
    kept: list[TrackEntry] = []
    for song_id, group in by_song.items():
        in_pairs = [t for t in group if t.key in paired]
        others = sorted((t for t in group if t.key not in paired), key=lambda t: t.key)
        random.Random(song_id).shuffle(others)
        kept += in_pairs + others[: max(cap - len(in_pairs), 0)]
    return sorted(kept, key=lambda t: t.key)


def build_split(
    data_root: Path,
    name: str,
    sources: dict[str, set[int] | None],
    exclude: set[int],
    min_score: float,
    min_duration: float,
    workers: int,
    kept_per_song: int,
    max_tracks_per_song: int,
) -> None:
    """Write manifests/{name}/ from one or more source splits.

    sources maps an official split to the works to take from it (None = all).
    """
    tracks: list[TrackEntry] = []
    pairs: list[PairEntry] = []
    too_short = 0
    stats = {"low_score": 0, "missing_track": 0, "missing_file": 0}
    for source, works in sources.items():
        audio = usable_audio(data_root, source, works, exclude)
        print(f"[{name}] {len(audio)} downloaded tracks from {source}")
        source_tracks, short = build_tracks(
            data_root, source, audio, min_duration, workers
        )
        too_short += short
        if not source_tracks:  # nothing to pair, and maybe nothing aligned yet
            continue
        source_pairs, source_stats = build_pairs(
            data_root,
            source,
            {t.key for t in source_tracks},
            min_score,
            kept_per_song,
            works,
        )
        tracks += source_tracks
        pairs += source_pairs
        for key, value in source_stats.items():
            stats[key] += value

    uncapped = len(tracks)
    tracks = cap_tracks(tracks, pairs, max_tracks_per_song)

    out_dir = manifest_dir(data_root, name)
    write_jsonl(out_dir / "tracks.jsonl", tracks)
    write_jsonl(out_dir / "pairs.jsonl", pairs)

    cap_note = (
        f", capped to {len(tracks)} at {max_tracks_per_song}/work"
        if max_tracks_per_song > 0
        else ""
    )
    print(
        f"[{name}] tracks: {uncapped} kept{cap_note}, {too_short} too short "
        f"(<{min_duration:.0f}s)"
    )
    print(
        f"[{name}] pairs: {len(pairs)} kept, {stats['low_score']} below score "
        f"{min_score}, {stats['missing_track']} referencing dropped tracks"
        + (f", {stats['missing_file']} missing files" if stats["missing_file"] else "")
    )
    works_with_tracks = {t.song_id for t in tracks}
    per_work: dict[int, int] = {w: 0 for w in works_with_tracks}
    for pair in pairs:
        per_work[pair.song_id] = per_work.get(pair.song_id, 0) + 1
    counts = sorted(per_work.values())
    if counts:
        line = (
            f"[{name}] {len(works_with_tracks)} works; pairs per work: min "
            f"{counts[0]}, median {counts[len(counts) // 2]}, max {counts[-1]}"
        )
        if kept_per_song > 0:
            line += f"; {sum(1 for c in counts if c < kept_per_song)} below {kept_per_song}"
        print(line)
    paired = {k for p in pairs for k in (p.key_a, p.key_b)}
    print(
        f"[{name}] {len(paired)}/{len(tracks)} tracks have >=1 aligned partner; "
        f"manifests written to {out_dir}"
    )


def main() -> None:
    designated = designated_splits()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--split",
        choices=[*SPLITS, *designated, "all"],
        default="all",
        help="an official split, a designated eval split from "
        "splits/eval_works.json, or all (= the official splits)",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.2,
        help="alignment quality threshold (cross-song negative p95, "
        "see validate_alignment.py)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=20.0,
        help="drop tracks shorter than this many seconds (= window size)",
    )
    parser.add_argument(
        "--kept-per-song",
        type=int,
        default=0,
        help="use only the pairs selected by align_covers.py --kept-per-song "
        "(0 = every aligned pair that passes --min-score)",
    )
    parser.add_argument(
        "--max-tracks-per-song",
        type=int,
        default=0,
        help="cap tracks per work, pair tracks first (0 = no cap)",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # Designated eval works must never reach training, whichever pool they
    # were drawn from.
    eval_works = {w for spec in designated.values() for w in spec}
    for split in SPLITS if args.split == "all" else [args.split]:
        if split in designated:
            sources: dict[str, set[int] | None] = {}
            for work, source in designated[split].items():
                sources.setdefault(source, set()).add(work)  # type: ignore[union-attr]
            exclude: set[int] = set()
        else:
            sources = {split: None}
            exclude = eval_works if split == "train" else set()
        build_split(
            args.data_root,
            split,
            sources,
            exclude,
            args.min_score,
            args.min_duration,
            args.workers,
            args.kept_per_song,
            args.max_tracks_per_song,
        )


if __name__ == "__main__":
    main()
