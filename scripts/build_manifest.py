#!/usr/bin/env python3
"""Build training manifests from preprocessed SHS100K data.

Scans a split's downloaded audio, probes durations, and cross-references the
alignment outputs to produce, under data/manifests/{split}/:

    tracks.jsonl  every usable track (downloaded, long enough)
    pairs.jsonl   every cover pair whose alignment score passes the quality
                  threshold (pair supervision only — tracks that fail pairing
                  still appear in tracks.jsonl and feed plain reconstruction)

Example:
    python scripts/build_manifest.py --split val
"""

from __future__ import annotations

import argparse
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
    existing_audio,
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


def build_split(
    data_root: Path, split: str, min_score: float, min_duration: float, workers: int
) -> None:
    audio = existing_audio(data_root, split)
    print(f"[{split}] {len(audio)} downloaded tracks")

    # The manifest, not the download, is what defines the training set, so the
    # de-contamination in shs100k_meta has to be re-applied to whatever is
    # actually on disk: audio fetched before the filter existed, or copied in
    # from elsewhere, would otherwise reach training.
    in_dataset = {t.key for t in split_tracks(split, drop_leaked=False)}
    allowed = {t.key for t in split_tracks(split)}
    leaked = {key for key in audio if key in in_dataset and key not in allowed}
    stale = {key for key in audio if key not in in_dataset}
    for key in leaked | stale:
        del audio[key]
    if leaked or stale:
        print(
            f"[{split}] excluded {len(leaked)} leaked and {len(stale)} stale "
            f"files on disk; {len(audio)} tracks usable"
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        durations = dict(
            zip(audio.keys(), pool.map(probe_duration, audio.values()))
        )

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
                has_chroma=(chroma_dir(data_root, split) / f"{key}.npz").exists(),
            )
        )
    kept_keys = {t.key for t in tracks}

    pairs: list[PairEntry] = []
    low_score = 0
    missing_track = 0
    align_paths = sorted(alignment_dir(data_root, split).glob("*.npz"))
    for path in align_paths:
        song_id, ver_a, ver_b = (int(x) for x in path.stem.split("_"))
        key_a, key_b = f"{song_id}_{ver_a}", f"{song_id}_{ver_b}"
        if key_a not in kept_keys or key_b not in kept_keys:
            missing_track += 1
            continue
        data = np.load(path)
        score = float(data["score"])
        if score < min_score:
            low_score += 1
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

    out_dir = manifest_dir(data_root, split)
    write_jsonl(out_dir / "tracks.jsonl", tracks)
    write_jsonl(out_dir / "pairs.jsonl", pairs)

    paired = {k for p in pairs for k in (p.key_a, p.key_b)}
    print(
        f"[{split}] tracks: {len(tracks)} kept, {too_short} too short "
        f"(<{min_duration:.0f}s)"
    )
    print(
        f"[{split}] pairs: {len(pairs)} kept, {low_score} below score "
        f"{min_score}, {missing_track} referencing dropped tracks"
    )
    print(
        f"[{split}] {len(paired)}/{len(tracks)} tracks have >=1 aligned partner; "
        f"manifests written to {out_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="all")
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
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    for split in SPLITS if args.split == "all" else [args.split]:
        build_split(
            args.data_root, split, args.min_score, args.min_duration, args.workers
        )


if __name__ == "__main__":
    main()
