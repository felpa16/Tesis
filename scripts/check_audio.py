#!/usr/bin/env python3
"""Validate downloaded audio before uploading it to S3.

The pipeline decodes everything through ffmpeg, so mixed containers (.webm,
.m4a, .mp4) are fine and need no conversion. What is worth catching before
shipping hundreds of GB is the small stuff that wastes space or breaks a worker
hours into preprocessing:

  * files carrying a video stream -- yt-dlp's "bestaudio/best" format string
    falls back to a combined stream when a video has no audio-only format, and
    those are many times larger for the same audio
  * unreadable or zero-duration files
  * leftover .part files from interrupted downloads

Exits non-zero if any problem is found, so it can gate an upload script.

Examples:
    python scripts/check_audio.py --split val
    python scripts/check_audio.py --split all --workers 16
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from shs100k_meta import DEFAULT_DATA_ROOT, SPLITS, audio_dir  # noqa: E402


def probe(path: Path) -> dict:
    """ffprobe one file into {duration, codecs, sample_rates, channels, video}."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,codec_name,sample_rate,channels",
        "-of", "json", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True, timeout=60).stdout
        data = json.loads(out)
    except Exception as exc:
        return {"path": path, "error": f"{type(exc).__name__}: {exc}"}

    streams = data.get("streams") or []
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    try:
        duration = float((data.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return {
        "path": path,
        "duration": duration,
        "video": any(s.get("codec_type") == "video" for s in streams),
        "codec": audio[0].get("codec_name") if audio else None,
        "sample_rate": audio[0].get("sample_rate") if audio else None,
        "channels": audio[0].get("channels") if audio else None,
        "size": path.stat().st_size,
    }


def check_split(data_root: Path, split: str, workers: int, min_duration: float) -> int:
    directory = audio_dir(data_root, split)
    if not directory.is_dir():
        print(f"[{split}] no audio directory {directory}")
        return 0

    partials = sorted(p for p in directory.iterdir() if ".part" in p.name)
    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and not p.name.startswith(".") and ".part" not in p.name
    )
    if not files:
        print(f"[{split}] no audio files")
        return 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(probe, files))

    broken = [r for r in results if r.get("error")]
    short = [r for r in results if not r.get("error") and r["duration"] < min_duration]
    with_video = [r for r in results if r.get("video")]
    total = sum(r.get("size", 0) for r in results)

    print(f"\n[{split}] {len(files)} files, {total / 1e9:.1f} GB, "
          f"{total / max(len(files), 1) / 1e6:.2f} MB average")

    for label, counter in (
        ("extension", collections.Counter(p.suffix for p in files)),
        ("codec", collections.Counter(r.get("codec") for r in results if not r.get("error"))),
        ("sample rate", collections.Counter(r.get("sample_rate") for r in results if not r.get("error"))),
        ("channels", collections.Counter(r.get("channels") for r in results if not r.get("error"))),
    ):
        summary = "  ".join(f"{k}={v}" for k, v in counter.most_common())
        print(f"  {label:12s} {summary}")

    problems = 0
    for label, rows, hint in (
        ("carry a video stream", with_video, "re-download these; they waste space"),
        ("unreadable", broken, "delete and re-download"),
        (f"shorter than {min_duration:g}s", short, "likely truncated; delete and re-download"),
        ("leftover .part files", [{"path": p} for p in partials], "safe to delete"),
    ):
        if not rows:
            continue
        problems += len(rows)
        print(f"  ! {len(rows)} {label} -- {hint}")
        for row in rows[:5]:
            print(f"      {row['path'].name}")
        if len(rows) > 5:
            print(f"      ... and {len(rows) - 5} more")

    if not problems:
        print("  no problems found; ready to upload")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--min-duration",
        type=float,
        default=20.0,
        help="flag files shorter than this many seconds (= the training window)",
    )
    args = parser.parse_args()

    splits = list(SPLITS) if args.split == "all" else [args.split]
    problems = sum(
        check_split(args.data_root, s, args.workers, args.min_duration) for s in splits
    )
    if problems:
        print(f"\n{problems} problem file(s) found")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
