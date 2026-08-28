#!/usr/bin/env python3
"""Re-check tracks the download log recorded as permanently unavailable.

A flagged IP is served "Video unavailable" for perfectly healthy videos, and the
error text is identical to real link rot. This samples keys the log marked
permanent, re-extracts them (metadata only, no download), and reports how many
are actually alive -- which tells you whether the log is trustworthy or whether
the machine was soft-blocked while it was written.

Run it from a DIFFERENT network than the one that produced the log, or after
waiting out a block, otherwise it just reproduces the same false answer.

Examples:
    python scripts/audit_failures.py --split train --sample 20
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent / "yt-dlp"))

import yt_dlp  # noqa: E402

from download_shs100k import permanent_failures  # noqa: E402
from shs100k_meta import DEFAULT_DATA_ROOT, SPLITS, split_tracks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    log_path = args.data_root / "logs" / f"download_{args.split}.csv"
    keys = permanent_failures(log_path)
    if not keys:
        print(f"no permanent failures logged in {log_path}")
        return

    by_key = {t.key: t for t in split_tracks(args.split)}
    candidates = [by_key[k] for k in sorted(keys) if k in by_key]
    random.Random(args.seed).shuffle(candidates)
    sample = candidates[: args.sample]
    print(
        f"\n{len(keys)} keys logged permanent; re-checking {len(sample)} of them "
        f"at {args.sleep:g}s intervals\n"
    )

    opts = {
        "format": "bestaudio/best", "quiet": True, "no_warnings": True,
        "noplaylist": True, "skip_download": True, "socket_timeout": 30,
    }
    alive = 0
    with yt_dlp.YoutubeDL(opts) as ydl:
        for track in sample:
            try:
                ydl.extract_info(track.url, download=False)
                alive += 1
                print(f"  ALIVE  {track.key:<18} {track.youtube_id}")
            except Exception as exc:
                detail = str(exc).replace("\n", " ")
                print(f"  dead   {track.key:<18} {track.youtube_id}  {detail[-70:]}")
            time.sleep(args.sleep)

    rate = alive / max(len(sample), 1)
    print(f"\n{alive}/{len(sample)} ({rate:.0%}) are actually available")
    if rate > 0.2:
        print(
            "The log is NOT trustworthy: these were soft-blocked, not dead.\n"
            "Re-run the downloader with --retry-permanent to attempt them again."
        )
    else:
        print("Consistent with genuine link rot; the log looks trustworthy.")


if __name__ == "__main__":
    main()
