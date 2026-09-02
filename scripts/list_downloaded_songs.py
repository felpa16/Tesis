#!/usr/bin/env python3
"""Write a CSV listing the audio files already downloaded for one split.

The downloader normally decides what to skip by looking at data/audio/{split}.
That only works while the audio is sitting on the same machine: once a split has
been uploaded to S3 (or moved to an external disk) the folder is empty and a
re-run would download everything again. This script snapshots the file names
while they are still reachable, so the downloader can be pointed at the snapshot
instead -- see `download_shs100k.py --using-csv`.

Output: data/logs/{split}_downloaded_songs.csv, one file name per line
(no header), sorted, e.g.

    221685_927680.webm
    221685_927681.m4a

Examples:
    python scripts/list_downloaded_songs.py --val
    python scripts/list_downloaded_songs.py --train
    python scripts/list_downloaded_songs.py --test --audio-dir /mnt/shs100k/test
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from shs100k_meta import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    audio_dir,
    downloaded_csv,
)


def audio_files(directory: Path) -> list[Path]:
    """Finished audio files in a directory, sorted by name.

    Same rules as shs100k_meta.existing_audio: no directories, no dotfiles
    (.DS_Store), and no .part leftovers from interrupted downloads -- listing
    one of those would make the downloader skip a track it never finished.
    """
    if not directory.is_dir():
        raise SystemExit(f"no such directory: {directory}")
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and not path.name.startswith(".") and ".part" not in path.name
    ]
    return sorted(files, key=lambda path: path.name)


def write_listing(files: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for path in files:
            writer.writerow([path.name])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for split in ("train", "val", "test"):
        group.add_argument(
            f"--{split}",
            dest="split",
            action="store_const",
            const=split,
            help=f"list the {split} split",
        )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="folder to read, if the audio does not live under "
        "--data-root/audio/{split} (an external disk, a staging copy)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output CSV (default: --data-root/logs/{split}_downloaded_songs.csv)",
    )
    args = parser.parse_args()

    directory = args.audio_dir or audio_dir(args.data_root, args.split)
    out_path = args.out or downloaded_csv(args.data_root, args.split)

    files = audio_files(directory)
    write_listing(files, out_path)

    keys = {path.stem for path in files}
    duplicates = len(files) - len(keys)
    note = f", {duplicates} sharing a track key" if duplicates else ""
    print(f"[{args.split}] {len(files)} files in {directory}{note}")
    print(f"[{args.split}] wrote {out_path}")


if __name__ == "__main__":
    main()
