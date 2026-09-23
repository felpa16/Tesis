#!/usr/bin/env python3
"""Append the audio files found on disk to a split's downloaded-songs CSV.

The downloader normally decides what to skip by looking at data/audio/{split}.
That only works while the audio is sitting on the same machine: once a split has
been uploaded to S3 (or moved to an external disk) the folder is empty and a
re-run would download everything again. This script records the file names while
they are still reachable, so the downloader can be pointed at the CSV instead --
see `download_shs100k.py --using-csv`.

The CSV is only ever appended to, never rewritten. It is the record of
everything ever downloaded, which quickly outgrows what the folder holds: after
an upload-and-delete cycle the audio folder is a small tail of the CSV, so
overwriting it would drop the history and re-download thousands of tracks.
Lines already present are left untouched and files already listed are skipped,
so re-running is harmless.

Output: data/logs/{split}_downloaded_songs.csv, one file name per line
(no header), in the order the files were appended, e.g.

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
    keys_from_csv,
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


def ends_without_newline(path: Path) -> bool:
    """Whether a non-empty file's last byte is not a line break.

    Appending to such a file would glue the first new name onto the last
    existing one, silently corrupting both rows.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as f:
        f.seek(-1, 2)
        return f.read(1) not in (b"\n", b"\r")


def append_listing(files: list[Path], out_path: Path) -> list[Path]:
    """Append the files the CSV does not list yet. Returns what was appended.

    Dedup is by track key rather than file name, so a track re-downloaded into
    a different container (.webm the first time, .m4a the second) is not listed
    twice under two names.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    listed = keys_from_csv(out_path)
    new: list[Path] = []
    for path in files:
        if path.stem not in listed:
            listed.add(path.stem)
            new.append(path)
    if not new:
        return []

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if ends_without_newline(out_path):
            f.write("\r\n")
        for path in new:
            writer.writerow([path.name])
    return new


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
        help="CSV to append to "
        "(default: --data-root/logs/{split}_downloaded_songs.csv)",
    )
    args = parser.parse_args()

    directory = args.audio_dir or audio_dir(args.data_root, args.split)
    out_path = args.out or downloaded_csv(args.data_root, args.split)

    before = len(keys_from_csv(out_path))
    files = audio_files(directory)
    appended = append_listing(files, out_path)

    duplicates = len(files) - len({path.stem for path in files})
    note = f", {duplicates} sharing a track key" if duplicates else ""
    print(f"[{args.split}] {len(files)} files in {directory}{note}")
    print(
        f"[{args.split}] appended {len(appended)} new names to {out_path} "
        f"({before} already listed, {before + len(appended)} total)"
    )


if __name__ == "__main__":
    main()
