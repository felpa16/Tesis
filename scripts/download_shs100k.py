#!/usr/bin/env python3
"""Download SHS100K audio from YouTube using the vendored yt-dlp.

Reads the SHS-100K-official-2025 split CSVs via shs100k_meta, which rebuilds
each watch URL from the dataset's 11-character YouTube video id.

Audio-only streams are saved as data/audio/{split}/{work_id}_{performance_id}.<ext>
(native codec, no re-encoding). Already-downloaded tracks are skipped, so the
script is safe to interrupt and re-run. Failures are appended to a CSV log so
the real dataset yield (link rot is expected) can be measured afterwards.

Examples:
    python scripts/download_shs100k.py --split val --limit 20      # smoke test
    python scripts/download_shs100k.py --split all --workers 8
    python scripts/download_shs100k.py --split train --data-root /mnt/shs100k
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent / "yt-dlp"))

import yt_dlp  # noqa: E402  (vendored checkout)

from shs100k_meta import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    SPLITS,
    Track,
    audio_dir,
    existing_audio,
    split_tracks,
)

_print_lock = threading.Lock()


def build_ydl_opts(
    out_dir: Path,
    key: str,
    max_duration: int,
    cookies: str | None,
    cookies_from_browser: str | None,
) -> dict:
    opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / f"{key}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
    }
    if max_duration > 0:
        opts["match_filter"] = yt_dlp.utils.match_filter_func(
            f"duration <= {max_duration}"
        )
    if cookies:
        opts["cookiefile"] = cookies
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    return opts


def download_one(
    track: Track,
    out_dir: Path,
    max_duration: int,
    cookies: str | None,
    cookies_from_browser: str | None,
) -> tuple[str, str]:
    """Download a single track. Returns (status, detail)."""
    error = ""
    try:
        opts = build_ydl_opts(
            out_dir, track.key, max_duration, cookies, cookies_from_browser
        )
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([track.url])
    except yt_dlp.utils.DownloadError as exc:
        error = str(exc).replace("\n", " ")
    except Exception as exc:  # network hiccups, extractor bugs, ...
        error = f"{type(exc).__name__}: {exc}"

    if any(out_dir.glob(f"{track.key}.*")):
        return "ok", ""
    if error:
        return "failed", error
    return "filtered", f"skipped (duration > {max_duration}s or no media)"


def run_split(
    split: str,
    data_root: Path,
    workers: int,
    limit: int,
    max_duration: int,
    cookies: str | None,
    cookies_from_browser: str | None,
) -> None:
    out_dir = audio_dir(data_root, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = data_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"download_{split}.csv"

    tracks = split_tracks(split)
    done = existing_audio(data_root, split)
    todo = [t for t in tracks if t.key not in done]
    if limit > 0:
        todo = todo[:limit]
    print(
        f"[{split}] {len(tracks)} tracks in split, {len(done)} already downloaded, "
        f"{len(todo)} to fetch"
    )
    if not todo:
        return

    counts = {"ok": 0, "failed": 0, "filtered": 0}
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        log = csv.writer(log_file)
        if write_header:
            log.writerow(["key", "url", "status", "detail"])

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    download_one, t, out_dir, max_duration, cookies,
                    cookies_from_browser,
                ): t
                for t in todo
            }
            for i, future in enumerate(as_completed(futures), 1):
                track = futures[future]
                status, detail = future.result()
                counts[status] += 1
                if status != "ok":
                    log.writerow([track.key, track.url, status, detail])
                    log_file.flush()
                if i % 50 == 0 or i == len(todo):
                    with _print_lock:
                        print(
                            f"[{split}] {i}/{len(todo)}  "
                            f"ok={counts['ok']} failed={counts['failed']} "
                            f"filtered={counts['filtered']}"
                        )

    print(f"[{split}] done: {counts}  (failures logged to {log_path})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--limit", type=int, default=0, help="max tracks to download (0 = no limit)"
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=900,
        help="skip videos longer than this many seconds (0 = no filter)",
    )
    parser.add_argument(
        "--cookies", type=str, default=None, help="cookies.txt for restricted videos"
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        help="browser to read YouTube cookies from (e.g. chrome, firefox, safari)",
    )
    args = parser.parse_args()

    splits = list(SPLITS) if args.split == "all" else [args.split]
    for split in splits:
        run_split(
            split,
            args.data_root,
            args.workers,
            args.limit,
            args.max_duration,
            args.cookies,
            args.cookies_from_browser,
        )


if __name__ == "__main__":
    main()
