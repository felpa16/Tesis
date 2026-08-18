#!/usr/bin/env python3
"""Download SHS100K audio from YouTube using the vendored yt-dlp.

Reads the SHS-100K-official-2025 split CSVs via shs100k_meta, which rebuilds
each watch URL from the dataset's 11-character YouTube video id.

Audio-only streams are saved as data/audio/{split}/{work_id}_{performance_id}.<ext>
(native codec, no re-encoding). Already-downloaded tracks are skipped, so the
script is safe to interrupt and re-run. Outcomes are appended to a CSV log,
classified so a re-run can target the retryable failures and ignore link rot.

Run anonymously. Passing cookies switches yt-dlp to a different, more heavily
policed set of player clients and usually makes yield *worse* -- see the
--cookies-from-browser help text.

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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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

# Errors that mean the video is gone for good: retrying wastes time, and these
# are the expected ~24% link rot, not a problem with the downloader.
GONE_MARKERS = (
    "video unavailable",
    "private video",
    "has been removed",
    "has been terminated",
    "no longer available",
    "removed for violating",
    "does not exist",
)

# Errors that mean YouTube is throttling this session. Not permanent: they are
# fixed by slowing down, never by authenticating.
THROTTLE_MARKERS = (
    "not a bot",
    "needs to be reloaded",
    "too many requests",
    "http error 429",
    "rate-limited",
)


@dataclass(frozen=True)
class DownloadOptions:
    """yt-dlp knobs shared by every worker.

    Only one quantity actually controls both throughput and bot-check exposure:
    the aggregate request rate, workers / (seconds per track + sleep). Sleeping
    before each download and raising --workers cancel out, so pacing is left to
    --workers, and sleep_requests -- which spaces the extraction API calls where
    the bot check actually lives -- carries a small default instead.
    """

    max_duration: int = 900
    cookies: str | None = None
    cookies_from_browser: str | None = None
    player_client: str | None = None
    sleep_requests: float = 1.0
    sleep_interval: float = 0.0
    max_sleep_interval: float = 0.0
    retries: int = 3


def classify(error: str) -> str:
    """'gone' (permanent), 'throttled', or 'failed' (retryable) for an error."""
    lowered = error.lower()
    if any(marker in lowered for marker in GONE_MARKERS):
        return "gone"
    if any(marker in lowered for marker in THROTTLE_MARKERS):
        return "throttled"
    return "failed"


def build_ydl_opts(out_dir: Path, key: str, options: DownloadOptions) -> dict:
    opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / f"{key}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": options.retries,
        "fragment_retries": options.retries,
        "socket_timeout": 30,
        "sleep_interval_requests": options.sleep_requests,
        "sleep_interval": options.sleep_interval,
        "max_sleep_interval": max(options.max_sleep_interval, options.sleep_interval),
    }
    if options.max_duration > 0:
        opts["match_filter"] = yt_dlp.utils.match_filter_func(
            f"duration <= {options.max_duration}"
        )
    if options.cookies:
        opts["cookiefile"] = options.cookies
    if options.cookies_from_browser:
        opts["cookiesfrombrowser"] = (options.cookies_from_browser,)
    if options.player_client:
        opts["extractor_args"] = {
            "youtube": {"player_client": options.player_client.split(",")}
        }
    return opts


def downloaded_file(out_dir: Path, key: str) -> Path | None:
    """The finished audio file for a key, ignoring in-progress .part files."""
    for path in out_dir.glob(f"{key}.*"):
        if path.is_file() and ".part" not in path.name:
            return path
    return None


def download_one(
    track: Track, out_dir: Path, options: DownloadOptions
) -> tuple[str, str]:
    """Download a single track. Returns (status, detail)."""
    error = ""
    try:
        with yt_dlp.YoutubeDL(build_ydl_opts(out_dir, track.key, options)) as ydl:
            ydl.download([track.url])
    except yt_dlp.utils.DownloadError as exc:
        error = str(exc).replace("\n", " ")
    except Exception as exc:  # network hiccups, extractor bugs, ...
        error = f"{type(exc).__name__}: {exc}"

    if downloaded_file(out_dir, track.key) is not None:
        return "ok", ""
    if error:
        return classify(error), error
    return "filtered", f"skipped (duration > {options.max_duration}s or no media)"


def run_split(
    split: str, data_root: Path, workers: int, limit: int, options: DownloadOptions
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
    pacing = f"{options.sleep_requests:g}s between requests"
    if options.sleep_interval > 0:
        pacing += (
            f", {options.sleep_interval:g}-{options.max_sleep_interval:g}s "
            f"before each download"
        )
    print(f"[{split}] {workers} workers, {pacing}")
    started = time.monotonic()

    counts = {"ok": 0, "gone": 0, "throttled": 0, "failed": 0, "filtered": 0}
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        log = csv.writer(log_file)
        if write_header:
            log.writerow(["key", "url", "status", "retryable", "detail"])

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(download_one, t, out_dir, options): t for t in todo
            }
            for i, future in enumerate(as_completed(futures), 1):
                track = futures[future]
                status, detail = future.result()
                counts[status] += 1
                if status != "ok":
                    retryable = "no" if status in ("gone", "filtered") else "yes"
                    log.writerow([track.key, track.url, status, retryable, detail])
                    log_file.flush()
                if i % 50 == 0 or i == len(todo):
                    rate = i / max(time.monotonic() - started, 1e-9) * 3600
                    with _print_lock:
                        print(
                            f"[{split}] {i}/{len(todo)}  "
                            + "  ".join(f"{k}={v}" for k, v in counts.items() if v)
                            + f"  {rate:.0f}/hr"
                        )

    rate = len(todo) / max(time.monotonic() - started, 1e-9) * 3600
    remaining = len(tracks) - len(done) - counts["ok"]
    print(f"[{split}] done: {counts}  (logged to {log_path})")
    print(
        f"[{split}] {rate:.0f} tracks/hr sustained; at this rate the remaining "
        f"{remaining} would take {remaining / max(rate, 1e-9) / 24:.1f} days. "
        f"Throughput and bot-check exposure are the same number "
        f"({rate / 3600:.2f} req/s) -- tune --workers, and watch 'throttled'."
    )
    if counts["throttled"]:
        print(
            f"[{split}] {counts['throttled']} downloads were rate-limited by "
            f"YouTube. Re-run to retry them, and if it persists lower --workers "
            f"or raise --sleep-interval. Do NOT reach for cookies: they switch "
            f"yt-dlp to more heavily policed player clients and make this worse."
        )


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
        "--retries", type=int, default=3, help="yt-dlp retries per download"
    )
    parser.add_argument(
        "--sleep-requests",
        type=float,
        default=1.0,
        help="seconds between extraction API requests, where the bot check "
        "lives. Cheap: a few seconds per track, per worker",
    )
    parser.add_argument(
        "--sleep-interval",
        type=float,
        default=0.0,
        help="minimum seconds before each media download. Off by default: it "
        "delays the media fetch rather than the API calls being rate-limited, "
        "so --workers is the better rate knob",
    )
    parser.add_argument(
        "--max-sleep-interval",
        type=float,
        default=0.0,
        help="upper bound of the randomized pre-download sleep",
    )
    parser.add_argument(
        "--player-client",
        type=str,
        default=None,
        help="override yt-dlp's YouTube player clients, comma-separated "
        "(e.g. 'visionos,android_vr'). Escape hatch only; the defaults are "
        "already chosen to avoid PO-token and bot-check paths",
    )
    parser.add_argument(
        "--cookies", type=str, default=None, help="cookies.txt for restricted videos"
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        help="browser to read YouTube cookies from. NOT RECOMMENDED: cookies "
        "make yt-dlp use _DEFAULT_AUTHED_CLIENTS (tv_downgraded, web) instead "
        "of the safer anonymous set, the rate limit becomes account-global "
        "instead of per-session, and a running browser rotates the cookies out "
        "from under the download",
    )
    args = parser.parse_args()

    options = DownloadOptions(
        max_duration=args.max_duration,
        cookies=args.cookies,
        cookies_from_browser=args.cookies_from_browser,
        player_client=args.player_client,
        sleep_requests=args.sleep_requests,
        sleep_interval=args.sleep_interval,
        max_sleep_interval=args.max_sleep_interval,
        retries=args.retries,
    )
    if options.cookies or options.cookies_from_browser:
        print(
            "[warning] downloading with cookies. This switches yt-dlp to the "
            "authenticated player clients, scopes YouTube's rate limit to the "
            "account rather than the session, and breaks when the browser "
            "rotates the cookies. Anonymous downloading is the supported path."
        )

    splits = list(SPLITS) if args.split == "all" else [args.split]
    for split in splits:
        run_split(split, args.data_root, args.workers, args.limit, options)


if __name__ == "__main__":
    main()
