#!/usr/bin/env python3
"""Download SHS100K audio from YouTube using the vendored yt-dlp.

Reads the SHS-100K-official-2025 split CSVs via shs100k_meta, which rebuilds
each watch URL from the dataset's 11-character YouTube video id.

Audio-only streams are saved as data/audio/{split}/{work_id}_{performance_id}.<ext>
(native codec, no re-encoding). Already-downloaded tracks are skipped, so the
script is safe to interrupt and re-run. Outcomes are appended to a CSV log,
classified so a re-run can target the retryable failures and ignore link rot.

Once a split has been uploaded to S3 and deleted locally, that folder is empty
and a plain re-run would fetch everything again. --using-csv reads the already-
downloaded keys from data/logs/{split}_downloaded_songs.csv instead, which
scripts/list_downloaded_songs.py writes while the files are still on disk.

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
import re
import shutil
import sys
import threading
import time
from collections.abc import Sequence
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
    audio_from_csv,
    downloaded_csv,
    existing_audio,
    split_tracks,
)

_print_lock = threading.Lock()

# yt-dlp colours its error text, which is noise once the message is stored in a
# CSV or reprinted with our own status column.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def short_detail(detail: str, limit: int = 88) -> str:
    """One-line console summary of an error, without yt-dlp's boilerplate."""
    text = ANSI_RE.sub("", detail).replace("\n", " ").strip()
    text = text.removeprefix("ERROR: ")
    return text if len(text) <= limit else f"{text[: limit - 1]}\u2026"

# Errors that mean the video is gone for good: retrying wastes time, and these
# are the expected ~24% link rot, not a problem with the downloader.
#
# Deliberately NOT listed: "this video is not available", which reads like link
# rot but is also what a soft-blocked IP gets served for healthy videos. It
# falls through to "failed", which is retried; promoting it here would make it
# PERMANENT and silently blacklist good tracks during a block.
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
# fixed by slowing down, never by authenticating. Checked BEFORE the login
# markers below, because the bot-check message also contains "sign in".
THROTTLE_MARKERS = (
    "not a bot",
    "needs to be reloaded",
    "too many requests",
    "http error 429",
    "rate-limited",
    # A flagged session gets player responses stripped of every audio/video
    # URL, leaving only storyboards. yt-dlp then fails on formats rather than
    # on the block, so match that too: it is session-scoped and clears on its
    # own, which is "throttled", not a broken video.
    "sabr-only",
    "only images are available",
)

# Errors that need an account: age-restricted, members-only, private-with-access.
# Permanent for us, since this pipeline downloads anonymously on purpose --
# cookies switch yt-dlp to the policed player clients and make yield worse.
LOGIN_MARKERS = (
    "please sign in",
    "confirm your age",
    "age-restricted",
    "members-only",
    "join this channel",
    "available to this channel's members",
    "inappropriate for some users",
)

# JavaScript runtimes yt-dlp may use, highest priority first. It needs one to
# solve YouTube's signature and "n" challenges. Without a runtime it falls back
# to the JS-less client set -- which is just visionos now that android_vr is
# dead -- and healthy videos fail with "This video is not available", an error
# that names nothing resembling the real cause. deno is yt-dlp's only default;
# node is listed so an existing nvm install counts. Also needs the solver
# script: `pip install yt-dlp-ejs`.
JS_RUNTIMES = ("deno", "node")

# Statuses that will never succeed on a re-run, so they are skipped next time.
PERMANENT = ("gone", "blocked", "filtered")

# Link rot in SHS100K runs ~24%. A run far above that is not finding dead
# videos: a flagged IP gets served "Video unavailable" for perfectly healthy
# ones, which is indistinguishable from real link rot in the error text. Above
# this fraction the classifications are not trustworthy and must not be baked in.
# Measured against gone+blocked+filtered AND failed, because a block lands in
# both buckets -- see the closing warning in run_split().
SOFT_BLOCK_RATE = 0.50


@dataclass(frozen=True)
class DownloadOptions:
    """yt-dlp knobs shared by every worker.

    Average request rate is not the whole story: YouTube blocks on burstiness
    too. With no pre-download sleep a worker hits the player API the instant its
    previous download finishes, and independent workers drift into sync, so the
    *peak* rate spikes even when the average looks safe. The randomized
    sleep_interval de-synchronizes them, which is why 4 workers with no sleep
    gets blocked while more workers with sleep do not. Defaults match yt-dlp's
    own `-t sleep` preset, which is what YouTube's rate-limit message
    recommends. Raise --workers to buy throughput back; do not remove the sleep.
    """

    max_duration: int = 900
    js_runtimes: tuple[str, ...] = JS_RUNTIMES
    cookies: str | None = None
    cookies_from_browser: str | None = None
    player_client: str | None = None
    sleep_requests: float = 1.0
    sleep_interval: float = 10.0
    max_sleep_interval: float = 20.0
    retries: int = 3


class WarningCollector:
    """yt-dlp logger that keeps warnings so classify() can see them.

    The bot check and HTTP 429 arrive as non-fatal *warnings*: yt-dlp logs them,
    carries on, and then fails with a format error instead -- "Only images are
    available for download", "Requested format is not available", or a bare
    HTTP 403 once the player response comes back with no audio URLs. None of
    those carry a throttle marker, so classifying on the fatal error alone files
    a rate-limited run under "failed", where it is indistinguishable from real
    breakage. Keeping the warnings restores the evidence.

    Safe under threads: one instance per download_one() call, never shared.

    Note that a logger takes priority over `no_warnings` in
    YoutubeDL.report_warning, so warnings still arrive here while the console
    stays quiet.
    """

    def __init__(self) -> None:
        self.messages: list[str] = []

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        self.messages.append(msg)

    def error(self, msg: str) -> None:
        self.messages.append(msg)


def _has_marker(text: str, markers: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def classify(error: str, warnings: Sequence[str] = ()) -> str:
    """Bucket an error: gone/blocked are permanent, throttled/failed retryable.

    `warnings` are yt-dlp's non-fatal messages for the same attempt. They are
    searched alongside the error because a throttled attempt announces itself
    only there; see WarningCollector.
    """
    combined = " ".join((error, *warnings))
    if _has_marker(combined, THROTTLE_MARKERS):
        return "throttled"
    if _has_marker(combined, GONE_MARKERS):
        return "gone"
    if _has_marker(combined, LOGIN_MARKERS):
        return "blocked"
    return "failed"


def permanent_failures(log_path: Path) -> set[str]:
    """Keys whose most recent logged outcome can never succeed on a re-run.

    Without this, every run retries the whole link-rot tail -- about 24% of the
    dataset -- so late runs spend nearly all their time re-confirming that dead
    videos are still dead. Columns 0 and 2 are key and status in both the
    current log format and the older four-column one, so old logs still parse.
    """
    if not log_path.exists():
        return set()
    latest: dict[str, str] = {}
    with open(log_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[0] != "key":
                latest[row[0]] = row[2]
    return {key for key, status in latest.items() if status in PERMANENT}


def build_ydl_opts(
    out_dir: Path, key: str, options: DownloadOptions, logger: WarningCollector
) -> dict:
    opts: dict = {
        "logger": logger,
        "js_runtimes": js_runtime_config(options.js_runtimes),
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
    collector = WarningCollector()
    error = ""
    try:
        opts = build_ydl_opts(out_dir, track.key, options, collector)
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([track.url])
    except yt_dlp.utils.DownloadError as exc:
        error = ANSI_RE.sub("", str(exc)).replace("\n", " ")
    except Exception as exc:  # network hiccups, extractor bugs, ...
        error = f"{type(exc).__name__}: {exc}"

    if downloaded_file(out_dir, track.key) is not None:
        return "ok", ""
    if error:
        status = classify(error, collector.messages)
        # When only a warning revealed the throttle, the fatal error alone reads
        # as an ordinary format failure. Record the deciding warning too, so the
        # log says why this row is "throttled" rather than "failed".
        if status == "throttled" and not _has_marker(error, THROTTLE_MARKERS):
            hint = next(
                (m for m in collector.messages if _has_marker(m, THROTTLE_MARKERS)),
                "",
            )
            if hint:
                error = f"{error} | warning: {hint}".replace("\n", " ")
        return status, error
    return "filtered", f"skipped (duration > {options.max_duration}s or no media)"


def js_runtime_config(runtimes: Sequence[str]) -> dict[str, dict[str, str | None]]:
    """Mirror yt-dlp's CLI parsing of --js-runtimes for the Python API.

    The CLI accepts a list of "runtime[:path]" strings, but YoutubeDL itself
    wants {runtime: {"path": path}} and raises ValueError on the list. See
    yt_dlp/__init__.py:784.
    """
    parsed: dict[str, dict[str, str | None]] = {}
    for arg in runtimes:
        runtime, _, path = arg.partition(":")
        parsed[runtime.lower()] = {"path": path or None}
    return parsed


def check_js_runtime(runtimes: Sequence[str]) -> None:
    """Warn when yt-dlp has no way to solve YouTube's JS challenges.

    Worth checking up front because the failure is silent and misleading: the
    extractor quietly drops to the JS-less client and reports healthy, public
    videos as "This video is not available", which classify() then files under
    "failed" alongside genuine breakage. Changing IP does not help, so it reads
    exactly like a block that will not lift.
    """
    names = [runtime.partition(":")[0] for runtime in runtimes]
    if not any(shutil.which(name) for name in names):
        print(
            "[warning] no JavaScript runtime found (looked for: "
            f"{', '.join(names) or 'none'}). yt-dlp needs one to solve "
            "YouTube's signature/n challenges; without it many healthy videos "
            "fail as 'This video is not available'. Install deno or node."
        )
        return
    try:
        import yt_dlp_ejs  # noqa: F401
    except ImportError:
        print(
            "[warning] JavaScript runtime found, but the yt-dlp-ejs solver "
            "script is not installed, so challenges still cannot be solved. "
            "Fix with 'pip install yt-dlp-ejs'."
        )


def run_split(
    split: str,
    data_root: Path,
    workers: int,
    limit: int,
    options: DownloadOptions,
    retry_permanent: bool = False,
    using_csv: bool = False,
) -> None:
    out_dir = audio_dir(data_root, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = data_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"download_{split}.csv"

    tracks = split_tracks(split)
    if using_csv:
        done = audio_from_csv(data_root, split)
        source = downloaded_csv(data_root, split)
    else:
        done = existing_audio(data_root, split)
        source = out_dir
    skip = set() if retry_permanent else permanent_failures(log_path)
    todo = [t for t in tracks if t.key not in done and t.key not in skip]
    if limit > 0:
        todo = todo[:limit]
    print(
        f"[{split}] {len(tracks)} tracks in split, {len(done)} already downloaded "
        f"(per {source}), {len(skip)} permanently unavailable, "
        f"{len(todo)} to fetch"
    )
    if not todo:
        reachable = len(done) + len(skip)
        if reachable >= len(tracks):
            print(
                f"[{split}] split complete: {len(done)} downloaded "
                f"({len(done) / max(len(tracks), 1):.0%} yield), {len(skip)} "
                f"permanently unavailable. Nothing left to retry."
            )
        return
    pacing = f"{options.sleep_requests:g}s between requests"
    if options.sleep_interval > 0:
        pacing += (
            f", {options.sleep_interval:g}-{options.max_sleep_interval:g}s "
            f"before each download"
        )
    print(f"[{split}] {workers} workers, {pacing}")
    started = time.monotonic()

    counts = {
        "ok": 0, "gone": 0, "blocked": 0, "throttled": 0, "failed": 0, "filtered": 0
    }
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
                # One line per track. yt-dlp's own stderr no longer reaches the
                # console (WarningCollector captures it so classify() can read
                # it), so this is the only per-track progress there is.
                with _print_lock:
                    line = (
                        f"[{split}] {i:>{len(str(len(todo)))}}/{len(todo)} "
                        f"{status:9s} {track.youtube_id}  {track.key:15s}"
                    )
                    if detail:
                        line += f"  {short_detail(detail)}"
                    print(line.rstrip())
                if status != "ok":
                    retryable = "no" if status in PERMANENT else "yes"
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
    permanent = counts["gone"] + counts["blocked"] + counts["filtered"]
    remaining = len(tracks) - len(done) - len(skip) - counts["ok"] - permanent
    print(f"[{split}] done: {counts}  (logged to {log_path})")
    print(
        f"[{split}] {rate:.0f} tracks/hr sustained ({rate / 3600:.2f} req/s); "
        f"at this rate the remaining {remaining} would take "
        f"{remaining / max(rate, 1e-9) / 24:.1f} days. To go faster raise "
        f"--workers and keep the sleep on; watch 'throttled'."
    )
    attempted = len(todo)
    # A soft block lands in BOTH buckets, so neither alone detects it: YouTube
    # serves healthy videos as unavailable (-> permanent) and returns player
    # responses carrying no audio URLs at all (-> failed, as "Only images are
    # available" / "Requested format is not available" / HTTP 403). Only the
    # aggregate rate catches a block that slipped past classify().
    suspect = permanent + counts["failed"]
    if suspect and attempted >= 20 and suspect / attempted > SOFT_BLOCK_RATE:
        print(
            f"\n[{split}] !! {suspect}/{attempted} "
            f"({suspect / attempted:.0%}) of this run produced no audio "
            f"({permanent} unavailable, {counts['failed']} failed). Link rot is "
            f"~24%, so this is almost certainly an IP-level SOFT BLOCK, not dead "
            f"videos: a flagged address is served 'Video unavailable' for healthy "
            f"videos, and player responses with no downloadable formats.\n"
            f"[{split}]    Verify by opening one of the failed video ids in a "
            f"browser. If it plays, these classifications are wrong -- a browser "
            f"passes the bot check that anonymous yt-dlp just failed.\n"
            f"[{split}]    Stop, wait several hours, then resume. 'failed' is "
            f"retried automatically; 'gone'/'blocked' are not, so add "
            f"--retry-permanent. Continuing now will blacklist good tracks.\n"
            f"[{split}]    Do NOT build a skip-list from 'failed': under a block "
            f"that bucket is mostly healthy videos.\n"
        )
    elif permanent:
        print(
            f"[{split}] {permanent} newly confirmed unavailable "
            f"({counts['gone']} gone, {counts['blocked']} need an account, "
            f"{counts['filtered']} filtered); they will be skipped from now on. "
            f"Use --retry-permanent to force another attempt."
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
        "--js-runtimes",
        type=str,
        default=",".join(JS_RUNTIMES),
        help="comma-separated JavaScript runtimes yt-dlp may use, highest "
        "priority first. One is required to solve YouTube's signature/n "
        "challenges, along with the solver script ('pip install yt-dlp-ejs'); "
        "without them healthy videos fail as 'This video is not available'. "
        "Pass an empty string to disable",
    )
    parser.add_argument(
        "--using-csv",
        action="store_true",
        help="decide what is already downloaded from "
        "data/logs/{split}_downloaded_songs.csv (written by "
        "scripts/list_downloaded_songs.py) instead of listing the audio "
        "folder. For resuming after the audio has been moved off this disk; "
        "newly downloaded files are NOT added to the CSV, so re-run the "
        "listing script after an upload",
    )
    parser.add_argument(
        "--retry-permanent",
        action="store_true",
        help="also retry tracks previously logged as gone/blocked/filtered. "
        "Off by default: those are the ~24%% link-rot tail and retrying them "
        "makes every later run slower without finding anything",
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
        default=10.0,
        help="minimum seconds before each download. Randomized up to "
        "--max-sleep-interval, which de-synchronizes workers and smooths the "
        "request bursts YouTube blocks on. Setting this to 0 gets you "
        "bot-checked even at low worker counts",
    )
    parser.add_argument(
        "--max-sleep-interval",
        type=float,
        default=20.0,
        help="upper bound of the randomized pre-download sleep",
    )
    parser.add_argument(
        "--player-client",
        type=str,
        default=None,
        help="override yt-dlp's YouTube player clients, comma-separated "
        "(e.g. 'visionos,web'). Escape hatch only; the defaults are already "
        "chosen to avoid PO-token and bot-check paths. Do not reach for "
        "'android_vr': YouTube has 403'd every format from it since 2026-08-17, "
        "which is why yt-dlp dropped it from the anonymous defaults",
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

    js_runtimes = tuple(r for r in args.js_runtimes.split(",") if r)
    check_js_runtime(js_runtimes)
    options = DownloadOptions(
        max_duration=args.max_duration,
        js_runtimes=js_runtimes,
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
        run_split(
            split,
            args.data_root,
            args.workers,
            args.limit,
            options,
            args.retry_permanent,
            args.using_csv,
        )


if __name__ == "__main__":
    main()
