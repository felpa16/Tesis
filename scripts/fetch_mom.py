#!/usr/bin/env python3
"""Mirror the MoM (Melody or Machine) synthetic-music benchmark from HF into S3.

Source: https://huggingface.co/datasets/anonymous2212/MoM-CLAM-dataset
(public, ungated, CC BY-NC 4.0; 36,417 mp3 / 124.9 GiB across four generators).

What the HF repo actually holds, which is less than the paper's 130 k songs:

    udio/chunk_{1,2,3}/*.mp3   19,500 files   78.3 GiB   Udio v1.5
    riffusion/*.mp3             7,043 files   20.3 GiB   Riffusion FUZZ-1.0
    yue/*.mp3                   5,278 files   19.8 GiB   YuE
    diffrythm/*.mp3             4,596 files    6.5 GiB   DiffRhythm

Everything else is links, not audio: the real half (`real_songs.csv`), the
YT-cover tier, the AI-cover tier, and *all* of Suno live as URL/filename CSVs
that the authors did not redistribute. That costs us nothing -- the human side
of our evaluation is SHS100K `val50`/`test50` by design (CLAUDE.md) -- but it
does mean MoM-via-HF is a four-generator benchmark, not a nine-generator one.

Since representation learning never sees synthetic music, *every* one of the
four is an unseen generator for us; MoM's own train/test generator split does
not apply. Score each generator separately, so the key is laid out to make
`--generators riffusion` a single S3 prefix listing:

    s3://BUCKET/mom/audio/{generator}/{basename}.mp3
    s3://BUCKET/mom/metadata/{*.csv}
    s3://BUCKET/mom/index/files.jsonl        what landed, with size + sha256

Each worker downloads one file to scratch, verifies its length and sha256
against what the repo tree declares, uploads it, and deletes it. Peak disk is
therefore `--workers` mp3s (tens of MB), not 125 GiB -- this box is usually the
one whose NVMe already holds the phase-2 feature cache. Re-runs list the
destination prefix first and transfer only what is missing, so interrupting is
free.

Audio is stored byte-for-byte as published. Do NOT normalize on the way in:
these are mp3s of mixed bitrate (and DiffRhythm's are all *exactly* 1,522,668
bytes, i.e. fixed-duration CBR) while SHS100K is YouTube opus/m4a, so one
shared decode-time normalization is the only thing that keeps the detector off
codec artifacts instead of the style-given-content signal. Keeping S3 raw
leaves that choice in code, where it is ablatable.

Rate limits
-----------
The Hub meters its buckets separately over fixed 5-minute windows
(https://huggingface.co/docs/hub/rate-limits). Two matter here, and their
quotas differ by 5x:

    bucket      what hits it                        anon    free     PRO
    api         /api/... -- the repo tree listing     500   1,000   2,500
    resolvers   /resolve/... -- the file bytes      3,000   5,000  12,000

Three consequences, all of which this script acts on:

  * **Pass a token.** HF's own first answer to a 429 is that the client never
    sent one, and an untokenized run is metered anonymously *per IP* -- shared
    with every other HF client on that address. `HF_TOKEN` lives in this
    repo's `.env`, which nothing exports, so `load_env_file` reads it and
    `--tier` defaults to the anonymous quotas when no token is found.

  * **Spend the resolver bucket, not the API bucket.** `hf_hub_download` does
    a HEAD for metadata and then a GET, i.e. two resolver requests per file,
    and the metadata is redundant because the tree listing already gave us
    every size and sha256. So files are fetched with one plain streaming GET
    against `/resolve/`, and the tree listing -- the only API-bucket consumer
    left -- is cached in `--plan` so a resumed run re-lists nothing.

  * **Pace, and obey the server.** `RateLimiter` holds each bucket under its
    quota client-side, and feeds back both headers every response carries:
    `RateLimit-Policy: "<bucket>";q=;w=` replaces the `--tier` guess with the
    quota this account really has (so a PRO upgrade needs no flag, and the
    documented drift in the free-tier numbers cannot desynchronise us), while
    `RateLimit: "<bucket>";r=;t=` parks *all* workers until the window resets
    once it runs low, rather than letting 24 threads hammer a closed door.
    huggingface_hub >= 1.2.0 does this for its own calls, but requirements.txt
    caps transformers < 5 and that major bump is not a thing to do
    mid-experiment.

At the free-tier resolver quota the 36,417 files need ~41 min of pure pacing,
so the transfer is bandwidth-bound rather than quota-bound. Anonymous is
~68 min, and no amount of --workers beats the quota.

Examples:
    # smoke test: metadata plus 20 files per generator
    python scripts/fetch_mom.py --bucket $BUCKET --limit 20

    # the 46.6 GiB that is ready soonest, then udio in a second pass
    python scripts/fetch_mom.py --bucket $BUCKET \\
        --generators riffusion yue diffrythm --workers 24
    python scripts/fetch_mom.py --bucket $BUCKET --generators udio --workers 24

    # size-matched subset instead of the full mirror (deterministic)
    python scripts/fetch_mom.py --bucket $BUCKET --per-generator 3000 --seed 0

    # if you upgraded the account
    python scripts/fetch_mom.py --bucket $BUCKET --tier pro --workers 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from src.s3 import S3  # noqa: E402

REPO_ID = "anonymous2212/MoM-CLAM-dataset"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"

# Top-level directory in the HF repo -> generator label. Udio is split across
# three chunk_* subdirectories that carry no meaning beyond upload batching.
GENERATOR_DIRS: dict[str, str] = {
    "diffrythm": "diffrythm",
    "riffusion": "riffusion",
    "yue": "yue",
    "udio": "udio",
}

# Small, worth having next to the audio: the audio index and the link tables
# for the tiers whose audio was not redistributed.
METADATA_FILES = (
    "metadata.csv",
    "ai_generated_music_metadata.csv",
    "real_yt_covers.csv",
    "real_songs.csv",
    "suno(v2_v4)_links/suno_2_metadata.csv",
    "suno(v2_v4)_links/suno_3_metadata.csv",
    "suno(v2_v4)_links/suno_3.5_metadata.csv",
    "suno(v2_v4)_links/suno_4_metadata.csv",
)

# Requests per fixed 5-minute window, per bucket, as documented September '25.
# "anon" is per IP address and is what an untokenized client gets.
TIERS: dict[str, tuple[int, int]] = {  # tier -> (api, resolvers)
    "anon": (500, 3_000),
    "free": (1_000, 5_000),
    "pro": (2_500, 12_000),
    "team": (3_000, 20_000),
    "enterprise": (6_000, 50_000),
}
RATE_WINDOW = 300.0

# RateLimit: "resolvers";r=4312;t=117
RATELIMIT_RE = re.compile(
    r'"(?P<bucket>[a-z]+)"\s*;\s*r=(?P<remaining>\d+)\s*;\s*t=(?P<reset>\d+)'
)
# RateLimit-Policy: "fixed window";"resolvers";q=5000;w=300
# The leading "fixed window" cannot match `bucket` -- the space breaks [a-z]+.
POLICY_RE = re.compile(
    r'"(?P<bucket>[a-z]+)"\s*;\s*q=(?P<quota>\d+)\s*;\s*w=(?P<window>\d+)'
)


@dataclass(frozen=True)
class Item:
    """One HF file and where it goes in the bucket."""

    repo_path: str
    key: str
    generator: str
    size: int
    sha256: str


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def load_env_file(path: Path) -> None:
    """Minimal .env reader, so HF_TOKEN does not have to be exported by hand.

    An unset token is the single most likely reason for a 429: it drops the run
    to the anonymous per-IP quota, which is shared with everything else using
    that address. Existing environment variables win.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip("'\"")
        if name and name not in os.environ:
            os.environ[name] = value


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


class RateLimiter:
    """Client-side fixed-window limiter for one Hub bucket, shared by threads.

    Two halves, because neither alone is enough. The local window counter keeps
    us from spending a whole quota in the first few seconds, which is what turns
    a transfer into a 429 storm. The server's headers are authoritative -- our
    counter cannot see other clients on the same token or IP, and the free-tier
    numbers are documented as subject to change -- so `observe` feeds back both
    of them: `RateLimit-Policy` replaces the `--tier` guess with the quota this
    account actually has, and `RateLimit` parks every worker until the window
    resets once it runs low.
    """

    def __init__(
        self,
        bucket: str,
        quota: int,
        window: float = RATE_WINDOW,
        margin: float = 1.0,
        locked: bool = False,
    ) -> None:
        self.bucket = bucket
        self.margin = margin
        self.locked = locked  # an explicit --*-quota override wins over the server
        self.quota = max(quota, 1)
        self.window = window
        self._stamps: deque[float] = deque()
        self._cv = threading.Condition()
        self._open_at = 0.0
        self.waited = 0.0
        self.pauses = 0
        self.policy: tuple[int, float] | None = None

    def acquire(self) -> None:
        """Block until this thread may issue one request against the bucket."""
        start = time.monotonic()
        with self._cv:
            while True:
                now = time.monotonic()
                if now < self._open_at:
                    self._cv.wait(timeout=self._open_at - now)
                    continue
                horizon = now - self.window
                while self._stamps and self._stamps[0] <= horizon:
                    self._stamps.popleft()
                if len(self._stamps) < self.quota:
                    self._stamps.append(now)
                    break
                self._cv.wait(timeout=max(self._stamps[0] + self.window - now, 0.01))
            self.waited += time.monotonic() - start

    def pause(self, seconds: float, reason: str = "") -> None:
        """Park every worker on this bucket for `seconds` (a 429, or near-empty)."""
        seconds = min(max(seconds, 1.0), self.window + 15.0)
        with self._cv:
            until = time.monotonic() + seconds
            if until <= self._open_at:
                return  # already parked at least that long
            self._open_at = until
            self.pauses += 1
            self._cv.notify_all()
        if reason:
            print(f"  [{self.bucket}] pausing {seconds:.0f}s: {reason}", flush=True)

    def adopt(self, quota: int, window: float) -> None:
        """Take the server's advertised policy over whatever --tier guessed."""
        if self.locked or self.policy == (quota, window):
            return
        budget = max(int(quota * self.margin), 1)
        with self._cv:
            self.policy = (quota, window)
            self.quota = budget
            self.window = window
            self._cv.notify_all()
        print(
            f"  [{self.bucket}] server policy {quota} per {window:.0f}s "
            f"-> spending {budget}",
            flush=True,
        )

    def observe(self, headers) -> None:
        """Reconcile with the server's own view of the window."""
        policy = headers.get("RateLimit-Policy") or headers.get("ratelimit-policy")
        if policy:
            for match in POLICY_RE.finditer(policy):
                if match.group("bucket") == self.bucket:
                    self.adopt(int(match.group("quota")), float(match.group("window")))
                    break
        header = headers.get("RateLimit") or headers.get("ratelimit")
        if not header:
            return
        for match in RATELIMIT_RE.finditer(header):
            if match.group("bucket") != self.bucket:
                continue
            remaining = int(match.group("remaining"))
            reset = float(match.group("reset"))
            # Below the number of requests already in flight, waiting out the
            # window is strictly faster than collecting 429s.
            if remaining <= 2:
                self.pause(reset + 1.0, f"{remaining} left, resets in {reset:.0f}s")
            return

    def retry_after(self, headers, default: float) -> float:
        """How long a 429 says to wait: Retry-After, else the window reset."""
        explicit = headers.get("Retry-After")
        if explicit:
            try:
                return float(explicit)
            except ValueError:
                pass
        header = headers.get("RateLimit") or headers.get("ratelimit") or ""
        for match in RATELIMIT_RE.finditer(header):
            if match.group("bucket") == self.bucket:
                return float(match.group("reset")) + 1.0
        return default


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def resolve_revision(limiter: RateLimiter, token: str | None) -> str:
    """Pin the mirror to one commit, so a resumed run cannot straddle two."""
    from huggingface_hub import HfApi

    limiter.acquire()
    return HfApi(token=token).dataset_info(REPO_ID).sha


def list_audio(
    generators: list[str], limiter: RateLimiter, token: str | None
) -> list[Item]:
    """Enumerate the mp3s of the requested generators, with sizes and hashes.

    The only API-bucket consumer in the script. list_repo_tree paginates at
    1,000 entries, so one slot is taken per page rather than per entry.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    items: list[Item] = []
    for generator in generators:
        seen = 0
        limiter.acquire()
        for entry in api.list_repo_tree(
            REPO_ID,
            repo_type="dataset",
            path_in_repo=GENERATOR_DIRS[generator],
            recursive=True,
        ):
            seen += 1
            if seen % 1000 == 0:
                limiter.acquire()
            path = entry.path
            if not path.endswith(".mp3"):
                continue
            items.append(
                Item(
                    repo_path=path,
                    key=f"audio/{generator}/{Path(path).name}",
                    generator=generator,
                    # size is None only for non-LFS files; every mp3 here is LFS.
                    size=entry.size or 0,
                    sha256=entry.lfs.sha256 if getattr(entry, "lfs", None) else "",
                )
            )
    return items


def load_plan(path: Path, generators: list[str]) -> tuple[str, list[Item]] | None:
    """Reuse a cached listing, which is what keeps the API bucket untouched."""
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            header = json.loads(f.readline())
            items = [Item(**json.loads(line)) for line in f if line.strip()]
    except Exception as exc:  # noqa: BLE001 -- a stale plan is not worth dying over
        print(f"  ignoring unreadable plan {path}: {exc}")
        return None
    if not set(generators).issubset(set(header.get("generators", ()))):
        return None
    wanted = set(generators)
    return header["revision"], [i for i in items if i.generator in wanted]


def save_plan(
    path: Path, revision: str, generators: list[str], items: list[Item]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"revision": revision, "generators": generators}) + "\n")
        for item in items:
            f.write(json.dumps(asdict(item)) + "\n")


def check_unique(items: list[Item]) -> None:
    """Flattening udio's chunk_* dirs is only safe if basenames do not collide."""
    seen: dict[str, str] = {}
    clashes: list[tuple[str, str]] = []
    for item in items:
        previous = seen.get(item.key)
        if previous is not None:
            clashes.append((previous, item.repo_path))
        else:
            seen[item.key] = item.repo_path
    if clashes:
        lines = "\n".join(f"  {a}\n  {b}" for a, b in clashes[:10])
        raise SystemExit(
            f"{len(clashes)} basename collision(s) while flattening the HF tree, "
            f"e.g.\n{lines}\n"
            "Re-run with --keep-subdirs to preserve the chunk_* directories in "
            "the S3 key."
        )


def subsample(items: list[Item], per_generator: int, seed: int) -> list[Item]:
    """Deterministic per-generator subset, for a size-matched eval set."""
    by_generator: dict[str, list[Item]] = {}
    for item in items:
        by_generator.setdefault(item.generator, []).append(item)
    kept: list[Item] = []
    for generator in sorted(by_generator):
        group = sorted(by_generator[generator], key=lambda i: i.repo_path)
        rng = random.Random(f"{seed}:{generator}")
        rng.shuffle(group)
        kept.extend(group[:per_generator])
    return kept


# --------------------------------------------------------------------------
# Transfer
# --------------------------------------------------------------------------


class Transfer:
    """Download-verify-upload-delete, one file at a time per worker."""

    def __init__(
        self,
        s3: S3 | None,
        prefix: str,
        scratch: Path,
        revision: str,
        token: str | None,
        limiter: RateLimiter,
        retries: int,
        dry_run: bool,
    ) -> None:
        self.s3 = s3
        self.prefix = prefix
        self.scratch = scratch
        self.revision = revision
        self.token = token
        self.limiter = limiter
        self.retries = retries
        self.dry_run = dry_run
        self.lock = threading.Lock()
        self.done = 0
        self.bytes = 0
        self.throttled = 0
        self.failed: list[tuple[str, str]] = []
        self._local = threading.local()

    @property
    def session(self):
        """One requests.Session per thread; Session is not thread-safe."""
        import requests

        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            if self.token:
                session.headers["Authorization"] = f"Bearer {self.token}"
            session.headers["User-Agent"] = "tesis-fetch-mom/1.0"
            self._local.session = session
        return session

    def fetch(self, repo_path: str, size: int, sha256: str, dest: Path) -> int:
        """One streaming GET against /resolve/, verified against the tree.

        No HEAD: the listing already told us the length and the hash, and a
        metadata call would double this file's cost in the resolver bucket.
        """
        import requests

        url = RESOLVE.format(
            repo=REPO_ID, revision=self.revision, path=quote(repo_path, safe="/")
        )
        delay = 2.0
        last = "no attempt made"
        for attempt in range(self.retries):
            self.limiter.acquire()
            try:
                with self.session.get(url, stream=True, timeout=(15, 180)) as response:
                    self.limiter.observe(response.headers)
                    if response.status_code == 429:
                        with self.lock:
                            self.throttled += 1
                        self.limiter.pause(
                            self.limiter.retry_after(response.headers, delay),
                            f"429 on {repo_path}",
                        )
                        last = "429 Too Many Requests"
                        continue
                    response.raise_for_status()
                    digest = hashlib.sha256()
                    written = 0
                    with open(dest, "wb") as f:
                        for chunk in response.iter_content(1 << 20):
                            f.write(chunk)
                            digest.update(chunk)
                            written += len(chunk)
                if size and written != size:
                    raise OSError(f"truncated: got {written} B, tree says {size} B")
                if sha256 and digest.hexdigest() != sha256:
                    raise OSError("sha256 mismatch against the repo tree")
                return written
            except requests.RequestException as exc:
                last = f"{type(exc).__name__}: {exc}"
            except OSError as exc:
                last = str(exc)
            dest.unlink(missing_ok=True)
            if attempt < self.retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise RuntimeError(f"{self.retries} attempts failed, last: {last}")

    def run(self, item: Item) -> Item | None:
        if self.dry_run:
            return item
        key = f"{self.prefix}/{item.key}"
        staging = self.scratch / item.generator
        staging.mkdir(parents=True, exist_ok=True)
        dest = staging / Path(item.repo_path).name
        try:
            written = self.fetch(item.repo_path, item.size, item.sha256, dest)
        except Exception as exc:  # noqa: BLE001 -- recorded and reported at the end
            with self.lock:
                self.failed.append((item.repo_path, f"download: {exc}"))
            return None
        try:
            self.s3.put_file(key, dest)
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.failed.append((item.repo_path, f"upload: {exc}"))
            return None
        finally:
            dest.unlink(missing_ok=True)
        with self.lock:
            self.done += 1
            self.bytes += written
        return item


def transfer_metadata(transfer: Transfer) -> None:
    staging = transfer.scratch / "metadata"
    for name in METADATA_FILES:
        key = f"{transfer.prefix}/metadata/{Path(name).name}"
        if transfer.dry_run:
            print(f"  would put {key}")
            continue
        staging.mkdir(parents=True, exist_ok=True)
        dest = staging / Path(name).name
        try:
            # These are not in the plan, so there is no size or hash to check.
            written = transfer.fetch(name, 0, "", dest)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {name}: {exc}")
            continue
        transfer.s3.put_file(key, dest)
        dest.unlink(missing_ok=True)
        print(f"  put {key} ({human_bytes(written)})")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bucket", required=True, help="destination S3 bucket")
    parser.add_argument("--region", default=None, help="AWS region of the bucket")
    parser.add_argument("--prefix", default="mom", help="S3 key prefix (default: mom)")
    parser.add_argument(
        "--generators",
        nargs="+",
        choices=sorted(GENERATOR_DIRS),
        default=sorted(GENERATOR_DIRS),
        help="which generators to mirror (default: all four)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="parallel download+upload workers; peak scratch is this many mp3s. "
        "The resolver quota, not this, sets the ceiling on throughput.",
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        default=None,
        help="staging directory (default: system temp). Point it at instance "
        "NVMe rather than a small root volume.",
    )
    parser.add_argument(
        "--tier",
        choices=sorted(TIERS),
        default=None,
        help="HF plan, which sets the per-5-minute request quotas: "
        + ", ".join(f"{t}={a}/{r}" for t, (a, r) in TIERS.items())
        + " (api/resolvers). Default: 'free' with a token, 'anon' without.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.9,
        help="fraction of each quota to actually spend, leaving room for other "
        "clients on the same token or IP (default: 0.9)",
    )
    parser.add_argument(
        "--api-quota",
        type=int,
        default=0,
        help="override the api-bucket quota per 5 min (0 = from --tier)",
    )
    parser.add_argument(
        "--resolver-quota",
        type=int,
        default=0,
        help="override the resolvers-bucket quota per 5 min (0 = from --tier)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=6,
        help="attempts per file before it is recorded as failed",
    )
    parser.add_argument(
        "--per-generator",
        type=int,
        default=0,
        help="0 = everything; otherwise keep this many files per generator, "
        "chosen deterministically from --seed",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=0, help="stop after N files (smoke test)"
    )
    parser.add_argument(
        "--keep-subdirs",
        action="store_true",
        help="keep udio's chunk_* directories in the S3 key instead of "
        "flattening (only needed if basenames collide)",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="mirror the CSV link tables and skip the audio",
    )
    parser.add_argument("--skip-metadata", action="store_true", help="audio only")
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("data/manifests/mom/plan.jsonl"),
        help="cached repo listing; reused when it covers --generators, so a "
        "resumed run spends nothing from the api bucket",
    )
    parser.add_argument(
        "--refresh-plan",
        action="store_true",
        help="re-list the repo even if the cached plan covers it",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/manifests/mom/files.jsonl"),
        help="local JSONL index of what is in the bucket; also uploaded to "
        "{prefix}/index/files.jsonl",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan and print the byte count, transfer nothing",
    )
    args = parser.parse_args()

    load_env_file(SCRIPTS_DIR.parent / ".env")
    token = os.environ.get("HF_TOKEN")
    tier = args.tier or ("free" if token else "anon")
    if not token:
        print(
            "WARNING: no HF_TOKEN in the environment or .env, so the Hub will "
            "meter this run anonymously, per IP address (500 api / 3,000 "
            "resolver requests per 5 min, shared with every other HF client on "
            "this address) rather than per account. This is HF's documented "
            "first cause of 429s -- set HF_TOKEN before a long transfer.",
            flush=True,
        )

    # --tier only bootstraps the budget; the first response's RateLimit-Policy
    # replaces it with this account's real quota unless it was overridden here.
    api_quota = args.api_quota or max(int(TIERS[tier][0] * args.margin), 1)
    resolver_quota = args.resolver_quota or max(int(TIERS[tier][1] * args.margin), 1)
    api_limit = RateLimiter(
        "api", api_quota, margin=args.margin, locked=bool(args.api_quota)
    )
    resolver_limit = RateLimiter(
        "resolvers", resolver_quota, margin=args.margin, locked=bool(args.resolver_quota)
    )
    print(
        f"tier={tier} token={'yes' if token else 'no'}  budget/5min: "
        f"api={api_quota} resolvers={resolver_quota} "
        f"({resolver_quota / RATE_WINDOW:.1f} files/s ceiling)"
    )

    scratch = args.scratch or Path(tempfile.mkdtemp(prefix="mom-"))
    scratch.mkdir(parents=True, exist_ok=True)

    # Planning stays importable on a machine without boto3 or credentials,
    # which is the point: this box is usually not the one doing the transfer.
    s3 = None if args.dry_run else S3(args.bucket, args.region)

    cached = None if args.refresh_plan else load_plan(args.plan, args.generators)
    if cached is not None:
        revision, items = cached
        print(f"plan: {args.plan} ({len(items)} files, revision {revision[:12]})")
    else:
        print(f"listing {REPO_ID} ({', '.join(args.generators)}) ...")
        revision = resolve_revision(api_limit, token)
        items = list_audio(args.generators, api_limit, token)
        save_plan(args.plan, revision, args.generators, items)
        print(f"  revision {revision[:12]}, plan cached at {args.plan}")

    transfer = Transfer(
        s3,
        args.prefix.rstrip("/"),
        scratch,
        revision,
        token,
        resolver_limit,
        args.retries,
        args.dry_run,
    )

    if not args.skip_metadata:
        print("metadata:")
        transfer_metadata(transfer)
    if args.metadata_only:
        return

    if not args.keep_subdirs:
        check_unique(items)
    else:
        items = [
            Item(i.repo_path, f"audio/{i.repo_path}", i.generator, i.size, i.sha256)
            for i in items
        ]
    total_bytes = sum(i.size for i in items)
    print(f"  {len(items)} files, {human_bytes(total_bytes)} in the repo")
    for generator in args.generators:
        group = [i for i in items if i.generator == generator]
        if group:
            size = sum(i.size for i in group)
            print(
                f"    {generator:<10} {len(group):>6} files  {human_bytes(size):>9}"
                f"  {human_bytes(size / len(group))}/file"
            )

    if args.per_generator:
        items = subsample(items, args.per_generator, args.seed)
        print(
            f"  subsampled to {len(items)} files "
            f"({human_bytes(sum(i.size for i in items))})"
        )

    if args.dry_run:
        # Planning must not need credentials, so the destination is not listed.
        present: set[str] = set()
        print("dry run: destination not listed, assuming empty")
    else:
        print(f"listing s3://{args.bucket}/{transfer.prefix}/audio/ ...")
        present = set(s3.list_keys(f"{transfer.prefix}/audio/"))
    pending = [i for i in items if f"{transfer.prefix}/{i.key}" not in present]
    print(f"  {len(present)} already there, {len(pending)} to transfer")
    if args.limit:
        pending = pending[: args.limit]

    pending_bytes = sum(i.size for i in pending)
    print(
        f"transferring {len(pending)} files, {human_bytes(pending_bytes)}, "
        f"{args.workers} workers, scratch={scratch}"
    )
    # resolver_limit.quota, not the bootstrap guess: by now a metadata fetch has
    # usually replaced it with the policy the server actually advertises.
    files_per_second = resolver_limit.quota / resolver_limit.window
    print(
        f"  rate-limit floor at {resolver_limit.quota} per "
        f"{resolver_limit.window:.0f}s: {len(pending) / files_per_second / 60:.0f} min"
    )
    if args.dry_run or not pending:
        return

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(transfer.run, item) for item in pending]
        for n, future in enumerate(as_completed(futures), start=1):
            future.result()  # per-file errors are collected in transfer.failed
            if n % 200 == 0 or n == len(futures):
                elapsed = time.time() - started
                rate = transfer.bytes / max(elapsed, 1e-9)
                remaining = (pending_bytes - transfer.bytes) / max(rate, 1e-9)
                print(
                    f"  {n}/{len(futures)}  {human_bytes(transfer.bytes)}  "
                    f"{human_bytes(rate)}/s  eta {remaining / 60:.0f} min  "
                    f"429s {transfer.throttled}  failed {len(transfer.failed)}",
                    flush=True,
                )

    # The index describes the bucket, so rebuild it from the bucket rather than
    # from this run alone -- runs are resumable and each one sees only its share.
    args.index.parent.mkdir(parents=True, exist_ok=True)
    by_key = {f"{transfer.prefix}/{i.key}": i for i in items}
    final = sorted(set(s3.list_keys(f"{transfer.prefix}/audio/")))
    with open(args.index, "w", encoding="utf-8") as f:
        for key in final:
            item = by_key.get(key)
            row = (
                asdict(item)
                if item is not None
                else {"repo_path": "", "key": key.removeprefix(f"{transfer.prefix}/")}
            )
            row["s3_key"] = key
            f.write(json.dumps(row) + "\n")
    s3.put_file(f"{transfer.prefix}/index/files.jsonl", args.index)

    elapsed = time.time() - started
    print(
        f"done: {transfer.done} transferred, {human_bytes(transfer.bytes)} in "
        f"{elapsed / 60:.1f} min, {len(final)} objects under "
        f"s3://{args.bucket}/{transfer.prefix}/audio/"
    )
    print(
        f"rate limiting: {transfer.throttled} 429s, {resolver_limit.pauses} pauses, "
        f"{resolver_limit.waited / max(args.workers, 1) / 60:.1f} min/worker waiting"
    )
    print(f"index: {args.index} -> {transfer.prefix}/index/files.jsonl")
    if transfer.failed:
        print(f"{len(transfer.failed)} failed (re-run to retry):")
        for path, reason in transfer.failed[:20]:
            print(f"  {path}: {reason}")


if __name__ == "__main__":
    main()
