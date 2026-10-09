#!/usr/bin/env python3
"""Mirror the MoM (Melody or Machine) synthetic-music benchmark from HF into S3.

Source: https://huggingface.co/datasets/anonymous2212/MoM-CLAM-dataset
(public, ungated, CC BY-NC 4.0; 134.25 GB of mp3 across four generators).

What the HF repo actually holds, which is less than the paper's 130 k songs:

    udio/chunk_{1,2,3}/*.mp3   84.0 GB   Udio v1.5
    riffusion/*.mp3            21.8 GB   Riffusion FUZZ-1.0
    yue/*.mp3                  21.3 GB   YuE
    diffrythm/*.mp3             7.0 GB   DiffRhythm

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

Each worker downloads one file to scratch, verifies its length against the
size HF reports, uploads it, and deletes it. Peak disk is therefore
`--workers` mp3s (tens of MB), not 134 GB -- this box is usually the one whose
NVMe already holds the phase-2 feature cache. Re-runs list the destination
prefix first and transfer only what is missing, so interrupting is free.

Audio is stored byte-for-byte as published. Do NOT normalize on the way in:
these are mp3s of mixed bitrate (and DiffRhythm's are all *exactly* 1,522,668
bytes, i.e. fixed-duration CBR) while SHS100K is YouTube opus/m4a, so one
shared decode-time normalization is the only thing that keeps the detector off
codec artifacts instead of the style-given-content signal. Keeping S3 raw
leaves that choice in code, where it is ablatable.

Examples:
    # smoke test: metadata plus 20 files per generator
    python scripts/fetch_mom.py --bucket $BUCKET --limit 20

    # the 50 GB that is ready soonest, then udio in a second pass
    python scripts/fetch_mom.py --bucket $BUCKET \\
        --generators riffusion yue diffrythm --workers 24
    python scripts/fetch_mom.py --bucket $BUCKET --generators udio --workers 24

    # size-matched subset instead of the full mirror (deterministic)
    python scripts/fetch_mom.py --bucket $BUCKET --per-generator 3000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from src.s3 import S3  # noqa: E402

REPO_ID = "anonymous2212/MoM-CLAM-dataset"

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


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def list_audio(generators: list[str], token: str | None) -> list[Item]:
    """Enumerate the mp3s of the requested generators, with sizes and hashes."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    items: list[Item] = []
    for generator in generators:
        directory = GENERATOR_DIRS[generator]
        for entry in api.list_repo_tree(
            REPO_ID, repo_type="dataset", path_in_repo=directory, recursive=True
        ):
            path = entry.path
            if not path.endswith(".mp3"):
                continue
            # entry.size is None for non-LFS files; every mp3 here is LFS.
            size = entry.size or 0
            sha = entry.lfs.sha256 if getattr(entry, "lfs", None) else ""
            items.append(
                Item(
                    repo_path=path,
                    key=f"audio/{generator}/{Path(path).name}",
                    generator=generator,
                    size=size,
                    sha256=sha,
                )
            )
    return items


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
        token: str | None,
        dry_run: bool,
    ) -> None:
        self.s3 = s3
        self.prefix = prefix
        self.scratch = scratch
        self.token = token
        self.dry_run = dry_run
        self.lock = threading.Lock()
        self.done = 0
        self.bytes = 0
        self.failed: list[tuple[str, str]] = []

    def run(self, item: Item) -> Item | None:
        from huggingface_hub import hf_hub_download

        key = f"{self.prefix}/{item.key}"
        if self.dry_run:
            return item

        # local_dir keeps the blob out of the shared HF cache, so deleting the
        # file after upload actually reclaims the space.
        staging = self.scratch / item.generator
        try:
            path = Path(
                hf_hub_download(
                    REPO_ID,
                    filename=item.repo_path,
                    repo_type="dataset",
                    local_dir=str(staging),
                    token=self.token,
                )
            )
        except Exception as exc:  # noqa: BLE001 -- recorded and reported at the end
            with self.lock:
                self.failed.append((item.repo_path, f"download: {exc}"))
            return None

        try:
            actual = path.stat().st_size
            if item.size and actual != item.size:
                raise RuntimeError(f"size {actual} != HF's {item.size}")
            self.s3.put_file(key, path)
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.failed.append((item.repo_path, f"upload: {exc}"))
            return None
        finally:
            path.unlink(missing_ok=True)

        with self.lock:
            self.done += 1
            self.bytes += actual
        return item


def transfer_metadata(transfer: Transfer, token: str | None) -> None:
    from huggingface_hub import hf_hub_download

    for name in METADATA_FILES:
        key = f"{transfer.prefix}/metadata/{Path(name).name}"
        if transfer.dry_run:
            print(f"  would put {key}")
            continue
        try:
            path = Path(
                hf_hub_download(
                    REPO_ID,
                    filename=name,
                    repo_type="dataset",
                    local_dir=str(transfer.scratch / "metadata"),
                    token=token,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {name}: {exc}")
            continue
        transfer.s3.put_file(key, path)
        print(f"  put {key} ({human_bytes(path.stat().st_size)})")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bucket", required=True, help="destination S3 bucket")
    parser.add_argument("--region", default=None, help="AWS region of the bucket")
    parser.add_argument(
        "--prefix", default="mom", help="S3 key prefix (default: mom)"
    )
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
        help="parallel download+upload workers; peak scratch is this many mp3s",
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        default=None,
        help="staging directory (default: system temp). Point it at instance "
        "NVMe rather than a small root volume.",
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
    parser.add_argument(
        "--skip-metadata", action="store_true", help="audio only"
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

    token = os.environ.get("HF_TOKEN")  # repo is public; a token only lifts limits
    scratch = args.scratch or Path(tempfile.mkdtemp(prefix="mom-"))
    scratch.mkdir(parents=True, exist_ok=True)

    # Planning stays importable on a machine without boto3 or credentials,
    # which is the point: this box is usually not the one doing the transfer.
    s3 = None if args.dry_run else S3(args.bucket, args.region)
    transfer = Transfer(s3, args.prefix.rstrip("/"), scratch, token, args.dry_run)

    if not args.skip_metadata:
        print("metadata:")
        transfer_metadata(transfer, token)

    if args.metadata_only:
        return

    print(f"listing {REPO_ID} ({', '.join(args.generators)}) ...")
    items = list_audio(args.generators, token)
    if not args.keep_subdirs:
        check_unique(items)
    else:
        items = [
            Item(
                i.repo_path,
                f"audio/{i.repo_path}",
                i.generator,
                i.size,
                i.sha256,
            )
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
    if args.dry_run:
        return
    if not pending:
        return

    started = time.time()
    landed: list[Item] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(transfer.run, item) for item in pending]
        for n, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result is not None:
                landed.append(result)
            if n % 200 == 0 or n == len(futures):
                elapsed = time.time() - started
                rate = transfer.bytes / max(elapsed, 1e-9)
                remaining = (pending_bytes - transfer.bytes) / max(rate, 1e-9)
                print(
                    f"  {n}/{len(futures)}  {human_bytes(transfer.bytes)}  "
                    f"{human_bytes(rate)}/s  eta {remaining / 60:.0f} min  "
                    f"failed {len(transfer.failed)}",
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
    print(f"index: {args.index} -> {transfer.prefix}/index/files.jsonl")
    if transfer.failed:
        print(f"{len(transfer.failed)} failed (re-run to retry):")
        for path, reason in transfer.failed[:20]:
            print(f"  {path}: {reason}")


if __name__ == "__main__":
    main()
