#!/usr/bin/env python3
"""Cache time-pooled MERT hidden states for the MERT-only flow baseline.

For every track of a split, K windows of S seconds (K equal slices, each window
centred in its slice: the same placement evaluate_mom.py scores) are decoded,
run through frozen MERT-v1-330M, and all 25 hidden states are mean-pooled over
time. Rows are (25, 1024) fp16, ~51 KB per window: 4 windows of ~50k train
tracks come to ~10 GB.

All 25 layers are kept, unmixed, because pooling commutes with the layer mix:
the frozen and the unfrozen arm of scripts/train_mert_flow.py both train from
this one cache.

Track order is shuffled with --seed before sharding, so a partial cache (or
--max-tracks) is a random sample rather than the first works of the manifest.
Resumable: finished shards are skipped, and the run refuses to extend a cache
built with different settings.

Examples:
    python scripts/cache_pooled_mert.py --split train --out-dir cache/pooled/train
    python scripts/cache_pooled_mert.py --split val50 --out-dir cache/pooled/val50
    # local smoke run
    python scripts/cache_pooled_mert.py --split val --out-dir /tmp/pc --max-tracks 2 \\
        --windows-per-track 1 --window-seconds 5 --device cpu --num-workers 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from evaluate_mom import window_starts  # noqa: E402
from shs100k_meta import DEFAULT_DATA_ROOT  # noqa: E402
from src.config import MertConfig  # noqa: E402
from src.data.manifest import read_tracks  # noqa: E402
from src.data.pooled_cache import shard_done, write_shard  # noqa: E402
from src.data.windows import WindowConfig, decode_window  # noqa: E402
from src.models.mert import MertExtractor  # noqa: E402
from src.models.mert_flow import pool_layers  # noqa: E402
from src.training import pick_device  # noqa: E402

# Settings a cache is extended under only if they match.
IDENTITY = ("split", "window_seconds", "windows_per_track", "max_tracks", "seed",
            "model_name", "shard_size")


class WindowList(Dataset):
    """(audio path, start) -> decoded window; failures come back flagged."""

    def __init__(self, items: list[dict], data_root: Path, config: WindowConfig) -> None:
        self.items = items
        self.data_root = data_root
        self.config = config

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        try:
            wave = decode_window(self.data_root / item["audio"], item["start"], self.config)
            ok = True
        except Exception:  # noqa: BLE001 - one unreadable file must not stop a 4-hour run
            wave = torch.zeros(self.config.window_samples)
            ok = False
        return {"wave": wave, "index": index, "ok": ok}


def plan_windows(tracks, window_seconds: float, per_track: int, max_tracks: int, seed: int) -> list[dict]:
    tracks = sorted((t for t in tracks if t.duration >= window_seconds), key=lambda t: t.key)
    random.Random(seed).shuffle(tracks)
    if max_tracks:
        tracks = tracks[:max_tracks]
    return [
        {"key": t.key, "audio": t.audio, "start": start}
        for t in tracks
        for start in window_starts(t.duration, window_seconds, per_track)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", required=True, help="manifest split, e.g. train, val50")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--window-seconds", type=float, default=20.0)
    parser.add_argument("--windows-per-track", type=int, default=4)
    parser.add_argument("--max-tracks", type=int, default=0, help="0 = every track")
    parser.add_argument("--seed", type=int, default=0, help="track-order shuffle")
    parser.add_argument("--shard-size", type=int, default=4096, help="windows per shard")
    parser.add_argument("--batch", type=int, default=8, help="windows per MERT call")
    parser.add_argument("--mert-micro-batch", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4, help="ffmpeg decode workers")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mert_config = MertConfig(micro_batch=args.mert_micro_batch)
    window = WindowConfig(window_seconds=args.window_seconds, sample_rate=mert_config.sample_rate)

    items = plan_windows(
        read_tracks(args.data_root, args.split), args.window_seconds,
        args.windows_per_track, args.max_tracks, args.seed,
    )
    n_shards = (len(items) + args.shard_size - 1) // args.shard_size
    meta = {
        "split": args.split,
        "window_seconds": args.window_seconds,
        "windows_per_track": args.windows_per_track,
        "max_tracks": args.max_tracks,
        "seed": args.seed,
        "model_name": mert_config.model_name,
        "shard_size": args.shard_size,
        "sample_rate": mert_config.sample_rate,
        "n_hidden_states": mert_config.n_hidden_states,
        "dim": mert_config.dim,
        "pooling": "time mean, fp32 accumulate, stored fp16",
        "n_windows_planned": len(items),
        "n_tracks": len({i["key"] for i in items}),
        "data_root": str(args.data_root),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "meta.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        changed = [k for k in IDENTITY if previous.get(k) != meta[k]]
        if changed:
            raise SystemExit(
                f"{args.out_dir} was cached with different {', '.join(changed)}; "
                "use another --out-dir"
            )
    meta_path.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")

    todo = [i for i in range(n_shards) if not shard_done(args.out_dir, i)]
    print(
        f"{args.split}: {meta['n_tracks']} tracks, {len(items)} windows of "
        f"{args.window_seconds:g} s, {n_shards} shards, {len(todo)} to do -> {args.out_dir}"
    )
    if not todo:
        return

    device = pick_device(args.device)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    )
    mert = MertExtractor(mert_config).to(device)
    started, n_done, n_failed = time.time(), 0, 0
    for shard in todo:
        chunk = items[shard * args.shard_size : (shard + 1) * args.shard_size]
        loader = DataLoader(
            WindowList(chunk, args.data_root, window),
            batch_size=args.batch, num_workers=args.num_workers,
        )
        pooled, rows = [], []
        for batch in loader:
            ok = batch["ok"]
            if not ok.any():
                n_failed += len(ok)
                continue
            out = pool_layers(mert, batch["wave"][ok].to(device), mert_config.micro_batch, autocast)
            pooled.append(out.cpu().numpy())
            rows += [
                {"key": chunk[i]["key"], "start": chunk[i]["start"]}
                for i in batch["index"][ok].tolist()
            ]
            n_failed += int((~ok).sum())
            n_done += int(ok.sum())
        if rows:
            write_shard(args.out_dir, shard, np.concatenate(pooled), rows)
        rate = n_done / max(time.time() - started, 1e-9)
        left = sum(
            min(args.shard_size, len(items) - s * args.shard_size) for s in todo if s > shard
        )
        print(
            f"shard {shard + 1}/{n_shards}: {len(rows)} windows  {rate:.1f} windows/s  "
            f"eta {left / max(rate, 1e-9) / 60:.0f} min  failed {n_failed}",
            flush=True,
        )
    print(f"done: {n_done} windows cached, {n_failed} failed to decode")


if __name__ == "__main__":
    main()
