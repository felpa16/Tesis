#!/usr/bin/env python3
"""Cache time-pooled MERT hidden states of MoM songs, for train_mert_probe.py.

The supervised probe (src/models/mert_probe.py) is the only thing in the
project that trains on synthetic music, so it is the only thing that needs
this cache. Windows and pooling are those of every other cache: K windows of S
seconds, each centred in one of K equal slices of the song
(evaluate_mom.window_starts), decode_window, one MERT forward, and all 25
hidden states mean-pooled over time (pool_layers). Rows are (25, 1024) fp16 in
the PooledCache layout (src/data/pooled_cache.py), resumable by shard.

Which songs. Each generator's songs are put in evaluate_mom.py's deterministic
order (deterministic_sample, same --seed) and the first --skip are left out.
Those are the songs evaluate_mom.py, the --track-metrics cache and
evaluate_mert_probe.py score at --per-source up to --skip. The next
--per-generator songs are cached. So no song is both trained on and evaluated,
as long as evaluation keeps --per-source <= --skip and the same seed;
evaluate_mert_probe.py also drops any overlap it finds.

Every generator is cached, the one a probe will leave out included, so one
cache serves every --leave-out. Generators are interleaved song by song, so a
partial cache is balanced across them.

Rows: {"key": "{generator}/{song}", "source": generator, "song": song, "start": s}

Examples:
    python scripts/cache_pooled_mom.py --out-dir $CACHE/mom --bucket $BUCKET
    # local smoke run
    python scripts/cache_pooled_mom.py --out-dir /tmp/pm --mom-root ~/mom \\
        --per-generator 2 --windows-per-song 1 --window-seconds 5 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

import evaluate_mom as em  # noqa: E402
from track_mom import song_loader  # noqa: E402
from src.config import MertConfig  # noqa: E402
from src.data.pooled_cache import shard_done, write_shard  # noqa: E402
from src.data.windows import WindowConfig  # noqa: E402
from src.models import MertExtractor  # noqa: E402
from src.models.mert_flow import pool_layers  # noqa: E402
from src.training import pick_device  # noqa: E402

# Settings a cache is extended under only if they match.
IDENTITY = ("generators", "per_generator", "skip", "seed", "window_seconds",
            "windows_per_song", "model_name", "songs_per_shard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--generators", nargs="+", default=list(em.GENERATORS))
    parser.add_argument("--per-generator", type=int, default=2000,
                        help="songs cached per generator (fewer if the generator has fewer)")
    parser.add_argument("--skip", type=int, default=300,
                        help="songs at the head of evaluate_mom.py's order, left for evaluation")
    parser.add_argument("--seed", type=int, default=0, help="must match evaluate_mom.py's --seed")
    parser.add_argument("--window-seconds", type=float, default=20.0)
    parser.add_argument("--windows-per-song", type=int, default=4)
    parser.add_argument("--songs-per-shard", type=int, default=1024)
    parser.add_argument("--bucket", default="songs-and-mert-features")
    parser.add_argument("--region", default=None)
    parser.add_argument("--mom-prefix", default="mom")
    parser.add_argument("--mom-root", type=Path, default=None,
                        help="local MoM mirror instead of S3")
    parser.add_argument("--tmp-dir", type=Path, default=None,
                        help="where S3 audio is staged (default: system temp)")
    parser.add_argument("--batch", type=int, default=16, help="windows per MERT call")
    parser.add_argument("--mert-micro-batch", type=int, default=4)
    parser.add_argument("--fetch-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mert_config = MertConfig(micro_batch=args.mert_micro_batch)
    window = WindowConfig(window_seconds=args.window_seconds, sample_rate=mert_config.sample_rate)

    s3 = None
    if args.mom_root is None:
        from src.s3 import S3

        s3 = S3(args.bucket, args.region)
    groups = []
    for generator in args.generators:
        if args.mom_root is not None:
            _, songs = em.mom_songs_local(args.mom_root, generator)
        else:
            _, songs = em.mom_songs_s3(s3, args.mom_prefix, generator)
        ordered = em.deterministic_sample(songs, 0, args.seed)
        picked = ordered[args.skip : args.skip + args.per_generator]
        print(f"{generator}: {len(songs)} songs, caching {len(picked)} "
              f"(after the first {args.skip}, kept for evaluation)")
        groups.append(picked)
    todo_songs = em.interleave(groups)
    n_shards = (len(todo_songs) + args.songs_per_shard - 1) // args.songs_per_shard

    meta = {
        "kind": "mom",
        "generators": list(args.generators),
        "per_generator": args.per_generator,
        "skip": args.skip,
        "seed": args.seed,
        "window_seconds": args.window_seconds,
        "windows_per_song": args.windows_per_song,
        "model_name": mert_config.model_name,
        "songs_per_shard": args.songs_per_shard,
        "sample_rate": mert_config.sample_rate,
        "n_hidden_states": mert_config.n_hidden_states,
        "dim": mert_config.dim,
        "pooling": "time mean, fp32 accumulate, stored fp16",
        "n_songs_planned": len(todo_songs),
        "songs_planned": {g: len(p) for g, p in zip(args.generators, groups)},
        "bucket": None if args.mom_root else args.bucket,
        "region": args.region,
        "mom_prefix": None if args.mom_root else args.mom_prefix,
        "mom_root": str(args.mom_root) if args.mom_root else None,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "meta.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        changed = [k for k in IDENTITY if previous.get(k) != meta[k]]
        if changed:
            raise SystemExit(f"{args.out_dir} was cached with different {', '.join(changed)}; "
                             "use another --out-dir")
    meta_path.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")

    todo = [i for i in range(n_shards) if not shard_done(args.out_dir, i)]
    print(f"{len(todo_songs)} songs in {n_shards} shards, {len(todo)} to do -> {args.out_dir}")
    if not todo:
        return

    device = pick_device(args.device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    mert = MertExtractor(mert_config).to(device)
    opener = em.AudioOpener(Path("."), s3, None, args.mom_root is not None, args.tmp_dir)
    load = song_loader(opener, window, args.windows_per_song)
    started, n_windows, failures = time.time(), 0, []

    for shard in todo:
        songs = todo_songs[shard * args.songs_per_shard : (shard + 1) * args.songs_per_shard]
        pooled: list[np.ndarray] = []
        rows: list[dict] = []
        pending: list[tuple[dict, torch.Tensor]] = []

        def flush(items: list[tuple[dict, torch.Tensor]]) -> None:
            out = pool_layers(mert, torch.stack([w for _, w in items]).to(device),
                              mert_config.micro_batch, autocast)
            pooled.append(out.half().cpu().numpy())
            rows.extend(row for row, _ in items)

        for loaded in em.prefetch(load, songs, args.fetch_workers):
            if loaded.error is not None:
                failures.append(f"{loaded.song.source}/{loaded.song.song_id}: {loaded.error}")
                if len(failures) <= 5:
                    print(f"  failed: {failures[-1]}", flush=True)
                continue
            song = loaded.song
            for start, wave in zip(loaded.starts, loaded.waves):
                pending.append(({"key": f"{song.source}/{song.song_id}", "source": song.source,
                                 "song": song.song_id, "start": start}, wave))
            while len(pending) >= args.batch:
                flush(pending[: args.batch])
                pending = pending[args.batch :]
        if pending:
            flush(pending)
        if not rows:
            raise SystemExit(f"shard {shard + 1}: every song failed; see the errors above "
                             "(check --bucket / --mom-prefix and that ffmpeg is installed)")
        write_shard(args.out_dir, shard, np.concatenate(pooled), rows)
        n_windows += len(rows)
        rate = n_windows / max(time.time() - started, 1e-9)
        songs_left = sum(min(args.songs_per_shard, len(todo_songs) - s * args.songs_per_shard)
                         for s in todo if s > shard)
        eta = songs_left * args.windows_per_song / max(rate, 1e-9) / 60
        print(f"shard {shard + 1}/{n_shards}: {len(rows)} windows  {rate:.1f} windows/s  "
              f"eta {eta:.0f} min  failed songs {len(failures)}", flush=True)
    print(f"done: {n_windows} windows cached, {len(failures)} songs failed")


if __name__ == "__main__":
    main()
