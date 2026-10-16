"""On-disk cache of time-pooled MERT hidden states, one row per window.

Layout of a cache directory (written by scripts/cache_pooled_mert.py):

    meta.json              what was cached: split, S, windows per track, model
    shard_00000.npy        (n, 25, 1024) fp16, pooled hidden states
    shard_00000.jsonl      one {"key", "start"} line per row of the .npy
    layer_stats.pt         training-set moments (written by train_mert_flow.py)

Each row is one exact S-second window through `decode_window` and a full
MERT forward (CLAUDE.md: MERT features are not local, so windows are cached,
never sliced out of a track-level cache). A shard counts only once its .jsonl
exists; both files are written via a temporary name and renamed, so an
interrupted extraction never leaves a half shard behind.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np


def shard_paths(root: Path, index: int) -> tuple[Path, Path]:
    stem = root / f"shard_{index:05d}"
    return stem.with_suffix(".npy"), stem.with_suffix(".jsonl")


def write_shard(root: Path, index: int, array: np.ndarray, rows: list[dict]) -> None:
    npy, jsonl = shard_paths(root, index)
    tmp = npy.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, array.astype(np.float16))
    os.replace(tmp, npy)
    tmp = jsonl.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, jsonl)


def shard_done(root: Path, index: int) -> bool:
    npy, jsonl = shard_paths(root, index)
    return npy.exists() and jsonl.exists()


class PooledCache:
    """Read-only view over a cache directory's complete shards (memory-mapped)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        meta = self.root / "meta.json"
        if not meta.exists():
            raise SystemExit(f"{self.root}: no meta.json, not a pooled-MERT cache")
        self.meta = json.loads(meta.read_text(encoding="utf-8"))
        npys = sorted(self.root.glob("shard_*.npy"))
        self.arrays = [
            np.load(p, mmap_mode="r") for p in npys if p.with_suffix(".jsonl").exists()
        ]
        if not self.arrays:
            raise SystemExit(f"{self.root}: no complete shards")
        self.offsets = np.cumsum([0] + [len(a) for a in self.arrays])
        self.n_layers, self.dim = self.arrays[0].shape[1:]

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def gather(self, indices: np.ndarray) -> np.ndarray:
        """Rows at global `indices` (any order) -> (n, L, D) float32, same order."""
        indices = np.asarray(indices)
        out = np.empty((len(indices), self.n_layers, self.dim), dtype=np.float32)
        shard = np.searchsorted(self.offsets, indices, side="right") - 1
        for s in np.unique(shard):
            mask = shard == s
            local = indices[mask] - self.offsets[s]
            order = np.argsort(local)  # sorted reads are sequential on disk
            rows = np.asarray(self.arrays[s][local[order]], dtype=np.float32)
            out[np.flatnonzero(mask)[order]] = rows
        return out

    def batches(
        self, batch_size: int, rng: np.random.Generator | None, drop_last: bool
    ) -> Iterator[np.ndarray]:
        """Batches over the whole cache; shuffled globally when `rng` is given."""
        order = rng.permutation(len(self)) if rng is not None else np.arange(len(self))
        stop = len(order) - (len(order) % batch_size if drop_last else 0)
        for start in range(0, stop, batch_size):
            yield self.gather(order[start : start + batch_size])

    def shards(self, max_rows: int = 0) -> Iterator[np.ndarray]:
        """Shard by shard as float32, for statistics; stops after max_rows (0 = all)."""
        seen = 0
        for array in self.arrays:
            if max_rows and seen >= max_rows:
                return
            take = len(array) if not max_rows else min(len(array), max_rows - seen)
            seen += take
            yield np.asarray(array[:take], dtype=np.float32)
