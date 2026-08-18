#!/usr/bin/env python3
"""Offline cover-pair alignment for SHS100K (see CLAUDE.md, "Cover Alignment").

Two resumable stages:

  chroma  For every downloaded track: decode audio (ffmpeg), compute
          beat-synchronous chroma (chroma_cqt on the harmonic component,
          median-aggregated between beats, L2-normalized) and cache it to
          data/chroma/{split}/{key}.npz  (chroma (12,T), times (T,)).

  align   For every pair of covers within a song (clique): pick the best of
          the 12 chroma transpositions (OTI), run Smith-Waterman local
          alignment on the binarized similarity matrix, and store the warping
          path as aligned time arrays plus a normalized confidence score in
          data/alignments/{split}/{song_id}_{verA}_{verB}.npz
          (t_a (L,), t_b (L,), score (scalar), oti (scalar)).

The normalized score is the quality-filtering confidence: training code should
drop pairs below a threshold (~0.2, the cross-song negative p95 measured
on 50 val tracks; see validate_alignment.py).

Examples:
    python scripts/align_covers.py --stage all --split val
    python scripts/align_covers.py --stage align --split train --workers 8
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from shs100k_meta import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    SPLITS,
    alignment_dir,
    chroma_dir,
    existing_audio,
)

CHROMA_SR = 22050
HOP_LENGTH = 512
MIN_BEATS = 20  # tracks with fewer beat segments than this are useless for alignment


def _numba_private_cache() -> None:
    """Pool-worker initializer: use a private numba cache directory.

    Concurrent workers compiling the same functions (librosa's beat tracker,
    our Smith-Waterman kernel) corrupt the shared on-disk numba cache, which
    later segfaults child processes. A per-worker cache dir avoids the race.
    """
    os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="numba_cache_")
    try:
        from numba.core import config

        config.reload_config()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Stage 1: beat-synchronous chroma
# --------------------------------------------------------------------------- #


def load_audio_ffmpeg(path: Path, sr: int, max_seconds: float | None = None) -> np.ndarray:
    """Decode any audio container to mono float32 at the given sample rate."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += [
        "-f", "f32le", "-ac", "1", "-ar", str(sr),
        "pipe:1",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def beat_sync_chroma(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (chroma (12,T), segment mid-times (T,)) synchronized to beats.

    Falls back to fixed ~0.5 s segments when beat tracking finds nothing.
    """
    import librosa

    y_harm = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=HOP_LENGTH)
    n_frames = chroma.shape[1]

    _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    beats = beats[(beats > 0) & (beats < n_frames)]
    if len(beats) < 4:
        step = int(0.5 * sr / HOP_LENGTH)
        beats = np.arange(step, n_frames, step)

    bounds = np.concatenate(([0], beats, [n_frames]))
    synced = librosa.util.sync(chroma, beats, aggregate=np.median)
    mids = (bounds[:-1] + bounds[1:]) / 2.0
    times = librosa.frames_to_time(mids, sr=sr, hop_length=HOP_LENGTH)

    norms = np.linalg.norm(synced, axis=0, keepdims=True)
    synced = synced / np.maximum(norms, 1e-8)
    return synced.astype(np.float32), times.astype(np.float32)


def chroma_worker(
    audio_path: Path, out_path: Path, max_seconds: float
) -> tuple[str, str]:
    try:
        y = load_audio_ffmpeg(audio_path, CHROMA_SR, max_seconds)
        if len(y) < CHROMA_SR * 5:
            return audio_path.stem, "too short"
        chroma, times = beat_sync_chroma(y, CHROMA_SR)
        if chroma.shape[1] < MIN_BEATS:
            return audio_path.stem, "too few beats"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, chroma=chroma, times=times)
        return audio_path.stem, ""
    except Exception as exc:
        return audio_path.stem, f"{type(exc).__name__}: {exc}"


def run_chroma_stage(
    data_root: Path, split: str, workers: int, max_seconds: float
) -> None:
    audio = existing_audio(data_root, split)
    out_dir = chroma_dir(data_root, split)
    todo = {
        key: path
        for key, path in audio.items()
        if not (out_dir / f"{key}.npz").exists()
    }
    print(f"[chroma/{split}] {len(audio)} audio files, {len(todo)} to process")
    if not todo:
        return

    failures = 0
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_numba_private_cache
    ) as pool:
        futures = [
            pool.submit(chroma_worker, path, out_dir / f"{key}.npz", max_seconds)
            for key, path in todo.items()
        ]
        for i, future in enumerate(as_completed(futures), 1):
            key, error = future.result()
            if error:
                failures += 1
                print(f"[chroma/{split}] {key}: {error}")
            if i % 100 == 0 or i == len(futures):
                print(f"[chroma/{split}] {i}/{len(futures)} ({failures} failed)")


# --------------------------------------------------------------------------- #
# Stage 2: OTI + Smith-Waterman alignment
# --------------------------------------------------------------------------- #


def optimal_transposition_index(chroma_a: np.ndarray, chroma_b: np.ndarray) -> int:
    """Circular shift of B's chroma that best matches A (Serrà's OTI)."""
    global_a = chroma_a.mean(axis=1)
    global_b = chroma_b.mean(axis=1)
    scores = [float(global_a @ np.roll(global_b, k)) for k in range(12)]
    return int(np.argmax(scores))


def _sw_matrix(scores: np.ndarray, gap: float) -> np.ndarray:
    """Smith-Waterman DP table (pure Python; numba-jitted when available)."""
    n_a, n_b = scores.shape
    table = np.zeros((n_a + 1, n_b + 1), dtype=np.float32)
    for i in range(1, n_a + 1):
        for j in range(1, n_b + 1):
            best = table[i - 1, j - 1] + scores[i - 1, j - 1]
            up = table[i - 1, j] - gap
            left = table[i, j - 1] - gap
            if up > best:
                best = up
            if left > best:
                best = left
            if best < 0.0:
                best = 0.0
            table[i, j] = best
    return table


try:
    from numba import njit

    _sw_matrix = njit(cache=True)(_sw_matrix)  # type: ignore[assignment]
except ImportError:
    print("note: numba not installed; Smith-Waterman will be slow (pip install numba)")


def smith_waterman_path(
    sim: np.ndarray, match_quantile: float, gap: float
) -> tuple[np.ndarray, float]:
    """Local alignment on a binarized similarity matrix.

    Returns (path (L,2) of beat-index pairs, score normalized by the shorter
    sequence length).
    """
    threshold = np.quantile(sim, match_quantile)
    scores = np.where(sim >= threshold, 1.0, -1.0).astype(np.float32)
    table = _sw_matrix(scores, gap)

    i, j = np.unravel_index(int(np.argmax(table)), table.shape)
    best = float(table[i, j])
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0 and table[i, j] > 0.0:
        path.append((i - 1, j - 1))
        diag = table[i - 1, j - 1] + scores[i - 1, j - 1]
        up = table[i - 1, j] - gap
        if np.isclose(table[i, j], diag):
            i, j = i - 1, j - 1
        elif np.isclose(table[i, j], up):
            i = i - 1
        else:
            j = j - 1
    path.reverse()
    norm = best / max(min(sim.shape), 1)
    return np.asarray(path, dtype=np.int32), norm


def align_worker(
    chroma_path_a: Path,
    chroma_path_b: Path,
    out_path: Path,
    match_quantile: float,
    gap: float,
) -> tuple[str, str]:
    try:
        data_a = np.load(chroma_path_a)
        data_b = np.load(chroma_path_b)
        chroma_a, times_a = data_a["chroma"], data_a["times"]
        chroma_b, times_b = data_b["chroma"], data_b["times"]

        oti = optimal_transposition_index(chroma_a, chroma_b)
        chroma_b_t = np.roll(chroma_b, oti, axis=0)
        sim = chroma_a.T @ chroma_b_t

        path, score = smith_waterman_path(sim, match_quantile, gap)
        if len(path) == 0:
            return out_path.stem, "empty alignment"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            t_a=times_a[path[:, 0]],
            t_b=times_b[path[:, 1]],
            score=np.float32(score),
            oti=np.int32(oti),
        )
        return out_path.stem, ""
    except Exception as exc:
        return out_path.stem, f"{type(exc).__name__}: {exc}"


def run_align_stage(
    data_root: Path, split: str, workers: int, match_quantile: float, gap: float
) -> None:
    in_dir = chroma_dir(data_root, split)
    out_dir = alignment_dir(data_root, split)
    if not in_dir.is_dir():
        print(f"[align/{split}] no chroma directory {in_dir}; run --stage chroma first")
        return

    by_song: dict[int, list[str]] = defaultdict(list)
    for path in sorted(in_dir.glob("*.npz")):
        song_id = int(path.stem.split("_")[0])
        by_song[song_id].append(path.stem)

    jobs: list[tuple[Path, Path, Path]] = []
    for song_id, keys in by_song.items():
        for key_a, key_b in itertools.combinations(sorted(keys), 2):
            out_path = out_dir / f"{song_id}_{key_a.split('_')[1]}_{key_b.split('_')[1]}.npz"
            if not out_path.exists():
                jobs.append((in_dir / f"{key_a}.npz", in_dir / f"{key_b}.npz", out_path))

    n_pairs = sum(len(k) * (len(k) - 1) // 2 for k in by_song.values())
    print(f"[align/{split}] {n_pairs} cover pairs total, {len(jobs)} to align")
    if not jobs:
        return

    failures = 0
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_numba_private_cache
    ) as pool:
        futures = [
            pool.submit(align_worker, a, b, out, match_quantile, gap)
            for a, b, out in jobs
        ]
        for i, future in enumerate(as_completed(futures), 1):
            stem, error = future.result()
            if error:
                failures += 1
                print(f"[align/{split}] {stem}: {error}")
            if i % 200 == 0 or i == len(futures):
                print(f"[align/{split}] {i}/{len(futures)} ({failures} failed)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("chroma", "align", "all"), default="all")
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--match-quantile",
        type=float,
        default=0.8,
        help="similarity quantile above which a cell counts as a match",
    )
    parser.add_argument(
        "--gap", type=float, default=0.5, help="Smith-Waterman gap penalty"
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=600.0,
        help="analyze at most this many seconds per track (caps HPSS/CQT memory; "
        "the chroma stage is memory-hungry, keep --workers low)",
    )
    args = parser.parse_args()

    splits = list(SPLITS) if args.split == "all" else [args.split]
    for split in splits:
        if args.stage in ("chroma", "all"):
            run_chroma_stage(args.data_root, split, args.workers, args.max_seconds)
        if args.stage in ("align", "all"):
            run_align_stage(
                args.data_root, split, args.workers, args.match_quantile, args.gap
            )


if __name__ == "__main__":
    main()
