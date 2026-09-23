#!/usr/bin/env python3
"""Offline cover-pair alignment for SHS100K (see CLAUDE.md, "Cover Alignment").

Two resumable stages:

  chroma  For every downloaded track: decode audio (ffmpeg), compute
          beat-synchronous chroma (chroma_cqt on the harmonic component,
          median-aggregated between beats, L2-normalized) and cache it to
          data/chroma/{split}/{key}.npz  (chroma (12,T), times (T,)).

  align   For every pair of covers within a song (clique, capped by
          --max-pairs-per-song): pick the best of the 12 chroma transpositions
          (OTI), run Smith-Waterman local alignment on the binarized similarity
          matrix, and store the warping path as aligned time arrays plus a
          normalized confidence score in
          data/alignments/{split}/{song_id}_{verA}_{verB}.npz
          (t_a (L,), t_b (L,), score (scalar), oti (scalar)).

The normalized score is the quality-filtering confidence: training code should
drop pairs below a threshold (~0.2, the cross-song negative p95 measured
on 50 val tracks; see validate_alignment.py).

With --kept-per-song K the align stage instead aims for a fixed number of
*usable* pairs per work: it walks each clique's candidates in round-robin
order (see round_robin_pairs), aligning in parallel rounds, until K of them
score >= --min-score or --max-pairs-per-song candidates have been tried. The
chosen pairs are written to data/alignments/{split}/selection_k{K}.json, which
`build_manifest.py --kept-per-song K` reads. Alignments already on disk are
reused, so the stage is resumable and never recomputes a pair.

Examples:
    python scripts/align_covers.py --stage all --split val
    python scripts/align_covers.py --stage align --split train --workers 8
    python scripts/align_covers.py --stage align --split train --workers 32 \\
        --kept-per-song 20 --min-score 0.2
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
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
    selection_path,
    split_tracks,
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
        partial = out_path.with_name(out_path.name + ".part")  # see align_worker
        with open(partial, "wb") as f:
            np.savez_compressed(f, chroma=chroma, times=times)
        os.replace(partial, out_path)
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
) -> tuple[str, str, float]:
    """Align one pair. Returns (stem, error or "", score or nan)."""
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
            return out_path.stem, "empty alignment", math.nan
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a worker killed mid-write (spot reclaim, Ctrl-C)
        # must not leave a truncated .npz that resume logic trusts as done.
        partial = out_path.with_name(out_path.name + ".part")
        with open(partial, "wb") as f:
            np.savez_compressed(
                f,
                t_a=times_a[path[:, 0]],
                t_b=times_b[path[:, 1]],
                score=np.float32(score),
                oti=np.int32(oti),
            )
        os.replace(partial, out_path)
        return out_path.stem, "", float(np.float32(score))
    except Exception as exc:
        return out_path.stem, f"{type(exc).__name__}: {exc}", math.nan


def clique_pairs(keys: list[str], max_pairs: int) -> list[tuple[str, str]]:
    """Cover pairs within one clique, capped at max_pairs (0 = every pair).

    Cliques in the 2025 dataset reach 1,991 versions, so taking every
    combination would ask for ~9M alignments on the train split alone. When the
    cap bites, pairs are sampled with an RNG seeded by the song id, so a re-run
    selects the same subset and the stage stays resumable.
    """
    keys = sorted(keys)
    total = len(keys) * (len(keys) - 1) // 2
    if max_pairs <= 0 or total <= max_pairs:
        return list(itertools.combinations(keys, 2))

    song_id = int(keys[0].split("_")[0])
    rng = random.Random(song_id)
    chosen: set[tuple[int, int]] = set()
    while len(chosen) < max_pairs:  # max_pairs << total here, so this converges
        i, j = rng.randrange(len(keys)), rng.randrange(len(keys))
        if i != j:
            chosen.add((i, j) if i < j else (j, i))
    return [(keys[i], keys[j]) for i, j in sorted(chosen)]


def run_align_stage(
    data_root: Path,
    split: str,
    workers: int,
    match_quantile: float,
    gap: float,
    max_pairs_per_song: int,
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
    n_pairs = 0
    for song_id, keys in by_song.items():
        pairs = clique_pairs(keys, max_pairs_per_song)
        n_pairs += len(pairs)
        for key_a, key_b in pairs:
            out_path = out_dir / f"{song_id}_{key_a.split('_')[1]}_{key_b.split('_')[1]}.npz"
            if not out_path.exists():
                jobs.append((in_dir / f"{key_a}.npz", in_dir / f"{key_b}.npz", out_path))

    print(f"[align/{split}] {n_pairs} cover pairs selected, {len(jobs)} to align")
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
            stem, error, _ = future.result()
            if error:
                failures += 1
                print(f"[align/{split}] {stem}: {error}")
            if i % 200 == 0 or i == len(futures):
                print(f"[align/{split}] {i}/{len(futures)} ({failures} failed)")


# --------------------------------------------------------------------------- #
# Stage 2, target mode: a fixed number of usable pairs per work
# --------------------------------------------------------------------------- #


def round_robin_pairs(song_id: int, keys: list[str]) -> Iterator[tuple[str, str]]:
    """Every cover pair of a clique exactly once, ordered for track coverage.

    The keys are shuffled with an RNG seeded by the song id, then scheduled by
    the circle method for round-robin tournaments: each round is a perfect
    matching, so the first n/2 candidates already touch every track of the
    clique, and no pair repeats. Stopping after the first K usable pairs
    therefore spreads the cover supervision over as many distinct recordings
    (i.e. styles) as possible. Random sampling would instead reuse whichever
    covers it happened to draw twice.

    The order depends on the exact key set: adding or removing a track
    reshuffles the clique. That is why the chosen pairs are recorded in a
    selection file rather than recomputed downstream.
    """
    players: list[str | None] = sorted(keys)
    random.Random(song_id).shuffle(players)
    if len(players) % 2:
        players.append(None)  # a bye: its partner sits out that round
    m = len(players)
    fixed, rest = players[0], players[1:]
    for r in range(m - 1):
        arrangement = [fixed] + rest[r:] + rest[:r]
        for i in range(m // 2):
            a, b = arrangement[i], arrangement[m - 1 - i]
            if a is not None and b is not None:
                yield (a, b) if a < b else (b, a)


def pair_stem(song_id: int, key_a: str, key_b: str) -> str:
    return f"{song_id}_{key_a.split('_')[1]}_{key_b.split('_')[1]}"


def read_score(path: Path) -> float | None:
    """Score of an alignment already on disk; None if it must be (re)computed.

    Files written before align_worker became atomic can be truncated. Those
    are deleted and redone instead of crashing the whole stage.
    """
    if not path.exists():
        return None
    try:
        return float(np.load(path)["score"])
    except Exception:
        path.unlink(missing_ok=True)
        return None


@dataclass
class CliqueProgress:
    """How far one work has walked its candidate order, and what it found."""

    n_tracks: int
    candidates: Iterator[tuple[str, str]]
    walked: list[str] = field(default_factory=list)  # stems, in candidate order
    scores: dict[str, float] = field(default_factory=dict)  # nan = failed
    exhausted: bool = False

    def passing(self, min_score: float) -> list[str]:
        """Stems that cleared the threshold, in candidate order."""
        return [
            s for s in self.walked if self.scores.get(s, math.nan) >= min_score
        ]

    def keep_rate(self, min_score: float, prior: float) -> float:
        """Observed pass rate once there is a little evidence, else the prior."""
        resolved = [s for s in self.walked if s in self.scores]
        if len(resolved) < 10:
            return prior
        return len(self.passing(min_score)) / len(resolved)


def run_target_align_stage(
    data_root: Path,
    split: str,
    workers: int,
    match_quantile: float,
    gap: float,
    kept_per_song: int,
    min_score: float,
    max_pairs_per_song: int,
    keep_rate: float,
) -> None:
    """Align each work's candidates in round-robin order until K pass.

    Works run in parallel rounds. Each round, every unfinished work queues
    enough new candidates to reach K at its expected keep rate, so the typical
    work needs two or three rounds, and a work whose covers rarely align stops
    at the --max-pairs-per-song budget instead of burning compute. The first K
    passing pairs in candidate order are the selection; extra passing pairs
    from an overshooting round stay on disk but are not selected.
    """
    in_dir = chroma_dir(data_root, split)
    out_dir = alignment_dir(data_root, split)
    if not in_dir.is_dir():
        print(f"[align/{split}] no chroma directory {in_dir}; run --stage chroma first")
        return

    # Leaked tracks never reach a manifest, so a pair that uses one would be
    # selected here and then silently dropped, leaving the work short of K.
    allowed = {track.key for track in split_tracks(split)}
    by_song: dict[int, list[str]] = defaultdict(list)
    not_allowed = 0
    for path in sorted(in_dir.glob("*.npz")):
        if path.stem not in allowed:
            not_allowed += 1
            continue
        by_song[int(path.stem.split("_")[0])].append(path.stem)
    if not_allowed:
        print(f"[align/{split}] ignoring {not_allowed} chroma files outside the split")

    progress = {
        song_id: CliqueProgress(len(keys), round_robin_pairs(song_id, keys))
        for song_id, keys in sorted(by_song.items())
        if len(keys) >= 2
    }
    budget = max_pairs_per_song if max_pairs_per_song > 0 else math.inf
    print(
        f"[align/{split}] target {kept_per_song} pairs scoring >= {min_score} per "
        f"work, <= {max_pairs_per_song or 'all'} candidates each; "
        f"{len(progress)} works with >= 2 chroma tracks"
    )

    def active(p: CliqueProgress) -> bool:
        return (
            not p.exhausted
            and len(p.walked) < budget
            and len(p.passing(min_score)) < kept_per_song
        )

    computed = failures = 0
    started = time.time()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_numba_private_cache
    ) as pool:
        for round_no in itertools.count(1):
            jobs: list[tuple[CliqueProgress, str, Path, Path, Path]] = []
            n_active = reused = 0
            for song_id, p in progress.items():
                if not active(p):
                    continue
                n_active += 1
                need = kept_per_song - len(p.passing(min_score))
                rate = max(p.keep_rate(min_score, keep_rate), 0.05)
                take = min(math.ceil(need / rate), budget - len(p.walked))
                for _ in range(int(take)):
                    pair = next(p.candidates, None)
                    if pair is None:
                        p.exhausted = True
                        break
                    stem = pair_stem(song_id, *pair)
                    p.walked.append(stem)
                    out_path = out_dir / f"{stem}.npz"
                    score = read_score(out_path)
                    if score is not None:
                        p.scores[stem] = score
                        reused += 1
                    else:
                        jobs.append(
                            (p, stem, in_dir / f"{pair[0]}.npz", in_dir / f"{pair[1]}.npz", out_path)
                        )
            if n_active == 0:
                break
            print(
                f"[align/{split}] round {round_no}: {n_active} works below target; "
                f"{reused} candidates already aligned, {len(jobs)} to align"
            )
            futures = {
                pool.submit(align_worker, a, b, out, match_quantile, gap): (p, stem)
                for p, stem, a, b, out in jobs
            }
            for i, future in enumerate(as_completed(futures), 1):
                p, stem = futures[future]
                _, error, score = future.result()
                p.scores[stem] = score
                computed += 1
                if error:
                    failures += 1
                    print(f"[align/{split}] {stem}: {error}")
                if i % 500 == 0 or i == len(futures):
                    rate = computed / max(time.time() - started, 1e-9)
                    print(
                        f"[align/{split}] round {round_no}: {i}/{len(futures)} "
                        f"({failures} failed so far, {rate:.1f} pairs/s)"
                    )

    write_selection(
        data_root, split, progress, kept_per_song, min_score, max_pairs_per_song,
        match_quantile, gap,
    )
    print(f"[align/{split}] {computed} alignments computed this run ({failures} failed)")


def write_selection(
    data_root: Path,
    split: str,
    progress: dict[int, CliqueProgress],
    kept_per_song: int,
    min_score: float,
    max_pairs_per_song: int,
    match_quantile: float,
    gap: float,
) -> None:
    works = {}
    short = {"exhausted": 0, "budget": 0}
    walked = passing_total = 0
    for song_id, p in progress.items():
        passing = p.passing(min_score)
        walked += len(p.walked)
        passing_total += len(passing)
        if len(passing) < kept_per_song:
            short["exhausted" if p.exhausted else "budget"] += 1
        works[str(song_id)] = {
            "pairs": passing[:kept_per_song],
            "candidates": len(p.walked),
            "n_tracks": p.n_tracks,
        }
    path = selection_path(data_root, split, kept_per_song)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "kept_per_song": kept_per_song,
                "min_score": min_score,
                "max_pairs_per_song": max_pairs_per_song,
                "match_quantile": match_quantile,
                "gap": gap,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "works": works,
            },
            f,
            indent=1,
        )

    counts = sorted(len(w["pairs"]) for w in works.values())
    n_selected = sum(counts)
    print(
        f"[align/{split}] selection -> {path}\n"
        f"[align/{split}]   {len(works)} works, {n_selected} pairs selected; "
        f"{len(works) - short['exhausted'] - short['budget']} reached "
        f"{kept_per_song}, {short['exhausted']} ran out of candidate pairs, "
        f"{short['budget']} hit the candidate budget"
    )
    if counts:
        print(
            f"[align/{split}]   pairs per work: min {counts[0]}, "
            f"median {counts[len(counts) // 2]}, max {counts[-1]}; "
            f"keep rate {passing_total / max(walked, 1):.1%} over {walked} candidates"
        )


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
        "--max-pairs-per-song",
        type=int,
        default=200,
        help="cap on aligned pairs per clique (0 = every pair). Cliques reach "
        "1991 versions in the 2025 dataset, so the uncapped train split is ~9M "
        "alignments; 200 keeps it near 340k. With --kept-per-song this is the "
        "per-work candidate budget instead",
    )
    parser.add_argument(
        "--kept-per-song",
        type=int,
        default=0,
        help="target mode: align each work's candidates in round-robin order "
        "until this many score >= --min-score, and record them in "
        "alignments/{split}/selection_k{K}.json (0 = off: align the first "
        "--max-pairs-per-song random pairs, the phase-1 behaviour)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.2,
        help="target mode: score a pair needs to count toward --kept-per-song "
        "(use the same value as build_manifest.py --min-score)",
    )
    parser.add_argument(
        "--keep-rate",
        type=float,
        default=0.4,
        help="target mode: expected fraction of candidates that pass, used to "
        "size a work's first round (phase 1 measured 42%%)",
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
        if args.stage in ("align", "all") and args.kept_per_song > 0:
            run_target_align_stage(
                args.data_root,
                split,
                args.workers,
                args.match_quantile,
                args.gap,
                args.kept_per_song,
                args.min_score,
                args.max_pairs_per_song,
                args.keep_rate,
            )
        elif args.stage in ("align", "all"):
            run_align_stage(
                args.data_root,
                split,
                args.workers,
                args.match_quantile,
                args.gap,
                args.max_pairs_per_song,
            )


if __name__ == "__main__":
    main()
