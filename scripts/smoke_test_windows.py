#!/usr/bin/env python3
"""End-to-end smoke test of the windowed datasets.

Checks, on the local val subset:
  1. Shapes/dtypes and decode latency of TrackWindowDataset and
     AlignedPairDataset (including multi-candidate MIL-NCE mode).
  2. Content validity of aligned windows, via the same beat-synchronous
     chroma + OTI + Smith-Waterman scoring used for offline alignment:
     the aligned (A, B) window pair must outscore both
       - the same A-window against a time-shifted B-window (same track), and
       - the same A-window against a window from a different song.
     The time-shifted control is the sharp one: it fails if window centers
     are mapped through the warping path incorrectly.

Example:
    python scripts/smoke_test_windows.py --n-pairs 8
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from align_covers import (  # noqa: E402
    beat_sync_chroma,
    optimal_transposition_index,
    smith_waterman_path,
)
from shs100k_meta import DEFAULT_DATA_ROOT  # noqa: E402
from src.data import (  # noqa: E402
    AlignedPairDataset,
    TrackWindowDataset,
    WindowConfig,
    read_pairs,
    read_tracks,
)
from src.data.windows import _clamped_start, decode_window  # noqa: E402


def window_pair_score(
    wave_a: np.ndarray, wave_b: np.ndarray, sr: int, quantile: float = 0.8
) -> float:
    """Alignment confidence between two windows (same scoring as offline)."""
    chroma_a, _ = beat_sync_chroma(wave_a, sr)
    chroma_b, _ = beat_sync_chroma(wave_b, sr)
    oti = optimal_transposition_index(chroma_a, chroma_b)
    sim = chroma_a.T @ np.roll(chroma_b, oti, axis=0)
    _, score = smith_waterman_path(sim, quantile, gap=0.5)
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--n-pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)

    tracks = read_tracks(args.data_root, args.split)
    pairs = read_pairs(args.data_root, args.split)
    config = WindowConfig()
    print(f"{len(tracks)} tracks, {len(pairs)} pairs; window={config.window_seconds}s")

    # 1a. plain windows
    plain = TrackWindowDataset(tracks, args.data_root, config)
    t0 = time.perf_counter()
    item = plain[random.randrange(len(plain))]
    dt = time.perf_counter() - t0
    assert item["wave"].shape == (config.window_samples,), item["wave"].shape
    assert item["wave"].dtype.is_floating_point
    print(f"plain window ok: {item['key']} shape={tuple(item['wave'].shape)} "
          f"decode={dt * 1000:.0f}ms")

    # 1b. aligned pairs, incl. MIL-NCE candidates
    milnce = WindowConfig(n_candidates=3)
    aligned = AlignedPairDataset(pairs, tracks, args.data_root, milnce)
    item = aligned[random.randrange(len(aligned))]
    assert item["wave_a"].shape == (milnce.window_samples,)
    assert item["waves_b"].shape == (3, milnce.window_samples)
    print(f"aligned pair ok: {item['key_a']}~{item['key_b']} "
          f"waves_b={tuple(item['waves_b'].shape)}")

    # 2. content check: aligned vs time-shifted vs cross-song windows
    aligned = AlignedPairDataset(pairs, tracks, args.data_root, config)
    indices = random.sample(range(len(aligned)), min(args.n_pairs, len(aligned)))
    sr = config.sample_rate
    s_aligned, s_shifted, s_cross = [], [], []
    for i in indices:
        item = aligned[i]
        wave_a = item["wave_a"].numpy()
        wave_b = item["waves_b"][0].numpy()
        s_aligned.append(window_pair_score(wave_a, wave_b, sr))

        # time-shifted control: same B track, window centered >=30s away
        pair = aligned.pairs[i]
        dur_b = aligned.durations[pair.key_b]
        path_b = args.data_root / aligned.audio_paths[pair.key_b]
        shift_center = random.uniform(0.0, dur_b)
        for _ in range(10):  # try to land far from the aligned center
            shift_center = random.uniform(0.0, dur_b)
            if dur_b < 60.0 or abs(shift_center - item["anchor_b"]) > 30.0:
                break
        wave_shift = decode_window(
            path_b, _clamped_start(shift_center, dur_b, config.window_seconds), config
        ).numpy()
        s_shifted.append(window_pair_score(wave_a, wave_shift, sr))

        other_songs = [
            j for j in range(len(aligned))
            if aligned.pairs[j].song_id != aligned.pairs[i].song_id
        ]
        other = aligned[random.choice(other_songs)]
        s_cross.append(window_pair_score(wave_a, other["waves_b"][0].numpy(), sr))

    ali, shi, cro = np.mean(s_aligned), np.mean(s_shifted), np.mean(s_cross)
    print(
        f"SW window scores (n={len(indices)}) — aligned: {ali:.3f}, "
        f"time-shifted same track: {shi:.3f}, cross-song: {cro:.3f}"
    )
    if ali > shi and ali > cro:
        print("PASS: aligned windows outscore both controls")
    else:
        print("FAIL: aligned windows do not outscore the controls")
        sys.exit(1)


if __name__ == "__main__":
    main()
