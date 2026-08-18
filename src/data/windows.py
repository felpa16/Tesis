"""On-the-fly S-second window datasets over full downloaded tracks.

Windows are never materialized to disk: each __getitem__ seeks into the
source file with ffmpeg and decodes exactly one window at the MERT sample
rate. Aligned cover pairs map a random anchor point on the stored warping
path (data/alignments/) from cover A's timeline to cover B's.

Both datasets return raw waveforms; MERT runs inside the training loop
(phase-1 online feature strategy, see CLAUDE.md "Feature caching strategy").
"""

from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.manifest import PairEntry, TrackEntry


@dataclass(frozen=True)
class WindowConfig:
    window_seconds: float = 20.0
    sample_rate: int = 24000  # MERT-v1-330M input rate
    # MIL-NCE hooks: number of candidate B-windows per aligned anchor and the
    # max jitter (seconds) of each extra candidate around the aligned point.
    n_candidates: int = 1
    candidate_spread_seconds: float = 4.0

    @property
    def window_samples(self) -> int:
        return round(self.window_seconds * self.sample_rate)


def decode_window(
    audio_path: Path, start_seconds: float, config: WindowConfig
) -> torch.Tensor:
    """Decode one window as mono float32 at config.sample_rate, exact length."""
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{max(start_seconds, 0.0):.3f}",
        "-i", str(audio_path),
        "-t", f"{config.window_seconds:.3f}",
        "-f", "f32le", "-ac", "1", "-ar", str(config.sample_rate),
        "pipe:1",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    wave = np.frombuffer(out, dtype=np.float32).copy()
    n = config.window_samples
    if len(wave) < n:
        wave = np.pad(wave, (0, n - len(wave)))
    return torch.from_numpy(wave[:n])


def _clamped_start(center: float, duration: float, window: float) -> float:
    """Window start so the window is centered at `center` but stays in-bounds."""
    return min(max(center - window / 2.0, 0.0), max(duration - window, 0.0))


class TrackWindowDataset(Dataset):
    """One random window per track per epoch (plain-reconstruction stream)."""

    def __init__(
        self, tracks: list[TrackEntry], data_root: Path, config: WindowConfig
    ) -> None:
        self.tracks = [t for t in tracks if t.duration >= config.window_seconds]
        self.data_root = data_root
        self.config = config

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, index: int) -> dict:
        track = self.tracks[index]
        start = random.uniform(0.0, track.duration - self.config.window_seconds)
        wave = decode_window(self.data_root / track.audio, start, self.config)
        return {"wave": wave, "key": track.key, "start": start}


class AlignedPairDataset(Dataset):
    """Aligned cover-pair windows (contrastive + cover-swap streams).

    Each item picks a random anchor on the pair's warping path, maps it from
    A's timeline to B's, and decodes a window centered on each side. With
    config.n_candidates > 1, extra B-windows are decoded at jittered centers
    around the aligned point (MIL-NCE-style soft positives).
    """

    def __init__(
        self,
        pairs: list[PairEntry],
        tracks: list[TrackEntry],
        data_root: Path,
        config: WindowConfig,
    ) -> None:
        durations = {t.key: t.duration for t in tracks}
        paths = {t.key: t.audio for t in tracks}
        self.pairs = [
            p
            for p in pairs
            if durations.get(p.key_a, 0.0) >= config.window_seconds
            and durations.get(p.key_b, 0.0) >= config.window_seconds
        ]
        self.durations = durations
        self.audio_paths = paths
        self.data_root = data_root
        self.config = config

    def __len__(self) -> int:
        return len(self.pairs)

    def _anchor(self, pair: PairEntry) -> tuple[float, float]:
        """Sample an aligned (t_a, t_b) anchor, preferring in-bounds centers."""
        data = np.load(self.data_root / pair.alignment)
        t_a, t_b = data["t_a"], data["t_b"]
        half = self.config.window_seconds / 2.0
        dur_a = self.durations[pair.key_a]
        dur_b = self.durations[pair.key_b]
        valid = (
            (t_a >= half)
            & (t_a <= dur_a - half)
            & (t_b >= half)
            & (t_b <= dur_b - half)
        )
        indices = np.flatnonzero(valid)
        if len(indices) == 0:  # short tracks: fall back to the path midpoint
            index = len(t_a) // 2
        else:
            index = int(random.choice(indices))
        return float(t_a[index]), float(t_b[index])

    def __getitem__(self, index: int) -> dict:
        pair = self.pairs[index]
        config = self.config
        anchor_a, anchor_b = self._anchor(pair)
        dur_a = self.durations[pair.key_a]
        dur_b = self.durations[pair.key_b]
        path_a = self.data_root / self.audio_paths[pair.key_a]
        path_b = self.data_root / self.audio_paths[pair.key_b]

        start_a = _clamped_start(anchor_a, dur_a, config.window_seconds)
        wave_a = decode_window(path_a, start_a, config)

        centers_b = [anchor_b] + [
            anchor_b + random.uniform(-1.0, 1.0) * config.candidate_spread_seconds
            for _ in range(config.n_candidates - 1)
        ]
        waves_b = torch.stack(
            [
                decode_window(
                    path_b, _clamped_start(c, dur_b, config.window_seconds), config
                )
                for c in centers_b
            ]
        )
        return {
            "wave_a": wave_a,
            "waves_b": waves_b,  # (n_candidates, window_samples)
            "key_a": pair.key_a,
            "key_b": pair.key_b,
            "score": pair.score,
            "oti": pair.oti,
            "anchor_a": anchor_a,  # aligned window centers, seconds
            "anchor_b": anchor_b,
        }
