"""Manifest entries describing the usable dataset after preprocessing.

The manifest is the bridge between offline preprocessing (download, chroma,
alignment — see scripts/) and training: it lists which tracks survived, which
cover pairs passed the alignment quality filter, and where their files live.
Paths are stored relative to the data root so manifests are portable between
machines (local smoke tests vs. AWS).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrackEntry:
    key: str
    song_id: int
    version_id: int
    audio: str  # path relative to data root
    duration: float  # seconds
    has_chroma: bool


@dataclass(frozen=True)
class PairEntry:
    key_a: str
    key_b: str
    song_id: int
    score: float
    oti: int
    alignment: str  # path relative to data root
    t_a_first: float  # aligned span endpoints, seconds
    t_a_last: float
    t_b_first: float
    t_b_last: float
    n_points: int


@dataclass(frozen=True)
class WindowPairEntry:
    """One materialized aligned window pair — the unit phase-2 training consumes.

    Written by scripts/extract_mert_features.py once the MERT mixes for both
    sides have been cached. `window_a`/`window_b` are the stems of the cached
    feature objects, so the content and style mixes of side A live at
    mert-features/{content,style}/{split}/{window_a}.npy.
    """

    song_id: int
    key_a: str
    key_b: str
    anchor_a: float  # the aligned point, seconds, in each cover's timeline
    anchor_b: float
    start_a: float  # window start after clamping to the track bounds
    start_b: float
    score: float
    oti: int
    window_a: str  # cached feature stem for side A
    window_b: str
    n_frames: int


def manifest_dir(data_root: Path, split: str) -> Path:
    return data_root / "manifests" / split


def write_jsonl(path: Path, entries: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry)) + "\n")


def read_tracks(data_root: Path, split: str) -> list[TrackEntry]:
    path = manifest_dir(data_root, split) / "tracks.jsonl"
    with open(path, encoding="utf-8") as f:
        return [TrackEntry(**json.loads(line)) for line in f if line.strip()]


def read_pairs(data_root: Path, split: str) -> list[PairEntry]:
    path = manifest_dir(data_root, split) / "pairs.jsonl"
    with open(path, encoding="utf-8") as f:
        return [PairEntry(**json.loads(line)) for line in f if line.strip()]


def read_window_pairs(data_root: Path, split: str) -> list[WindowPairEntry]:
    path = manifest_dir(data_root, split) / "window_pairs.jsonl"
    with open(path, encoding="utf-8") as f:
        return [WindowPairEntry(**json.loads(line)) for line in f if line.strip()]
