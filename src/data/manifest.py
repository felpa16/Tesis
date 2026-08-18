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
