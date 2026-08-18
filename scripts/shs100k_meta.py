"""Shared utilities for reading SHS100K metadata and locating dataset files.

SHS100K2/meta/list columns (tab-separated):
    song_id  version_id  title  artist  youtube_url  flag

Split files (SHS100K-TRAIN / SHS100K-VAL / SHS100K-TEST) contain:
    song_id  version_id

A "song" (clique) groups all covers of one composition; a "track" is one
specific cover. Tracks are identified everywhere by the key "{song_id}_{version_id}".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
META_DIR = REPO_ROOT / "SHS100K2" / "meta"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"

SPLITS = ("train", "val", "test")
SPLIT_FILES = {
    "train": META_DIR / "SHS100K-TRAIN",
    "val": META_DIR / "SHS100K-VAL",
    "test": META_DIR / "SHS100K-TEST",
}


@dataclass(frozen=True)
class Track:
    song_id: int
    version_id: int
    title: str
    artist: str
    url: str

    @property
    def key(self) -> str:
        """Unique filename stem for this track, e.g. '5982_0'."""
        return f"{self.song_id}_{self.version_id}"


def load_tracks(list_path: Path = META_DIR / "list") -> dict[tuple[int, int], Track]:
    """Parse the master metadata list into {(song_id, version_id): Track}."""
    tracks: dict[tuple[int, int], Track] = {}
    with open(list_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            song_id, version_id, title, artist, url = parts[:5]
            track = Track(int(song_id), int(version_id), title, artist, url)
            tracks[(track.song_id, track.version_id)] = track
    return tracks


def load_split(split: str) -> list[tuple[int, int]]:
    """Return the (song_id, version_id) pairs belonging to a split."""
    pairs: list[tuple[int, int]] = []
    with open(SPLIT_FILES[split], encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def split_tracks(split: str) -> list[Track]:
    """Tracks of a split that exist in the master list (missing ones are dropped)."""
    tracks = load_tracks()
    pairs = load_split(split)
    found = [tracks[p] for p in pairs if p in tracks]
    missing = len(pairs) - len(found)
    if missing:
        print(f"[{split}] {missing} split entries not found in meta/list (skipped)")
    return found


def audio_dir(data_root: Path, split: str) -> Path:
    return data_root / "audio" / split


def chroma_dir(data_root: Path, split: str) -> Path:
    return data_root / "chroma" / split


def alignment_dir(data_root: Path, split: str) -> Path:
    return data_root / "alignments" / split


def existing_audio(data_root: Path, split: str) -> dict[str, Path]:
    """Map of track key -> audio file for everything already fully downloaded."""
    directory = audio_dir(data_root, split)
    if not directory.is_dir():
        return {}
    files: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and not path.name.startswith(".") and ".part" not in path.name:
            files[path.stem] = path
    return files
