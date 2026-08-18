"""Shared utilities for reading SHS100K metadata and locating dataset files.

Dataset: SHS-100K-official-2025, published by SecondHandSongs itself (the
`shs-100k/` checkout). One headerless CSV per split, five columns:

    performance_id, work_id, title, artist, youtube_id

Note the column order: the *performance* id comes first, the *work* id second.
Fields are comma-separated with standard CSV quoting, because titles and artist
names contain commas ("Nat King Cole with Orch. & Chorus cond. by ...").

A "song" (clique, SHS "work") groups all covers of one composition; a "track"
(SHS "performance") is one specific cover. Tracks keep the same identity
convention as the 2017 version of the dataset — the key "{song_id}_{version_id}"
— so the on-disk layout of audio/, chroma/, alignments/ and the manifests is
unchanged. Only the id values themselves differ.

Full YouTube URLs are not stored in the dataset; they are rebuilt from the
11-character video id.

Two integrity problems in the 2025 release are corrected here; see
HELD_OUT_AGAINST and `split_tracks`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "shs-100k"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"

YOUTUBE_URL = "https://youtube.com/watch?v={video_id}"

SPLITS = ("train", "val", "test")
SPLIT_FILES = {
    "train": DATASET_DIR / "train.csv",
    "val": DATASET_DIR / "validate.csv",
    "test": DATASET_DIR / "test.csv",
}

# The 2025 README claims "the musical works remain separate by subset". They are
# not: 40 train works also appear in test and 27 in validate, and 3,031 YouTube
# videos appear in more than one split. Loading a split listed here drops every
# track whose work — or whose individual video — also occurs in the splits it is
# meant to be disjoint from, so contamination cannot reach training.
#
# validate and test additionally share 2 works (5854, 186755) and 76 videos with
# each other. That is deliberately NOT resolved here: which of the two should
# lose them is an evaluation decision, not a loading one. `split_tracks` warns.
HELD_OUT_AGAINST: dict[str, tuple[str, ...]] = {"train": ("val", "test")}


@dataclass(frozen=True)
class Track:
    song_id: int  # SHS work id: the clique all covers of this composition share
    version_id: int  # SHS performance id: this particular cover
    title: str
    artist: str
    youtube_id: str

    @property
    def key(self) -> str:
        """Unique filename stem for this track, e.g. '221685_927680'."""
        return f"{self.song_id}_{self.version_id}"

    @property
    def url(self) -> str:
        return YOUTUBE_URL.format(video_id=self.youtube_id)


_parsed: dict[str, list[Track]] = {}
_warned_about_eval_overlap = False


def _parse_split(split: str) -> list[Track]:
    """Deduplicated tracks of one split's CSV, in file order. Cached.

    The 2025 CSVs contain exact duplicate rows (17,154 of train's 116,381).
    Dropping them is required, not just tidy: two download workers handed the
    same key would race on the same output file.
    """
    if split in _parsed:
        return _parsed[split]

    tracks: dict[tuple[int, int], Track] = {}
    duplicates = malformed = 0
    with open(SPLIT_FILES[split], newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 5:
                malformed += 1
                continue
            performance_id, work_id, title, artist, youtube_id = (
                field.strip() for field in row[:5]
            )
            try:
                track = Track(
                    song_id=int(work_id),
                    version_id=int(performance_id),
                    title=title,
                    artist=artist,
                    youtube_id=youtube_id,
                )
            except ValueError:  # non-numeric ids (a stray header row, say)
                malformed += 1
                continue
            if not track.youtube_id:
                malformed += 1
                continue
            if (track.song_id, track.version_id) in tracks:
                duplicates += 1
                continue
            tracks[(track.song_id, track.version_id)] = track

    notes = []
    if duplicates:
        notes.append(f"{duplicates} duplicate rows dropped")
    if malformed:
        notes.append(f"{malformed} unparseable rows skipped")
    suffix = f" ({', '.join(notes)})" if notes else ""
    print(f"[{split}] {len(tracks)} tracks in CSV{suffix}")

    _parsed[split] = list(tracks.values())
    return _parsed[split]


def work_ids(split: str) -> set[int]:
    """Work (clique) ids present in a split."""
    return {track.song_id for track in _parse_split(split)}


def video_ids(split: str) -> set[str]:
    """YouTube video ids present in a split."""
    return {track.youtube_id for track in _parse_split(split)}


def _warn_eval_overlap() -> None:
    global _warned_about_eval_overlap
    if _warned_about_eval_overlap:
        return
    _warned_about_eval_overlap = True
    shared_works = work_ids("val") & work_ids("test")
    shared_videos = video_ids("val") & video_ids("test")
    if shared_works or shared_videos:
        print(
            f"[warning] val and test share {len(shared_works)} works and "
            f"{len(shared_videos)} videos with each other; unresolved by design "
            f"(see HELD_OUT_AGAINST in shs100k_meta.py)"
        )


def split_tracks(split: str, drop_leaked: bool = True) -> list[Track]:
    """Tracks of one split, deduplicated and de-contaminated.

    With drop_leaked (the default), a split listed in HELD_OUT_AGAINST loses
    every track whose work id, or whose own video id, also appears in a split it
    must stay disjoint from. For train that removes 65 works / 3,032 tracks
    (3.1%), leaving 96,195 tracks in 1,639 works with no clique falling below
    two versions. Pass drop_leaked=False to measure the contaminated baseline.
    """
    tracks = _parse_split(split)
    if not drop_leaked:
        return list(tracks)

    if split in ("val", "test"):
        _warn_eval_overlap()

    against = HELD_OUT_AGAINST.get(split)
    if not against:
        return list(tracks)

    blocked_works: set[int] = set()
    blocked_videos: set[str] = set()
    for other in against:
        blocked_works |= work_ids(other)
        blocked_videos |= video_ids(other)

    kept: list[Track] = []
    leaked_works: set[int] = set()
    leaked_videos = 0
    for track in tracks:
        if track.song_id in blocked_works:
            leaked_works.add(track.song_id)
        elif track.youtube_id in blocked_videos:
            leaked_videos += 1
        else:
            kept.append(track)

    dropped = len(tracks) - len(kept)
    if dropped:
        print(
            f"[{split}] dropped {dropped} tracks leaking into "
            f"{'/'.join(against)}: {len(leaked_works)} shared works, "
            f"{leaked_videos} shared videos; {len(kept)} remain"
        )
    return kept


def split_songs(split: str, drop_leaked: bool = True) -> dict[int, list[Track]]:
    """Tracks of one split grouped into cliques, keyed by song (work) id."""
    songs: dict[int, list[Track]] = {}
    for track in split_tracks(split, drop_leaked=drop_leaked):
        songs.setdefault(track.song_id, []).append(track)
    return songs


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
