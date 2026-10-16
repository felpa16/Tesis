#!/usr/bin/env python3
"""Listen to the positive pairs the contrastive loss trains on.

The content loss is only meaningful if an aligned cover pair really shares
melody and harmony at the aligned point. This script writes N such pairs to WAV
so a human can check that by ear: each pair is two S-second windows of the same
work (SHS clique), taken from two different recordings, at the point the
alignment maps one onto the other. Every pair comes from a different work.

Pairs are taken from the same place training takes them, in this order:

  1. manifests/{split}/window_pairs.jsonl -- the materialized phase-2 windows;
     the stored start offsets are used exactly as is.
  2. manifests/{split}/pairs.jsonl -- anchors from the same evenly spaced
     points extract_mert_features.py materializes (--windows-per-pair), one
     picked at random.
  3. alignments/{split}/*.npz -- when no manifest exists locally, the alignment
     files themselves, filtered by --min-score and, if selection_k{K}.json is
     present, by the per-work selection (--kept-per-song). Same anchors as 2.

Audio is decoded with src.data.windows.decode_window: 24 kHz mono, the exact
waveform MERT sees. Tracks missing from data/audio/{split}/ are fetched with
the aws CLI from --audio-s3 (the AUDIO_S3 prefix of the runbook) into
{out}/audio/, so only the tracks actually needed are downloaded.

Output in --out:

  pair_XX_A.wav, pair_XX_B.wav   the two windows
  pair_XX_AB.wav                 A, a short silence, then B: back-to-back listening
  pairs.csv                      metadata, with an empty `same_content` column to fill in
  index.html                     players + metadata in one page (open it in a browser)

Remember `oti` when listening: B may be transposed relative to A (OTI = the
chroma shift, in semitones, the aligner applied). That is a valid cover.

Examples:
    python scripts/listen_pairs.py --data-root $DATA --split train_w25
    python scripts/listen_pairs.py --split train --audio-s3 $AUDIO_S3
    python scripts/listen_pairs.py --data-root $DATA --split train --n 15 --seed 1
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from extract_mert_features import pair_anchors, quantize  # noqa: E402
from shs100k_meta import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    Track,
    _parse_split,
    alignment_dir,
    audio_dir,
    downloaded_csv,
    selection_path,
)
from src.data.manifest import (  # noqa: E402
    PairEntry,
    manifest_dir,
    read_pairs,
    read_tracks,
    read_window_pairs,
)
from src.data.windows import WindowConfig, _clamped_start, decode_window  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    """One positive pair: either a fixed window pair, or a pair to anchor."""

    song_id: int
    key_a: str
    key_b: str
    score: float
    oti: int
    pair: PairEntry | None = None  # set when the anchor still has to be chosen
    start_a: float | None = None  # set when the windows are already materialized
    start_b: float | None = None


# --------------------------------------------------------------------------
# Where the pairs come from
# --------------------------------------------------------------------------


def pairs_from_alignments(
    data_root: Path, split: str, min_score: float, kept_per_song: int
) -> list[PairEntry]:
    """PairEntry for every alignment that passes the training filters."""
    directory = alignment_dir(data_root, split)
    selection = selection_path(data_root, split, kept_per_song)
    if kept_per_song > 0 and selection.exists():
        with open(selection, encoding="utf-8") as f:
            works = json.load(f)["works"]
        paths = [directory / f"{s}.npz" for w in works.values() for s in w["pairs"]]
        print(f"using the K={kept_per_song} selection in {selection}")
    else:
        if kept_per_song > 0:
            print(f"no selection at {selection}; using every alignment above --min-score")
        paths = sorted(directory.glob("*.npz"))

    downloaded = _downloaded_files(data_root, split)
    pairs = []
    skipped = {"not downloaded": 0, "missing file": 0, "below --min-score": 0}
    for path in paths:
        song_id, ver_a, ver_b = (int(x) for x in path.stem.split("_"))
        key_a, key_b = f"{song_id}_{ver_a}", f"{song_id}_{ver_b}"
        if downloaded and (key_a not in downloaded or key_b not in downloaded):
            skipped["not downloaded"] += 1
            continue
        if not path.exists():
            skipped["missing file"] += 1
            continue
        data = np.load(path)
        score = float(data["score"])
        if score < min_score:
            skipped["below --min-score"] += 1
            continue
        t_a, t_b = data["t_a"], data["t_b"]
        pairs.append(
            PairEntry(
                key_a=key_a,
                key_b=key_b,
                song_id=song_id,
                score=score,
                oti=int(data["oti"]),
                alignment=str(path.relative_to(data_root)),
                t_a_first=float(t_a[0]),
                t_a_last=float(t_a[-1]),
                t_b_first=float(t_b[0]),
                t_b_last=float(t_b[-1]),
                n_points=int(len(t_a)),
            )
        )
    print(f"{len(paths)} alignment files, {len(pairs)} kept; skipped: "
          + ", ".join(f"{n} {why}" for why, n in skipped.items()))
    return pairs


def load_candidates(
    data_root: Path, split: str, min_score: float, kept_per_song: int
) -> list[Candidate]:
    directory = manifest_dir(data_root, split)
    window_pairs = directory / "window_pairs.jsonl"
    if window_pairs.exists() and window_pairs.stat().st_size > 0:
        print(f"pairs from {window_pairs}")
        return [
            Candidate(w.song_id, w.key_a, w.key_b, w.score, w.oti,
                      start_a=w.start_a, start_b=w.start_b)
            for w in read_window_pairs(data_root, split)
        ]
    pairs_file = directory / "pairs.jsonl"
    if pairs_file.exists() and pairs_file.stat().st_size > 0:
        print(f"pairs from {pairs_file}")
        pairs = read_pairs(data_root, split)
    else:
        alignments = alignment_dir(data_root, split)
        print(f"no manifest pairs at {pairs_file}; reading {alignments}")
        if not alignments.is_dir():
            manifests = sorted(
                d.name for d in (data_root / "manifests").glob("*")
                if (d / "pairs.jsonl").exists()
            )
            raise SystemExit(
                f"{alignments} does not exist. Is --data-root right (it defaults to "
                f"$DATA, else the repo's data/)? Manifests with pairs under "
                f"{data_root}: {', '.join(manifests) or 'none'}"
            )
        pairs = pairs_from_alignments(data_root, split, min_score, kept_per_song)
    return [Candidate(p.song_id, p.key_a, p.key_b, p.score, p.oti, pair=p) for p in pairs]


def choose(candidates: list[Candidate], n: int, rng: random.Random) -> list[Candidate]:
    """n candidates from n distinct works, one random pair per work."""
    by_song: dict[int, list[Candidate]] = {}
    for c in candidates:
        if c.key_a != c.key_b:
            by_song.setdefault(c.song_id, []).append(c)
    if len(by_song) < n:
        print(f"warning: only {len(by_song)} works have pairs; writing {len(by_song)}")
    songs = rng.sample(sorted(by_song), min(n, len(by_song)))
    return [rng.choice(by_song[s]) for s in songs]


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------


def _downloaded_files(data_root: Path, split: str) -> dict[str, str]:
    """Track key -> audio file name, from the split's downloaded-songs CSV."""
    path = downloaded_csv(data_root, split)
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        names = [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]
    return {Path(name).stem: name for name in names}


class AudioSource:
    """Local audio if present, otherwise a one-off `aws s3 cp` into a cache.

    `audio` maps track key -> path relative to the data root ("audio/train/
    10154_10161.webm"), as manifests store it. The S3 object is the same path
    with the leading "audio/" replaced by --audio-s3.
    """

    def __init__(
        self, data_root: Path, audio: dict[str, str], audio_s3: str | None, cache: Path
    ) -> None:
        self.data_root = data_root
        self.audio = audio
        self.audio_s3 = audio_s3.rstrip("/") if audio_s3 else None
        self.cache = cache

    def path(self, key: str) -> Path:
        relative = Path(self.audio[key])
        for candidate in (self.data_root / relative, self.cache / relative.name):
            if candidate.exists():
                return candidate
        if self.audio_s3 is None:
            raise SystemExit(
                f"no audio at {self.data_root / relative}; "
                f"pass --audio-s3 s3://BUCKET/PREFIX"
            )
        self.cache.mkdir(parents=True, exist_ok=True)
        target = self.cache / relative.name
        url = f"{self.audio_s3}/{relative.relative_to('audio').as_posix()}"
        result = subprocess.run(
            ["aws", "s3", "cp", url, str(target), "--only-show-errors"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise SystemExit(f"could not fetch {url}: {result.stderr.strip()}")
        return target


def audio_paths(data_root: Path, split: str) -> dict[str, str]:
    """Track key -> audio path relative to the data root.

    From the manifest's tracks.jsonl when there is one (this also covers
    derived splits such as train_w25, whose audio lives under audio/train/),
    else from the official split's downloaded-songs CSV.
    """
    if (manifest_dir(data_root, split) / "tracks.jsonl").exists():
        return {t.key: t.audio for t in read_tracks(data_root, split)}
    directory = audio_dir(data_root, split).relative_to(data_root)
    return {
        key: (directory / name).as_posix()
        for key, name in _downloaded_files(data_root, split).items()
    }


def titles(audio: dict[str, str]) -> dict[str, Track]:
    """SHS metadata for the tracks, from each one's official split (optional)."""
    meta: dict[str, Track] = {}
    for source in sorted({Path(p).parts[1] for p in audio.values()}):
        try:
            meta |= {t.key: t for t in _parse_split(source)}
        except Exception:  # the SHS CSVs only add titles; never fatal
            pass
    return meta


def probe_duration(path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "csv=p=0", str(path)]
    return float(subprocess.run(cmd, capture_output=True, check=True, text=True).stdout)


def write_wav(path: Path, wave_f32: np.ndarray, sample_rate: int) -> None:
    """16-bit PCM WAV, peak-normalized only if the window would clip."""
    peak = float(np.abs(wave_f32).max()) if wave_f32.size else 0.0
    if peak > 1.0:
        wave_f32 = wave_f32 / peak
    pcm = (np.clip(wave_f32, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _clock(seconds: float) -> str:
    tenths = round(seconds * 10)
    return f"{tenths // 600}:{(tenths % 600) / 10:04.1f}"


def write_index(out: Path, rows: list[dict]) -> None:
    items = []
    for r in rows:
        def side(s: str) -> str:
            return (
                f"<div class='side'><b>{s}</b> {html.escape(r[f'artist_{s.lower()}'])} "
                f"&mdash; <i>{html.escape(r[f'title_{s.lower()}'])}</i><br>"
                f"<small>{r[f'key_{s.lower()}']} @ {_clock(r[f'start_{s.lower()}'])}"
                f" &middot; <a href='{r[f'url_{s.lower()}']}&t={int(r[f'start_{s.lower()}'])}'>"
                f"YouTube</a></small><br>"
                f"<audio controls preload='none' src='{r[f'file_{s.lower()}']}'></audio></div>"
            )
        items.append(
            f"<section><h2>Pair {r['pair']:02d} &middot; work {r['song_id']}</h2>"
            f"<p>alignment score {r['score']:.3f} &middot; OTI {r['oti']} semitones</p>"
            f"{side('A')}{side('B')}"
            f"<div class='side'><b>A then B</b><br>"
            f"<audio controls preload='none' src='{r['file_ab']}'></audio></div></section>"
        )
    page = (
        "<!doctype html><meta charset='utf-8'><title>Positive pairs</title>"
        "<style>body{font:15px system-ui;max-width:900px;margin:2em auto;padding:0 16px}"
        "section{border-top:1px solid #ccc;padding:.5em 0}.side{margin:.6em 0}"
        "audio{width:100%;max-width:600px}</style>"
        "<h1>Positive pairs</h1><p>Same work, different recordings, aligned windows. "
        "Does each pair share melody and harmony? Record the verdict in "
        "<code>pairs.csv</code> (<code>same_content</code>).</p>" + "".join(items)
    )
    (out / "index.html").write_text(page, encoding="utf-8")


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("DATA", DEFAULT_DATA_ROOT)),
                        help="default: $DATA if set, else the repo's data/")
    parser.add_argument("--split", default="train",
                        help="manifest (train_w25, train_w100, ...) or official split")
    parser.add_argument("--n", type=int, default=15, help="pairs to write, one per work")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=20.0)
    parser.add_argument("--windows-per-pair", type=int, default=4,
                        help="anchors per pair to choose from, as in extract_mert_features.py")
    parser.add_argument("--min-score", type=float, default=0.2,
                        help="alignment keep threshold (alignment fallback only)")
    parser.add_argument("--kept-per-song", type=int, default=40,
                        help="per-work selection to honour (alignment fallback only)")
    parser.add_argument("--audio-s3", default=None,
                        help="s3://BUCKET/PREFIX holding {split}/ audio, for missing tracks")
    parser.add_argument("--gap-seconds", type=float, default=1.5,
                        help="silence between A and B in pair_XX_AB.wav")
    parser.add_argument("--out", type=Path, default=Path("listen_pairs"))
    args = parser.parse_args()

    config = WindowConfig(window_seconds=args.window_seconds)
    rng = random.Random(args.seed)
    candidates = load_candidates(args.data_root, args.split, args.min_score,
                                 args.kept_per_song)
    if not candidates:
        raise SystemExit("no positive pairs found")
    chosen = choose(candidates, args.n, rng)
    print(f"{len(candidates)} pairs over {len({c.song_id for c in candidates})} works; "
          f"writing {len(chosen)}")

    durations: dict[str, float] = {}
    manifest_tracks = manifest_dir(args.data_root, args.split) / "tracks.jsonl"
    if manifest_tracks.exists():
        durations = {t.key: t.duration for t in read_tracks(args.data_root, args.split)}
    paths = audio_paths(args.data_root, args.split)
    for c in chosen:
        for key in (c.key_a, c.key_b):
            # not in any listing: assume the usual layout and let the fetch say
            paths.setdefault(key, f"audio/{args.split}/{key}.webm")
    meta = titles({k: paths[k] for c in chosen for k in (c.key_a, c.key_b)})

    args.out.mkdir(parents=True, exist_ok=True)
    audio = AudioSource(args.data_root, paths, args.audio_s3, args.out / "audio")
    gap = np.zeros(round(args.gap_seconds * config.sample_rate), dtype=np.float32)
    rows = []
    for i, c in enumerate(chosen, start=1):
        path_a, path_b = audio.path(c.key_a), audio.path(c.key_b)
        if c.start_a is not None:
            start_a, start_b = c.start_a, c.start_b
        else:
            for key, path in ((c.key_a, path_a), (c.key_b, path_b)):
                if key not in durations:
                    durations[key] = probe_duration(path)
            anchor_a, anchor_b = rng.choice(
                pair_anchors(args.data_root, c.pair, durations, config,
                             args.windows_per_pair)
            )
            start_a = quantize(_clamped_start(anchor_a, durations[c.key_a],
                                              config.window_seconds))
            start_b = quantize(_clamped_start(anchor_b, durations[c.key_b],
                                              config.window_seconds))

        wave_a = decode_window(path_a, start_a, config).numpy()
        wave_b = decode_window(path_b, start_b, config).numpy()
        stem = f"pair_{i:02d}"
        write_wav(args.out / f"{stem}_A.wav", wave_a, config.sample_rate)
        write_wav(args.out / f"{stem}_B.wav", wave_b, config.sample_rate)
        write_wav(args.out / f"{stem}_AB.wav", np.concatenate([wave_a, gap, wave_b]),
                  config.sample_rate)

        row = {"pair": i, "song_id": c.song_id, "score": round(c.score, 3), "oti": c.oti,
               "file_ab": f"{stem}_AB.wav", "same_content": ""}
        for side, key, start in (("a", c.key_a, start_a), ("b", c.key_b, start_b)):
            track = meta.get(key)
            row |= {
                f"key_{side}": key,
                f"start_{side}": start,
                f"title_{side}": track.title if track else "",
                f"artist_{side}": track.artist if track else "",
                f"url_{side}": track.url if track else "",
                f"file_{side}": f"{stem}_{side.upper()}.wav",
            }
        rows.append(row)
        print(f"{stem}  work {c.song_id}  {c.key_a}@{_clock(start_a)}  "
              f"{c.key_b}@{_clock(start_b)}  score {c.score:.3f}  oti {c.oti}"
              f"  {row['artist_a']} / {row['artist_b']} - {row['title_a']}")

    columns = ["pair", "song_id", "key_a", "start_a", "key_b", "start_b", "score", "oti",
               "title_a", "artist_a", "title_b", "artist_b", "url_a", "url_b",
               "same_content"]
    with open(args.out / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_index(args.out, rows)
    print(f"wrote {len(rows)} pairs to {args.out}/ -- open {args.out}/index.html")


if __name__ == "__main__":
    main()
