#!/usr/bin/env python3
"""Listen to same-artist style positives, blind, against random controls.

A style positive (scripts/mine_artist_pairs.py) is two recordings by the same
performer of two different works, from one album or released less than a year
apart. Training on them assumes the two share style: instrumentation, timbre,
vocal delivery, expressive timing, production. If they do not, they are
useless as positives. This script writes them to WAV so a human can check.

"Do these share style?" has no zero point on its own. Some random pairs of
recordings sound alike too (two 1950s vocal-and-orchestra sides), so the
verdicts are only meaningful against a baseline. The script therefore mixes in
random *control* pairs (two different performers, two different works, drawn
from the same pool of recordings) and shuffles everything. Judge each pair
without opening its metadata, then score: positives must be judged "same
style" far more often than controls.

Windows: one S-second window per recording, at a random point in the middle
60 % of the track (intros and outros are often sparser than the song), decoded
exactly as MERT sees it (24 kHz mono, src.data.windows.decode_window).

Output in --out:

  pair_XX_A.wav, pair_XX_B.wav, pair_XX_AB.wav   the two windows, and A-gap-B
  pairs.csv    pair, file_ab, same_style, notes   <- fill same_style with y / p / n
  key.csv      what each pair really is (positive or control) and its metadata;
               do not open it before judging
  index.html   players; metadata hidden under "reveal" per pair

Then:

    python scripts/listen_artist_pairs.py --score listen_artist_pairs

prints the share judged same-style (y = 1, p = 0.5, n = 0) for positives and
controls, overall and by relation (same_album, within_year).

Examples:
    python scripts/listen_artist_pairs.py --data-root $DATA --audio-s3 $AUDIO_S3
    python scripts/listen_artist_pairs.py --n 30 --controls 10 --relation same_album
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from extract_mert_features import quantize  # noqa: E402
from listen_pairs import (  # noqa: E402
    AudioSource,
    _clock,
    _downloaded_files,
    probe_duration,
    write_wav,
)
from mine_artist_pairs import performer_key  # noqa: E402
from shs100k_meta import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    YOUTUBE_URL,
    audio_dir,
    split_tracks,
)
from src.data.windows import WindowConfig, _clamped_start, decode_window  # noqa: E402

VERDICT = {"y": 1.0, "p": 0.5, "n": 0.0}


# --------------------------------------------------------------------------
# Choosing what to listen to
# --------------------------------------------------------------------------


def read_pairs(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no pairs at {path}; run scripts/mine_artist_pairs.py first")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def choose_positives(
    pairs: list[dict], n: int, relation: str, rng: random.Random
) -> list[dict]:
    """n pairs from n distinct performers, alternating relations when 'any'."""
    if relation != "any":
        pairs = [p for p in pairs if p["relation"] == relation]
    by_relation: dict[str, dict[str, list[dict]]] = {}
    for p in pairs:
        by_relation.setdefault(p["relation"], {}).setdefault(p["performer"], []).append(p)
    if not by_relation:
        raise SystemExit("no pairs to choose from")
    order = sorted(by_relation)
    used: set[str] = set()
    chosen: list[dict] = []
    pools = {r: rng.sample(sorted(by_relation[r]), len(by_relation[r])) for r in order}
    while len(chosen) < n and any(pools.values()):
        for r in order:
            while pools[r] and pools[r][-1] in used:
                pools[r].pop()
            if pools[r] and len(chosen) < n:
                performer = pools[r].pop()
                used.add(performer)
                chosen.append(rng.choice(by_relation[r][performer]))
    if len(chosen) < n:
        print(f"warning: only {len(chosen)} performers have pairs; writing {len(chosen)}")
    return chosen


def choose_controls(pairs: list[dict], n: int, rng: random.Random) -> list[dict]:
    """n random pairs of recordings by different performers, of different works.

    Drawn from the recordings that appear in the positives' pool, so controls
    and positives come from the same population of performers and eras.
    """
    pool: dict[str, dict] = {}
    for p in pairs:
        for side in ("a", "b"):
            pool[p[f"key_{side}"]] = {
                "key": p[f"key_{side}"], "work": p[f"work_{side}"],
                "artist": p[f"artist_{side}"], "title": p[f"title_{side}"],
                "date": p[f"date_{side}"], "album": p[f"album_{side}"],
                "performer": p["performer"],
            }
    recordings = sorted(pool.values(), key=lambda r: r["key"])
    if len({r["performer"] for r in recordings}) < 2:
        return []
    controls = []
    for _ in range(n * 50):
        if len(controls) == n:
            break
        a, b = rng.sample(recordings, 2)
        if a["performer"] == b["performer"] or a["work"] == b["work"]:
            continue
        if performer_key(a["artist"], "loose") == performer_key(b["artist"], "loose"):
            continue  # "X and His Trio" against "X with Orchestra": not random
        controls.append({
            "performer": f"{a['performer']} / {b['performer']}",
            "relation": "control",
            "key_a": a["key"], "key_b": b["key"],
            "work_a": a["work"], "work_b": b["work"],
            "artist_a": a["artist"], "artist_b": b["artist"],
            "title_a": a["title"], "title_b": b["title"],
            "date_a": a["date"], "date_b": b["date"],
            "album_a": a["album"], "album_b": b["album"],
            "max_gap_days": None,
        })
    return controls


def audio_paths(data_root: Path, split: str) -> dict[str, str]:
    """Track key -> audio path relative to the data root ("audio/train/X.webm").

    Any downloaded track can be in a style pair, not only the aligned subset a
    training manifest lists, so names come from the downloaded-songs CSV (under
    data_root, else the repo's data/) and from the files on disk.
    """
    names = _downloaded_files(data_root, split) or _downloaded_files(DEFAULT_DATA_ROOT, split)
    local = audio_dir(data_root, split)
    if local.is_dir():
        names |= {p.stem: p.name for p in local.iterdir() if p.is_file()}
    return {key: (Path("audio") / split / name).as_posix() for key, name in names.items()}


def middle_start(duration: float, window: float, rng: random.Random) -> float:
    """A window start whose centre lies in the middle 60 % of the track."""
    centre = rng.uniform(0.2 * duration, 0.8 * duration)
    return quantize(_clamped_start(centre, duration, window))


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def write_index(out: Path, rows: list[dict]) -> None:
    items = []
    for r in rows:
        def side(s: str) -> str:
            return (
                f"<li><b>{s.upper()}</b> {html.escape(r[f'artist_{s}'])} &mdash; "
                f"<i>{html.escape(r[f'title_{s}'])}</i> (work {r[f'work_{s}']}) "
                f"&middot; {html.escape(str(r[f'date_{s}'] or '?'))} &middot; "
                f"album: {html.escape(r[f'album_{s}'] or '?')} &middot; "
                f"{r[f'key_{s}']} @ {_clock(r[f'start_{s}'])} &middot; "
                f"<a href='{r[f'url_{s}']}&t={int(r[f'start_{s}'])}'>YouTube</a></li>"
            )
        items.append(
            f"<section><h2>Pair {r['pair']:02d}</h2>"
            f"<div class='row'><span>A</span><audio controls preload='none' "
            f"src='{r['file_a']}'></audio></div>"
            f"<div class='row'><span>B</span><audio controls preload='none' "
            f"src='{r['file_b']}'></audio></div>"
            f"<div class='row'><span>A&rarr;B</span><audio controls preload='none' "
            f"src='{r['file_ab']}'></audio></div>"
            f"<details><summary>reveal (after judging)</summary><p>{r['relation']}</p>"
            f"<ul>{side('a')}{side('b')}</ul></details></section>"
        )
    page = (
        "<!doctype html><meta charset='utf-8'><title>Style pairs</title>"
        "<style>body{font:15px system-ui;max-width:900px;margin:2em auto;padding:0 16px}"
        "section{border-top:1px solid #ccc;padding:.5em 0}"
        ".row{display:flex;align-items:center;gap:.6em;margin:.3em 0}"
        ".row span{width:3em}audio{width:100%;max-width:600px}"
        "details{margin-top:.4em;color:#555}</style>"
        "<h1>Style pairs</h1>"
        "<p>Each pair is two windows of two <em>different songs</em>. Some pairs are "
        "by the same performer, some are random. Judge only whether they share "
        "<b>style</b> (instrumentation, timbre, vocal delivery, expressive timing, "
        "production), not whether they are the same song. Write <code>y</code>, "
        "<code>p</code> (partly) or <code>n</code> in the <code>same_style</code> "
        "column of <code>pairs.csv</code>, before opening <i>reveal</i>.</p>"
        + "".join(items)
    )
    (out / "index.html").write_text(page, encoding="utf-8")


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def score(directory: Path) -> None:
    """Share judged same-style, positives against controls."""
    with open(directory / "pairs.csv", newline="", encoding="utf-8") as f:
        verdicts = {int(r["pair"]): r["same_style"].strip().lower()[:1]
                    for r in csv.DictReader(f)}
    with open(directory / "key.csv", newline="", encoding="utf-8") as f:
        key = {int(r["pair"]): r for r in csv.DictReader(f)}
    groups: dict[str, list[float]] = {}
    unlabelled = 0
    for pair, verdict in verdicts.items():
        if verdict not in VERDICT:
            unlabelled += 1
            continue
        relation = key[pair]["relation"]
        kind = "control" if relation == "control" else "positive"
        groups.setdefault(kind, []).append(VERDICT[verdict])
        if kind == "positive":
            groups.setdefault(f"  {relation}", []).append(VERDICT[verdict])
    print(f"{'group':<16} {'n':>4} {'same style':>11}   95% interval (y/n only)")
    for name in ["positive", *sorted(g for g in groups if g.startswith("  ")), "control"]:
        values = groups.get(name, [])
        if not values:
            continue
        share = sum(values) / len(values)
        hard = [v for v in values if v != 0.5]
        lo, hi = wilson(sum(hard), len(hard))
        print(f"{name:<16} {len(values):>4} {share:>11.2f}   [{lo:.2f}, {hi:.2f}]")
    if unlabelled:
        print(f"({unlabelled} pairs without a y/p/n verdict ignored)")


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--score", type=Path, default=None, metavar="DIR",
                        help="score a judged listening folder instead of writing one")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("DATA", DEFAULT_DATA_ROOT)),
                        help="default: $DATA if set, else the repo's data/")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--pairs", type=Path, default=None,
                        help="mined pairs (default: data/artist_pairs/{split}.jsonl)")
    parser.add_argument("--n", type=int, default=24, help="positives, one per performer")
    parser.add_argument("--controls", type=int, default=None,
                        help="random different-performer pairs mixed in (default n/3)")
    parser.add_argument("--relation", default="any",
                        choices=("any", "same_album", "within_year"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=20.0)
    parser.add_argument("--audio-s3", default=None,
                        help="s3://BUCKET/PREFIX holding {split}/ audio, for missing tracks")
    parser.add_argument("--gap-seconds", type=float, default=1.5)
    parser.add_argument("--out", type=Path, default=Path("listen_artist_pairs"))
    args = parser.parse_args()

    if args.score is not None:
        score(args.score)
        return

    rng = random.Random(args.seed)
    config = WindowConfig(window_seconds=args.window_seconds)
    pairs = read_pairs(args.pairs or DEFAULT_DATA_ROOT / "artist_pairs" / f"{args.split}.jsonl")
    positives = choose_positives(pairs, args.n, args.relation, rng)
    n_controls = round(len(positives) / 3) if args.controls is None else args.controls
    controls = choose_controls(pairs, n_controls, rng)
    chosen = positives + controls
    rng.shuffle(chosen)
    print(f"{len(pairs)} mined pairs; writing {len(positives)} positives and "
          f"{len(controls)} controls, shuffled")

    paths = audio_paths(args.data_root, args.split)
    for p in chosen:
        for key in (p["key_a"], p["key_b"]):
            paths.setdefault(key, f"audio/{args.split}/{key}.webm")
    videos = {t.key: t.youtube_id for t in split_tracks(args.split)}

    args.out.mkdir(parents=True, exist_ok=True)
    audio = AudioSource(args.data_root, paths, args.audio_s3, args.out / "audio")
    gap = np.zeros(round(args.gap_seconds * config.sample_rate), dtype=np.float32)
    rows = []
    for i, p in enumerate(chosen, start=1):
        stem = f"pair_{i:02d}"
        row = dict(p, pair=i, file_ab=f"{stem}_AB.wav")
        waves = []
        for side in ("a", "b"):
            key = p[f"key_{side}"]
            path = audio.path(key)
            duration = probe_duration(path)
            start = middle_start(duration, config.window_seconds, rng)
            wave = decode_window(path, start, config).numpy()
            write_wav(args.out / f"{stem}_{side.upper()}.wav", wave, config.sample_rate)
            waves.append(wave)
            row |= {
                f"start_{side}": start,
                f"file_{side}": f"{stem}_{side.upper()}.wav",
                f"url_{side}": YOUTUBE_URL.format(video_id=videos.get(key, "")),
            }
        write_wav(args.out / f"{stem}_AB.wav", np.concatenate([waves[0], gap, waves[1]]),
                  config.sample_rate)
        rows.append(row)
        print(f"{stem}  {p['key_a']}@{_clock(row['start_a'])}  "
              f"{p['key_b']}@{_clock(row['start_b'])}")

    with open(args.out / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "file_ab", "same_style", "notes"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**r, "same_style": "", "notes": ""} for r in rows)
    key_columns = ["pair", "relation", "performer", "key_a", "start_a", "key_b", "start_b",
                   "artist_a", "title_a", "work_a", "date_a", "album_a",
                   "artist_b", "title_b", "work_b", "date_b", "album_b",
                   "max_gap_days", "url_a", "url_b"]
    with open(args.out / "key.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=key_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_index(args.out, rows)
    print(f"wrote {len(rows)} pairs to {args.out}/ -- open {args.out}/index.html, fill "
          f"pairs.csv, then: python scripts/listen_artist_pairs.py --score {args.out}")


if __name__ == "__main__":
    main()
