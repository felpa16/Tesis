#!/usr/bin/env python3
"""Measure what codec each source's audio actually is.

For --per-source songs of the human split and of each MoM generator (taken in
evaluate_mom.py's deterministic order, i.e. the head of the evaluation
sample), ffprobe reads codec, sample rate, channels, bitrate and duration.

Why. A generator whose files all share one codec, bitrate and sample rate can
be recognized by those alone, and a supervised probe will do so if the
signature survives into MERT (experiments.md, section 13). This report says
how fixed each source's format is and gives the two settings the codec
controls need:
  * per generator, its own settings, for evaluate_mert_probe.py --codec-check
    (human songs in that generator's codec)
  * a common chain for --codec (cache_pooled_mert.py, cache_pooled_mom.py,
    evaluate_mert_probe.py), below the lowest bitrate of any source, so that
    its artifacts, not those of each source's original codec, dominate on both
    sides

A share near 100 % in the "top bitrate" or "top duration" column means a
fixed setting (CBR at one rate, or one fixed length, as DiffRhythm's songs
have).

Example:
    python scripts/codec_report.py --data-root $DATA --bucket $BUCKET
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

import evaluate_mom as em  # noqa: E402
from shs100k_meta import DEFAULT_DATA_ROOT  # noqa: E402
from src.data.codec import ENCODERS  # noqa: E402

MP3_LADDER = (32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--human-split", default="test50")
    parser.add_argument("--generators", nargs="+", default=list(em.GENERATORS))
    parser.add_argument("--per-source", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bucket", default="songs-and-mert-features")
    parser.add_argument("--region", default=None)
    parser.add_argument("--mom-prefix", default="mom")
    parser.add_argument("--mom-root", type=Path, default=None, help="local MoM mirror instead of S3")
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--fetch-workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None, help="also write the per-file records as JSON")
    return parser.parse_args()


def number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name,sample_rate,channels,bit_rate:format=format_name,bit_rate,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    stream = (info.get("streams") or [{}])[0]
    fmt = info.get("format", {})
    bitrate = number(stream.get("bit_rate")) or number(fmt.get("bit_rate"))  # opus in webm: container only
    return {
        "codec": stream.get("codec_name"),
        "container": fmt.get("format_name"),
        "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        "channels": stream.get("channels"),
        "kbps": bitrate / 1000 if bitrate else None,
        "duration": number(fmt.get("duration")),
    }


def top(values: list, digits: int | None = None) -> tuple[object, float]:
    """Most common value (rounded to `digits`) and its share."""
    kept = [round(v, digits) if digits is not None else v for v in values if v is not None]
    if not kept:
        return None, 0.0
    value, count = Counter(kept).most_common(1)[0]
    return value, count / len(kept)


def shares(values: list) -> str:
    counts = Counter(values)
    return ", ".join(f"{k} {v / len(values):.0%}" for k, v in counts.most_common(3))


def main() -> None:
    args = parse_args()
    s3 = None
    if args.mom_root is None:
        from src.s3 import S3

        s3 = S3(args.bucket, args.region)
    groups = [em.deterministic_sample(em.human_songs(args.data_root, args.human_split, 20.0),
                                      args.per_source, args.seed)]
    for generator in args.generators:
        _, songs = (em.mom_songs_local(args.mom_root, generator) if args.mom_root is not None
                    else em.mom_songs_s3(s3, args.mom_prefix, generator))
        groups.append(em.deterministic_sample(songs, args.per_source, args.seed))
    opener = em.AudioOpener(args.data_root, s3, None, args.mom_root is not None, args.tmp_dir)

    def load(song: em.Song) -> dict:
        try:
            with opener.open(song) as path:
                return {"source": song.source, "song": song.song_id, **ffprobe(path)}
        except Exception as exc:  # noqa: BLE001 - reported, the song is left out
            return {"source": song.source, "song": song.song_id, "error": f"{type(exc).__name__}: {exc}"}

    records = list(em.prefetch(load, em.interleave(groups), args.fetch_workers))
    failed = [r for r in records if "error" in r]
    for r in failed[:5]:
        print(f"failed: {r['source']}/{r['song']}: {r['error']}")

    print(f"\n{'source':<11} {'n':>4}  {'codec':<16} {'sample rate':<22} {'channels':<14} "
          f"{'kbps min/med/max':<17} {'top bitrate':<14} top duration")
    suggestions: dict[str, str] = {}
    medians: dict[str, float] = {}
    for source in [em.HUMAN, *args.generators]:
        rows = [r for r in records if r["source"] == source and "error" not in r]
        if not rows:
            print(f"{source:<11} {0:>4}")
            continue
        kbps = [r["kbps"] for r in rows if r["kbps"]]
        lo, med, hi = (np.percentile(kbps, [0, 50, 100]) if kbps else (np.nan,) * 3)
        rate_top, rate_share = top(kbps, -1)  # to the nearest 10 kbps
        dur_top, dur_share = top([r["duration"] for r in rows], 1)
        print(f"{source:<11} {len(rows):>4}  {shares([r['codec'] for r in rows]):<16} "
              f"{shares([r['sample_rate'] for r in rows]):<22} {shares([r['channels'] for r in rows]):<14} "
              f"{lo:>4.0f}/{med:>4.0f}/{hi:>4.0f}     {rate_top} kbps {rate_share:.0%}  "
              f"{dur_top} s {dur_share:.0%}")
        medians[source] = float(med)
        codec, _ = top([r["codec"] for r in rows])
        rate, _ = top([r["sample_rate"] for r in rows])
        channels, _ = top([r["channels"] for r in rows])
        if source != em.HUMAN and codec in ENCODERS and kbps:
            suggestions[source] = f"{codec}:{int(round(med))}k:{rate}:{channels}"

    if suggestions:
        print("\n--codec-check settings that replay each generator's own format on human songs:")
        for source, spec in suggestions.items():
            print(f"  {source:<11} {spec}")
    if medians:
        lowest = min(medians.values())
        # about half the lowest median, so the new quantization swamps every source's
        # original artifacts; never below 48k, where it erases the music too
        below = [b for b in MP3_LADDER if 48 <= b <= lowest / 2]
        pick = below[-1] if below else 48
        print(f"\nlowest median bitrate: {lowest:.0f} kbps ({min(medians, key=medians.get)}). Suggested "
              f"common --codec: mp3:{pick}k (about half of it). Just below the lowest median is not "
              "enough: a lightly re-encoded mp3 keeps its original artifacts.")
    if args.out:
        args.out.write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
