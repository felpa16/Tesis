#!/usr/bin/env python3
"""Does the augmentation preserve content? Measure it, don't assume it.

src/data/augment.py only changes timbre and production, so an augmented window
must still align with the original as well as the original aligns with itself.
This script checks that with the project's own content metric — beat-synchronous
chroma, OTI, Smith-Waterman — so a transform that quietly destroys melody or
harmony shows up as a low score here instead of as a silently worse encoder.

Read two columns:

  score   the alignment score of augmented-vs-original, against the clean
          self-alignment in the first row. A transform that preserves content
          scores close to it; anything near the 0.2 keep threshold is
          destroying the content the contrastive loss is supposed to hold fixed.
  oti     the transposition the aligner had to apply. These transforms do not
          shift pitch, so anything but 0 means the spectrum change was severe
          enough to look like a key change.

Example:
    python scripts/check_augment.py --data-root $DATA --split val --n 12
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from align_covers import (  # noqa: E402
    beat_sync_chroma,
    optimal_transposition_index,
    smith_waterman_path,
)
from shs100k_meta import DEFAULT_DATA_ROOT, existing_audio  # noqa: E402
from src.config import AugmentConfig  # noqa: E402
from src.data.augment import augment_waveform  # noqa: E402
from src.data.windows import WindowConfig, decode_window  # noqa: E402

CHROMA_SR = 22050

# Each row forces one transform by setting probability 1 and disabling the rest.
OFF = dict(
    gain_db=0.0, tilt_db=0.0, peak_db=0.0, saturation=1.0, reverb_seconds=0.0,
    noise_snr_db=(120.0, 120.0), lowpass_hz=(1e9, 1e9),
)
TRANSFORMS: dict[str, dict] = {
    "gain": {**OFF, "gain_db": 6.0},
    "eq tilt": {**OFF, "tilt_db": 6.0},
    "eq peak": {**OFF, "peak_db": 8.0},
    "lowpass": {**OFF, "lowpass_hz": (6000.0, 6000.0)},
    "noise": {**OFF, "noise_snr_db": (20.0, 20.0)},
    "saturation": {**OFF, "saturation": 3.0},
    "reverb": {**OFF, "reverb_seconds": 0.25},
    "all (defaults)": {},
}


def score_against(clean: np.ndarray, other: np.ndarray, quantile: float, gap: float):
    oti = optimal_transposition_index(clean, other)
    sim = clean.T @ np.roll(other, oti, axis=0)
    _, score = smith_waterman_path(sim, quantile, gap)
    return score, oti


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--n", type=int, default=12, help="windows to average over")
    parser.add_argument("--window-seconds", type=float, default=20.0)
    parser.add_argument("--match-quantile", type=float, default=0.8)
    parser.add_argument("--gap", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    audio = existing_audio(args.data_root, args.split)
    if not audio:
        raise SystemExit(f"no audio under {args.data_root}/audio/{args.split}")
    rng = random.Random(args.seed)
    keys = sorted(audio)[: args.n * 4]
    rng.shuffle(keys)

    # chroma wants 22.05 kHz; MERT's 24 kHz would work too but this matches
    # exactly what align_covers.py measured the 0.2 threshold on.
    window = WindowConfig(window_seconds=args.window_seconds, sample_rate=CHROMA_SR)
    results: dict[str, list[tuple[float, int]]] = {name: [] for name in TRANSFORMS}
    results["(clean self)"] = []

    used = 0
    for key in keys:
        if used >= args.n:
            break
        try:
            wave = decode_window(audio[key], 30.0, window)
            clean, _ = beat_sync_chroma(wave.numpy(), CHROMA_SR)
        except Exception as exc:
            print(f"  skipped {key}: {type(exc).__name__}: {exc}")
            continue
        if clean.shape[1] < 20:
            continue
        used += 1
        results["(clean self)"].append(
            score_against(clean, clean, args.match_quantile, args.gap)
        )
        for name, overrides in TRANSFORMS.items():
            config = AugmentConfig(enabled=True, probability=1.0, **overrides)
            augmented = augment_waveform(
                wave, CHROMA_SR, config, random.Random(args.seed + used)
            )
            try:
                chroma, _ = beat_sync_chroma(augmented.numpy(), CHROMA_SR)
            except Exception as exc:
                print(f"  {name} failed on {key}: {type(exc).__name__}: {exc}")
                continue
            results[name].append(
                score_against(clean, chroma, args.match_quantile, args.gap)
            )

    if not used:
        raise SystemExit("no usable windows")
    print(f"\n{used} windows from {args.split}, {args.window_seconds:.0f} s each\n")
    print("| transform | score | vs clean self | oti != 0 |")
    print("|---|---|---|---|")
    reference = float(np.mean([s for s, _ in results["(clean self)"]]))
    for name in ["(clean self)", *TRANSFORMS]:
        rows = results[name]
        if not rows:
            continue
        score = float(np.mean([s for s, _ in rows]))
        shifted = sum(1 for _, o in rows if o != 0)
        print(f"| {name} | {score:.3f} | {score / reference:5.1%} | {shifted}/{len(rows)} |")
    print(
        "\nA transform well below the clean self-score, or one that needs a "
        "transposition,\nis changing content rather than style and should be "
        "turned down in AugmentConfig."
    )


if __name__ == "__main__":
    main()
