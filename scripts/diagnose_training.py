#!/usr/bin/env python3
"""Is the run learning? Read a train.log instead of squinting at single steps.

train.py prints the loss of *one batch* every --log-every steps. At
--batch-pairs 4 the contrastive term is a 4-way classification, so its
per-step value is nearly bimodal (≈0.05 when all four anchors rank their
positive first, >1.4 when one or two fail) and swings by 2.0 between adjacent
lines whatever the model is doing. Nothing can be read off individual lines.

This script blocks the samples, averages inside each block, and bootstraps the
first-vs-last difference over the sampled steps, so "is it going down" gets an
answer with an interval on it. It also places each term against the baseline
that makes it interpretable:

  contrastive   ln(P*K) is chance for P pairs and K candidates per pair
  recon, swap   ~2.0 is predicting the dataset mean (CLAUDE.md, Decoder)

Example:
    python scripts/diagnose_training.py checkpoints/lc-w100/train.log \\
        --batch-pairs 4
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STEP_RE = re.compile(r"^epoch (\d+) step (\d+)/(\d+)\s+(.*)$")
TERM_RE = re.compile(r"(\w+)=(-?[\d.]+(?:e[-+]?\d+)?)")

# Loss value a trivial predictor reaches, so "high" can be judged (CLAUDE.md).
BASELINES = {
    "recon": ("dataset mean", 2.0),
    "swap": ("dataset mean", 2.0),
}


def parse(path: Path) -> list[dict]:
    """One record per logged step: {'epoch', 'step', term: value, ...}."""
    records = []
    for line in path.read_text(errors="replace").splitlines():
        m = STEP_RE.match(line.strip())
        if not m:
            continue
        record = {"epoch": int(m.group(1)), "step": int(m.group(2))}
        record.update({k: float(v) for k, v in TERM_RE.findall(m.group(4))})
        records.append(record)
    return records


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def bootstrap_diff(early: list[float], late: list[float], n: int = 2000,
                   seed: int = 0) -> tuple[float, float]:
    """95% interval for mean(late) - mean(early), resampling steps."""
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        a = mean([early[rng.randrange(len(early))] for _ in early])
        b = mean([late[rng.randrange(len(late))] for _ in late])
        diffs.append(b - a)
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def block_table(records: list[dict], terms: list[str], blocks: int) -> None:
    size = max(1, len(records) // blocks)
    print(f"| steps | " + " | ".join(terms) + " |")
    print("|" + "---|" * (len(terms) + 1))
    for start in range(0, len(records) - size + 1, size):
        chunk = records[start:start + size]
        cells = [f"{mean([r[t] for r in chunk if t in r]):.3f}"
                 if any(t in r for r in chunk) else "—" for t in terms]
        print(f"| {chunk[0]['step']}–{chunk[-1]['step']} | " + " | ".join(cells) + " |")


def trend(records: list[dict], terms: list[str], frac: float) -> None:
    cut = max(1, int(len(records) * frac))
    print(f"\nFirst {cut} logged steps vs last {cut}, 95% interval over steps:")
    for term in terms:
        early = [r[term] for r in records[:cut] if term in r]
        late = [r[term] for r in records[-cut:] if term in r]
        if not early or not late:
            continue
        low, high = bootstrap_diff(early, late)
        verdict = ("falling" if high < 0 else "rising" if low > 0
                   else "flat (interval spans 0)")
        print(f"  {term:14s} {mean(early):6.3f} -> {mean(late):6.3f}  "
              f"Δ {mean(late) - mean(early):+.3f} [{low:+.3f}, {high:+.3f}]  {verdict}")


def context(records: list[dict], pairs: int, candidates: int) -> None:
    print("\nAgainst the baseline that makes each term interpretable:")
    chance = math.log(pairs * candidates)
    last = records[-max(1, len(records) // 4):]
    if any("contrastive" in r for r in last):
        values = [r["contrastive"] for r in last if "contrastive" in r]
        m = mean(values)
        below = sum(1 for v in values if v < chance) / len(values)
        print(f"  contrastive    {m:.3f} vs chance ln({pairs}*{candidates}) = "
              f"{chance:.3f}; {below:.0%} of steps below chance")
        # loss if every anchor is correct by cosine margin d, temperature 0.1
        best = min(values)
        n_neg = pairs * candidates - candidates
        if best < math.log(1 + n_neg):
            margin = -0.1 * math.log((math.exp(best) - 1) / max(n_neg, 1))
            print(f"                 best step {best:.3f} => every anchor correct "
                  f"with cosine margin ≈ {margin:.2f} (so the encoder has "
                  f"learned real structure; the mean is dragged up by a minority "
                  f"of failing anchors, not by an absence of signal)")
    for term, (label, value) in BASELINES.items():
        if any(term in r for r in last):
            m = mean([r[term] for r in last if term in r])
            gap = (value - m) / value
            flag = ("  <-- barely off the baseline, which is run #1's"
                    " pathology" if gap < 0.08 else "")
            print(f"  {term:14s} {m:.3f} vs {label} ≈ {value:.2f} "
                  f"({gap:.0%} below){flag}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("--blocks", type=int, default=12,
                        help="rows in the block table")
    parser.add_argument("--trend-fraction", type=float, default=0.15,
                        help="fraction of logged steps used at each end")
    parser.add_argument("--batch-pairs", type=int, default=4)
    parser.add_argument("--n-candidates", type=int, default=1)
    parser.add_argument("--from-step", type=int, default=0,
                        help="ignore steps before this (e.g. skip warmup)")
    args = parser.parse_args()

    records = [r for r in parse(args.log) if r["step"] >= args.from_step]
    if not records:
        raise SystemExit(f"no 'epoch N step M/T total=...' lines in {args.log}")
    terms = [k for k in records[-1] if k not in ("epoch", "step")]
    print(f"{len(records)} logged steps, step {records[0]['step']}–"
          f"{records[-1]['step']}, epochs {records[0]['epoch']}–{records[-1]['epoch']}\n")
    block_table(records, terms, args.blocks)
    trend(records, terms, args.trend_fraction)
    context(records, args.batch_pairs, args.n_candidates)


if __name__ == "__main__":
    main()
