#!/usr/bin/env python3
"""Evaluate a representation checkpoint on a held-out pair split (e.g. test50).

Runs exactly the validation pass of scripts/train.py, same windows (seeded),
same losses, same retrieval metrics, on another split:

  <split>/total, recon, contrastive, swap, cycle, decorrelation
  <split>/content_recall@1       nearest B window is the query's own pair
  <split>/content_work_recall@1  nearest B window is a cover of the same work
  <split>/content_work_map       mAP of ranking same-work B windows first

The retrieval metrics come with 95% bootstrap intervals that resample *works*.
Works are the independent unit, and on a 50-work test set the interval is
what tells a real difference between two runs from noise.

Writes {checkpoint dir}/eval_{split}.json, which learning_curve.py reads. The
file keeps the per-query results too. The window sampling is seeded, so runs
evaluated with the same seed are scored on identical windows, and
learning_curve.py can compare them pairwise, query by query.

Example:
    python scripts/evaluate_encoder.py --checkpoint checkpoints/lc-w25/best.pt \\
        --split test50 --data-root $DATA
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from shs100k_meta import DEFAULT_DATA_ROOT  # noqa: E402
from src.config import config_from_dict  # noqa: E402
from src.metrics import bootstrap_over_works  # noqa: E402
from src.models import DisentanglementModel, MertExtractor  # noqa: E402
from src.training import pick_device  # noqa: E402
from train import build_val_loader, validate  # noqa: E402

RETRIEVAL = {
    "pair_hit": "content_recall@1",
    "work_hit": "content_work_recall@1",
    "work_ap": "content_work_map",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test50")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-batches", type=int, default=0, help="cap on batches; 0 = full split"
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out", type=Path, help="default: next to the checkpoint")
    args = parser.parse_args()

    device = pick_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = config_from_dict(checkpoint["config"])
    config.data.data_root = str(args.data_root)
    config.data.val_split = args.split
    config.data.val_max_batches = args.max_batches

    model = DisentanglementModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    mert = MertExtractor(config.mert).to(device)
    loader = build_val_loader(config, args.data_root)
    if loader is None:
        raise SystemExit(f"no usable aligned pairs in split {args.split!r}")
    print(
        f"{args.checkpoint} (step {checkpoint['step']}) on {args.split}: "
        f"{len(loader.dataset)} pairs"
    )

    metrics, queries = validate(
        mert, model, config, loader, device, prefix=args.split, return_queries=True
    )
    intervals = {
        f"{args.split}/{name}": bootstrap_over_works(
            queries[key], queries["work"], n=args.bootstrap
        )
        for key, name in RETRIEVAL.items()
    }
    for name, value in metrics.items():
        ci = intervals.get(name)
        note = f"   95% CI over works [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
        print(f"  {name:36s} {value:.4f}{note}")

    result = {
        "checkpoint": str(args.checkpoint),
        "step": checkpoint["step"],
        "split": args.split,
        "seed": config.seed,
        "n_pairs": len(queries["work"]),
        "n_works": len(set(queries["work"].tolist())),
        "metrics": metrics,
        "ci95": {name: list(ci) for name, ci in intervals.items()},
        # per-query results: runs evaluated with the same seed see the same
        # windows, so learning_curve.py can bootstrap the paired difference
        "queries": {key: queries[key].tolist() for key in ("work", *RETRIEVAL)},
    }
    out = args.out or args.checkpoint.parent / f"eval_{args.split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
