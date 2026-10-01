#!/usr/bin/env python3
"""Rebuild a lost run_info.json from a checkpoint, without retraining.

Every field is recoverable. The checkpoint carries the full TrainConfig, so the
splits, batch sizes, queue size, augmentation flag, layer-weights path, freeze
flag, seed and step budget all come straight out of it; the pair and track
counts come from re-reading the manifests the config names; and the start time
is the name of the run's TensorBoard directory, which train.py creates as
time.strftime("%Y%m%d-%H%M%S").

The dict is built by train.run_info_dict, the same function the training loop
uses, so a rebuilt file cannot disagree with a written one about its fields.

Only re-reads manifests, never audio, so it runs in a second on any machine
that has $DATA — no GPU and no MERT.

Example:
    python scripts/rebuild_run_info.py --checkpoint-dir checkpoints/qa-w25 \\
        --data-root $DATA
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
from src.data.manifest import read_pairs, read_tracks  # noqa: E402
from train import run_info_dict  # noqa: E402


def started_from_logs(log_dir: Path) -> str | None:
    """train.py names each run's TensorBoard dir with its start time."""
    if not log_dir.is_dir():
        return None
    stamps = sorted(d.name for d in log_dir.iterdir() if d.is_dir())
    for stamp in stamps:  # "20260928-034037"
        if len(stamp) == 15 and stamp[8] == "-" and stamp.replace("-", "").isdigit():
            return (
                f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]} "
                f"{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}"
            )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="which checkpoint to read the config from (default: last.pt, "
        "else best.pt)",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="the run's TensorBoard directory, for the start time "
        "(default: runs/<checkpoint dir name>)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args()

    path = args.checkpoint
    if path is None:
        for name in ("last.pt", "best.pt"):
            if (args.checkpoint_dir / name).exists():
                path = args.checkpoint_dir / name
                break
    if path is None or not path.exists():
        raise SystemExit(f"no checkpoint found in {args.checkpoint_dir}")

    out = args.checkpoint_dir / "run_info.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists; pass --force to overwrite")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = config_from_dict(checkpoint["config"])
    split = config.data.train_split
    pairs = read_pairs(args.data_root, split)
    tracks = read_tracks(args.data_root, split)

    # The budget is config.max_steps when it was set, which it is for every run
    # in this experiment; otherwise reconstruct it the way train.py does.
    total_steps = config.max_steps or config.epochs * (
        len(pairs) // max(config.data.batch_pairs, 1)
    )
    log_dir = args.log_dir or Path(config.log_dir or "runs") / args.checkpoint_dir.name
    started = started_from_logs(log_dir)

    info = run_info_dict(config, pairs, tracks, total_steps, started)
    if started is None:
        info["started"] = "unknown (rebuilt from checkpoint)"
        print(f"note: no timestamped run directory under {log_dir}")
    info["rebuilt_from"] = str(path)
    info["rebuilt_at_step"] = int(checkpoint.get("step", -1))

    with open(out, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    print(f"wrote {out}")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
