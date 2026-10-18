#!/usr/bin/env python3
"""Score test50 human songs against the left-out MoM generator with a trained probe.

The evaluation half of scripts/train_mert_probe.py. Songs are sampled, cut and
pooled exactly as evaluate_mom.py does it (300 per source, 4 windows of 20 s,
seed 0 by default), so the headline number sits next to the MERT-only flow's
evaluate_mom.py AUROC on the same songs. The MERT pass runs once into an
evaluation cache shared by every probe (track_mom.build_eval_cache, all 25
layers, every generator so one cache serves every --leave-out), default
{train-cache}/../eval-test50; later runs score from it in seconds.

Per-window score: minus the probe's logit, so human scores higher, as in
evaluate_mom.py. Song score: the min over its windows (evaluate_mom.py's
default) and the mean, both reported.

Reported per source:
  * the left-out generator, which the probe never saw: the headline
  * each training generator, on songs the probe never trained on (any eval song
    found in --mom-cache is dropped). The gap between these and the headline is
    how much of the probe's skill is specific to the generators it saw.
For each: AUROC with a bootstrap 95 % interval (human songs resampled by work,
since the songs of one work are not independent, synthetic songs by song), and
the share of synthetic songs flagged at 5 % human FPR.

Writes {checkpoint-dir}/eval_{human-split}.json.

Example:
    python scripts/evaluate_mert_probe.py --checkpoint checkpoints/mert-probe-no-udio/best.pt \\
        --data-root $DATA --bucket $BUCKET
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

import evaluate_mom as em  # noqa: E402
import track_mom as tm  # noqa: E402
from src.config import MertConfig  # noqa: E402
from src.data.pooled_cache import PooledCache  # noqa: E402
from src.models.mert_probe import MertProbe  # noqa: E402
from src.training import pick_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="train_mert_probe.py best.pt")
    parser.add_argument("--eval-cache", type=Path, default=None,
                        help="default: {train-cache}/../eval-{human-split}")
    parser.add_argument("--human-split", default="test50")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="human audio and manifests (default: the train cache's data_root)")
    parser.add_argument("--bucket", default="songs-and-mert-features")
    parser.add_argument("--region", default=None)
    parser.add_argument("--mom-prefix", default="mom")
    parser.add_argument("--mom-root", type=Path, default=None)
    parser.add_argument("--generators", nargs="+", default=list(em.GENERATORS),
                        help="generators in the evaluation cache (must include the left-out one)")
    parser.add_argument("--per-source", type=int, default=300)
    parser.add_argument("--windows-per-song", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="song sampling seed")
    parser.add_argument("--fpr", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=1000, help="resamples for the intervals")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_probe(path: Path, device: torch.device) -> tuple[MertProbe, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("kind") != "mert_probe":
        raise SystemExit(f"{path} is not a train_mert_probe.py checkpoint")
    model = MertProbe(checkpoint["n_layers"], checkpoint["dim"])
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    model.eval()
    return model.to(device), checkpoint


def work_of(data_root: Path, split: str) -> dict[str, int] | None:
    """Track key -> work id from the split's manifest, or None without one."""
    from src.data.manifest import read_tracks

    try:
        return {t.key: t.song_id for t in read_tracks(data_root, split)}
    except FileNotFoundError:
        return None


def song_scores(rows: list[dict], scores: np.ndarray, exclude: set) -> dict[str, dict]:
    """source -> {"songs": [...], "min": (n,), "mean": (n,)} over each song's windows."""
    windows: dict[tuple[str, str], list[float]] = {}
    for row, score in zip(rows, scores):
        key = (row["source"], row["song"])
        if key not in exclude:
            windows.setdefault(key, []).append(float(score))
    out: dict[str, dict] = {}
    for (source, song), values in windows.items():
        entry = out.setdefault(source, {"songs": [], "min": [], "mean": []})
        entry["songs"].append(song)
        entry["min"].append(min(values))
        entry["mean"].append(float(np.mean(values)))
    return {
        s: {"songs": e["songs"], "min": np.array(e["min"]), "mean": np.array(e["mean"])}
        for s, e in out.items()
    }


def bootstrap_interval(
    human: np.ndarray, groups: list, synthetic: np.ndarray, reps: int, rng: np.random.Generator
) -> list[float]:
    """95 % percentile interval of the AUROC; human resampled by group, synthetic by song."""
    _, inverse = np.unique(np.array(groups, dtype=str), return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]
    values = []
    for _ in range(reps):
        picked = rng.integers(0, len(members), len(members))
        h = human[np.concatenate([members[k] for k in picked])]
        s = synthetic[rng.integers(0, len(synthetic), len(synthetic))]
        values.append(em.auroc(h, s))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    model, checkpoint = load_probe(args.checkpoint, device)
    leave_out, seen = checkpoint["leave_out"], list(checkpoint["train_generators"])
    if leave_out not in args.generators:
        raise SystemExit(f"--generators must include the left-out generator {leave_out}")
    mert_config = MertConfig()
    if mert_config.model_name != checkpoint["model_name"]:
        raise SystemExit(f"probe was trained on {checkpoint['model_name']}, "
                         f"MertConfig is {mert_config.model_name}")
    data_root = args.data_root or Path(checkpoint["data_root"])

    root = args.eval_cache or Path(checkpoint["train_cache"]).parent / f"eval-{args.human_split}"
    settings = {
        "human_split": args.human_split,
        "per_source": args.per_source,
        "windows_per_song": args.windows_per_song,
        "window_seconds": float(checkpoint["window_seconds"]),
        "seed": args.seed,
        "generators": list(args.generators),
        "model_name": checkpoint["model_name"],
    }
    if not tm.cache_matches(root, settings):
        tm.build_eval_cache(
            root,
            data_root=data_root,
            human_split=settings["human_split"],
            generators=settings["generators"],
            per_source=settings["per_source"],
            windows_per_song=settings["windows_per_song"],
            window_seconds=settings["window_seconds"],
            seed=settings["seed"],
            bucket=args.bucket,
            region=args.region,
            mom_prefix=args.mom_prefix,
            mom_root=args.mom_root,
            mert_config=mert_config,
            device=device,
        )
    cache = tm.EvalCache(root, device)

    with torch.no_grad():
        logits = torch.cat([model(chunk.float()) for chunk in cache.pooled.split(1024)])
    scores = (-logits).float().cpu().numpy()  # human scores higher, as in evaluate_mom.py

    exclude: set = set()
    mom_cache = Path(checkpoint["mom_cache"])
    if (mom_cache / "meta.json").exists():
        exclude = {(r["source"], r["song"]) for r in PooledCache(mom_cache).rows() if r["source"] in seen}
    else:
        print(f"warning: {mom_cache} not found; cannot check the training generators' eval songs "
              "for overlap with training (cache_pooled_mom.py --skip keeps them apart by construction)")
    eval_songs = {(r["source"], r["song"]) for r in cache.rows}
    dropped = len(eval_songs & exclude)
    by_source = song_scores(cache.rows, scores, exclude)
    if em.HUMAN not in by_source or leave_out not in by_source:
        raise SystemExit(f"{root} holds no {em.HUMAN} or no {leave_out} songs")

    works = work_of(data_root, args.human_split)
    human_songs = by_source[em.HUMAN]["songs"]
    groups = [works.get(s, s) for s in human_songs] if works else human_songs
    unit = "work" if works else "song (no manifest found; intervals are too narrow)"
    rng = np.random.default_rng(0)

    report: dict = {
        "checkpoint": str(args.checkpoint),
        "epoch": checkpoint["epoch"],
        "leave_out": leave_out,
        "train_generators": seen,
        "human_split": args.human_split,
        "eval_cache": str(root),
        "songs": {s: len(e["songs"]) for s, e in by_source.items()},
        "human_works": len(set(groups)),
        "eval_songs_dropped_as_trained_on": dropped,
        "bootstrap": {"reps": args.bootstrap, "human_unit": unit},
        "fpr": args.fpr,
        "sources": {},
    }
    for source in [leave_out, *seen]:
        if source not in by_source:
            continue
        entry: dict = {"role": "left out" if source == leave_out else "seen in training (unseen songs)"}
        for agg in ("min", "mean"):
            human, synthetic = by_source[em.HUMAN][agg], by_source[source][agg]
            threshold = np.quantile(human, args.fpr)
            entry[agg] = {
                "auroc": em.auroc(human, synthetic),
                "ci95": bootstrap_interval(human, groups, synthetic, args.bootstrap, rng),
                "flagged_at_fpr": float(np.mean(synthetic < threshold)),
            }
        report["sources"][source] = entry

    print(f"\nprobe without {leave_out} (epoch {checkpoint['epoch']}), human = {args.human_split} "
          f"({len(human_songs)} songs, {len(set(groups))} works); song score = min / mean of windows")
    print(f"{'source':<12} {'role':<9} {'songs':>5}  {'AUROC min [95% CI]':<24} "
          f"{'AUROC mean [95% CI]':<24} flagged@{args.fpr:g} (min / mean)")
    for source, entry in report["sources"].items():
        cells = []
        for agg in ("min", "mean"):
            lo, hi = entry[agg]["ci95"]
            cells.append(f"{entry[agg]['auroc']:.3f} [{lo:.3f}, {hi:.3f}]")
        role = "LEFT OUT" if source == leave_out else "seen"
        print(f"{source:<12} {role:<9} {report['songs'][source]:>5}  {cells[0]:<24} {cells[1]:<24} "
              f"{entry['min']['flagged_at_fpr']:.2f} / {entry['mean']['flagged_at_fpr']:.2f}")
    seen_scores = [report["sources"][g]["mean"]["auroc"] for g in seen if g in report["sources"]]
    if seen_scores:
        gap = float(np.mean(seen_scores)) - report["sources"][leave_out]["mean"]["auroc"]
        report["seen_minus_left_out_mean_auroc"] = gap
        print(f"seen-generator mean AUROC minus left-out AUROC (mean aggregation): {gap:+.3f}")
    if dropped:
        print(f"{dropped} evaluation songs were in the training cache and were left out of the table")

    out = args.out or args.checkpoint.parent / f"eval_{args.human_split}.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
