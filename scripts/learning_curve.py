#!/usr/bin/env python3
"""Summarize learning-curve runs into one table (no GPU, no data needed).

Each argument is a run's checkpoint directory, as written by train.py and
evaluate_encoder.py:

    run_info.json       what the run trained on (works, pairs, tracks, budget)
    best_metrics.json   the validation point that produced best.pt
    val_metrics.jsonl   every validation point, with train losses in between
    eval_{split}.json   best.pt on the test split, with bootstrap intervals

Per run it reports the training set size, the best checkpoint's validation
numbers, the train/val contrastive gap there (the overfitting signal), and
the test numbers with their 95% intervals over works. Read the curve from the
test columns. The paired differences printed below the table say whether a
step up the curve is real.

Example:
    python scripts/learning_curve.py checkpoints/lc-w25 checkpoints/lc-w50 \\
        checkpoints/lc-w100 --split test50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import bootstrap_over_works  # noqa: E402


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_trajectory(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def with_ci(result: dict | None, name: str) -> str:
    if result is None:
        return "—"
    value = result["metrics"].get(name)
    ci = result["ci95"].get(name)
    if ci is None:
        return fmt(value)
    return f"{fmt(value)} [{fmt(ci[0])}, {fmt(ci[1])}]"


def paired_differences(runs: list[Path], split: str) -> list[str]:
    """Gap between consecutive runs on the same test queries, with 95% CI.

    Much tighter than comparing the runs' separate intervals, because the
    per-query difficulty that dominates each interval cancels out of the
    difference. Only valid when both runs were scored on identical windows
    (same split, seed and queries), which is checked here.
    """
    lines = []
    results = [(run, load_json(run / f"eval_{split}.json")) for run in runs]
    for (run_a, a), (run_b, b) in zip(results, results[1:]):
        if a is None or b is None or "queries" not in a or "queries" not in b:
            continue
        if a["seed"] != b["seed"] or a["queries"]["work"] != b["queries"]["work"]:
            lines.append(
                f"- {run_b.name} vs {run_a.name}: not scored on the same queries "
                f"(different seed or manifest); no paired comparison"
            )
            continue
        works = a["queries"]["work"]
        for key, label in (("work_ap", "work mAP"), ("work_hit", "work R@1")):
            diff = [y - x for x, y in zip(a["queries"][key], b["queries"][key])]
            low, high = bootstrap_over_works(diff, works)
            mean = sum(diff) / len(diff)
            verdict = (
                "real gain" if low > 0 else "real loss" if high < 0 else "within noise"
            )
            lines.append(
                f"- {run_b.name} − {run_a.name}: Δ {split} {label} = {mean:+.3f} "
                f"[{low:+.3f}, {high:+.3f}] paired over {len(diff)} queries / "
                f"{len(set(works))} works -> {verdict}"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", type=Path, nargs="+", help="checkpoint directories")
    parser.add_argument("--split", default="test50", help="test split evaluated")
    parser.add_argument("--val-prefix", default="val")
    parser.add_argument("--out", type=Path, help="also write the table here")
    args = parser.parse_args()

    v, t = args.val_prefix, args.split
    header = (
        "| run | works | pairs | best step | val total | val contr. | train contr. "
        f"| val work mAP | {t} work mAP [95% CI] | {t} work R@1 [95% CI] "
        f"| {t} pair R@1 | {t} recon | {t} contr. |"
    )
    lines = [header, "|" + "---|" * (header.count("|") - 1)]
    notes = []
    for run in args.runs:
        info = load_json(run / "run_info.json") or {}
        best = load_json(run / "best_metrics.json") or {}
        test = load_json(run / f"eval_{args.split}.json")
        trajectory = load_trajectory(run / "val_metrics.jsonl")
        lines.append(
            "| "
            + " | ".join(
                [
                    run.name,
                    str(info.get("n_pair_works", "—")),
                    str(info.get("n_pairs", "—")),
                    str(best.get("step", "—")),
                    fmt(best.get(f"{v}/total")),
                    fmt(best.get(f"{v}/contrastive")),
                    fmt(best.get("train/contrastive")),
                    fmt(best.get(f"{v}/content_work_map")),
                    with_ci(test, f"{t}/content_work_map"),
                    with_ci(test, f"{t}/content_work_recall@1"),
                    fmt(test["metrics"].get(f"{t}/content_recall@1") if test else None),
                    fmt(test["metrics"].get(f"{t}/recon") if test else None),
                    fmt(test["metrics"].get(f"{t}/contrastive") if test else None),
                ]
            )
            + " |"
        )
        if trajectory:
            peak = max(trajectory, key=lambda r: r.get(f"{v}/content_work_map", -1.0))
            final = trajectory[-1]
            notes.append(
                f"- {run.name}: val work mAP peaked at {fmt(peak.get(f'{v}/content_work_map'))} "
                f"(step {peak['step']}), ended at {fmt(final.get(f'{v}/content_work_map'))} "
                f"(step {final['step']}); {len(trajectory)} validation points"
            )
        if test is None:
            notes.append(f"- {run.name}: no eval_{args.split}.json yet (run evaluate_encoder.py)")

    notes += paired_differences(args.runs, args.split)
    text = "\n".join(lines + [""] + notes) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
