#!/usr/bin/env python3
"""Score test50 human songs against every MoM generator with trained probes.

The evaluation half of scripts/train_mert_probe.py. Each --checkpoints entry
is one row of a matrix and each generator one column. Songs are sampled, cut
and pooled exactly as evaluate_mom.py does it (300 per source, 4 windows of
20 s, seed 0 by default), so every cell sits next to the MERT-only flow's
evaluate_mom.py AUROC on the same songs. The MERT pass runs once per codec
condition into an evaluation cache shared by every probe
(track_mom.build_eval_cache, all 25 layers, every generator),
{eval-dir}/eval-{human-split}[-{chain}]; later runs score from it in seconds.

Per-window score: minus the probe's logit, so human scores higher, as in
evaluate_mom.py. Song score: the min over its windows (evaluate_mom.py's
default) and the mean, both reported. Every row scores the same songs: an
evaluation song found in any probe's --mom-cache is dropped from all rows
(none should be: cache_pooled_mom.py --skip keeps them apart).

The matrix. Cell (probe P, generator G): AUROC with a bootstrap 95 % interval
(human songs resampled by work, since the songs of one work are not
independent; synthetic songs by song) and the share of G's songs flagged at
5 % human FPR. A * marks a generator P trained on: that cell measures
separability, on songs P never saw. The other cells measure transfer.

Codec. The evaluation audio goes through the probes' own training chain
(--codec overrides it; "none" = original audio). --codec-check SPEC ... adds
the counterfactual: the same songs again, with SPEC applied before that chain.
Only the codec history differs, so a probe that ignores the codec scores both
versions alike. Reported per probe and source:
  shift AUROC  P(the as-is version scores higher than the re-encoded one).
               0.5: the re-encoding does not move the probe. Toward 1: it
               makes songs look synthetic. Toward 0: human.
  flagged      share of re-encoded songs below the as-is human 5 % FPR
               threshold. For human songs, that is the false-positive rate
               the codec alone induces.
Pass one generator's own settings (scripts/codec_report.py measures them) to
ask whether human music in that generator's mp3 looks like the generator, and
YouTube's (e.g. opus:160k) to ask whether generator songs in YouTube's codec
look human.

Writes {common checkpoint dir}/eval_{human-split}[_{chain}].json.

Example:
    python scripts/evaluate_mert_probe.py --checkpoints checkpoints/mert-probe/*/best.pt \\
        --data-root $DATA --bucket $BUCKET --codec-check mp3:192k opus:160k
"""

from __future__ import annotations

import argparse
import json
import os
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
from src.data.codec import Chain, chain_name, chain_slug, check_encoders, parse_chain  # noqa: E402
from src.data.pooled_cache import PooledCache  # noqa: E402
from src.models.mert_probe import MertProbe  # noqa: E402
from src.training import pick_device  # noqa: E402

AGGREGATES = ("min", "mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                        help="train_mert_probe.py best.pt files, one matrix row each")
    parser.add_argument("--codec", default=None,
                        help="chain the evaluation audio goes through (default: the probes' "
                             "training chain; 'none' = original audio)")
    parser.add_argument("--codec-check", nargs="*", default=[],
                        help="chains applied before --codec to the same songs, e.g. mp3:192k opus:160k")
    parser.add_argument("--eval-dir", type=Path, default=None,
                        help="where the evaluation caches live (default: next to the train cache)")
    parser.add_argument("--human-split", default="test50")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="human audio and manifests (default: the train cache's data_root)")
    parser.add_argument("--bucket", default="songs-and-mert-features")
    parser.add_argument("--region", default=None)
    parser.add_argument("--mom-prefix", default="mom")
    parser.add_argument("--mom-root", type=Path, default=None)
    parser.add_argument("--generators", nargs="+", default=list(em.GENERATORS),
                        help="the matrix columns")
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
    if checkpoint.get("kind") != "mert_probe" or "train_generators" not in checkpoint:
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


def codec_shift(base: dict, check: dict, threshold: float) -> dict:
    """One source's as-is vs re-encoded song scores, matched by song."""
    as_is = dict(zip(base["songs"], base["value"]))
    coded = dict(zip(check["songs"], check["value"]))
    common = sorted(as_is.keys() & coded.keys())
    b = np.array([as_is[s] for s in common])
    c = np.array([coded[s] for s in common])
    return {
        "songs": len(common),
        "shift_auroc": em.auroc(b, c),
        "flagged_at_fpr": float(np.mean(c < threshold)) if common else float("nan"),
        "flagged_as_is": float(np.mean(b < threshold)) if common else float("nan"),
    }


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    mert_config = MertConfig()

    probes: dict[str, tuple[MertProbe, dict, Path]] = {}
    for path in args.checkpoints:
        model, checkpoint = load_probe(path, device)
        name = checkpoint.get("name") or "+".join(checkpoint["train_generators"])
        if name in probes:
            name = str(path.parent)
        probes[name] = (model, checkpoint, path)
    checkpoints = [c for _, c, _ in probes.values()]
    first = checkpoints[0]
    for name, (_, checkpoint, _) in probes.items():
        if checkpoint["model_name"] != mert_config.model_name:
            raise SystemExit(f"{name} was trained on {checkpoint['model_name']}, "
                             f"MertConfig is {mert_config.model_name}")
        if float(checkpoint["window_seconds"]) != float(first["window_seconds"]):
            raise SystemExit("the probes were trained on different window lengths")

    trained = {c.get("codec") for c in checkpoints}
    if args.codec is None:
        if len(trained) > 1:
            raise SystemExit(f"the probes were trained through different codec chains "
                             f"({', '.join(str(t) for t in trained)}); pass --codec to choose one")
        base = parse_chain(next(iter(trained)))
    else:
        base = parse_chain(args.codec)
        strangers = [n for n, (_, c, _) in probes.items() if c.get("codec") != chain_name(base)]
        if strangers:
            print(f"note: scoring {chain_name(base) or 'original audio'}, which "
                  f"{', '.join(strangers)} were not trained on")
    checks = [parse_chain(spec) for spec in args.codec_check]
    if any(not c for c in checks):
        raise SystemExit("--codec-check needs real chains, not 'none'")
    check_encoders(base + tuple(s for c in checks for s in c))

    data_root = args.data_root or Path(first["data_root"])
    eval_dir = args.eval_dir or Path(first["train_cache"]).parent
    settings = {
        "human_split": args.human_split,
        "per_source": args.per_source,
        "windows_per_song": args.windows_per_song,
        "window_seconds": float(first["window_seconds"]),
        "seed": args.seed,
        "generators": list(args.generators),
        "model_name": mert_config.model_name,
    }

    def condition(chain: Chain) -> tuple[Path, list[dict], dict[str, np.ndarray]]:
        """Build or load the evaluation cache under `chain`; every probe's window scores."""
        root = eval_dir / (f"eval-{args.human_split}" + (f"-{chain_slug(chain)}" if chain else ""))
        wanted = dict(settings, codec=chain_name(chain))
        if not tm.cache_matches(root, wanted):
            tm.build_eval_cache(
                root,
                data_root=data_root,
                human_split=args.human_split,
                generators=settings["generators"],
                per_source=args.per_source,
                windows_per_song=args.windows_per_song,
                window_seconds=settings["window_seconds"],
                seed=args.seed,
                bucket=args.bucket,
                region=args.region,
                mom_prefix=args.mom_prefix,
                mom_root=args.mom_root,
                mert_config=mert_config,
                device=device,
                codec=chain,
            )
        cache = tm.EvalCache(root, device)
        scores = {}
        with torch.no_grad():
            for name, (model, _, _) in probes.items():
                logits = torch.cat([model(chunk.float()) for chunk in cache.pooled.split(1024)])
                scores[name] = (-logits).float().cpu().numpy()  # human scores higher
        rows = cache.rows
        del cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return root, rows, scores

    exclude: set = set()
    for mom_cache in sorted({Path(c["mom_cache"]) for c in checkpoints}):
        if (mom_cache / "meta.json").exists():
            exclude |= {(r["source"], r["song"]) for r in PooledCache(mom_cache).rows()}
        else:
            print(f"warning: {mom_cache} not found; cannot check the evaluation songs for overlap "
                  "with training (cache_pooled_mom.py --skip keeps them apart by construction)")

    root, rows, scores = condition(base)
    dropped = len({(r["source"], r["song"]) for r in rows} & exclude)
    tables = {name: song_scores(rows, scores[name], exclude) for name in probes}
    some = next(iter(tables.values()))
    if em.HUMAN not in some:
        raise SystemExit(f"{root} holds no {em.HUMAN} songs")
    columns = [g for g in args.generators if g in some]
    missing = [g for g in args.generators if g not in some]
    if missing:
        print(f"warning: {root} holds no songs of {', '.join(missing)}")

    works = work_of(data_root, args.human_split)
    human_songs = some[em.HUMAN]["songs"]
    groups = [works.get(s, s) for s in human_songs] if works else human_songs
    unit = "work" if works else "song (no manifest found; intervals are too narrow)"
    rng = np.random.default_rng(0)

    matrix: dict[str, dict] = {}
    for name, (_, checkpoint, _) in probes.items():
        table, seen = tables[name], set(checkpoint["train_generators"])
        matrix[name] = {}
        for g in columns:
            cell: dict = {"seen": g in seen}
            for agg in AGGREGATES:
                human, synthetic = table[em.HUMAN][agg], table[g][agg]
                threshold = np.quantile(human, args.fpr)
                cell[agg] = {
                    "auroc": em.auroc(human, synthetic),
                    "ci95": bootstrap_interval(human, groups, synthetic, args.bootstrap, rng),
                    "flagged_at_fpr": float(np.mean(synthetic < threshold)),
                }
            matrix[name][g] = cell

    width = max(len(n) for n in probes) + 2
    print(f"\nhuman = {args.human_split} ({len(human_songs)} songs, {len(set(groups))} works); "
          f"audio: {chain_name(base) or 'original'}; * = generator the probe trained on "
          "(separability on unseen songs), others = transfer")
    summary: dict = {}
    for agg in AGGREGATES:
        print(f"\nAUROC [95% CI], song score = {agg} of windows")
        print(f"{'probe':<{width}}" + "".join(f"{g:>22}" for g in columns))
        for name, row in matrix.items():
            cells = []
            for g in columns:
                e = row[g][agg]
                lo, hi = e["ci95"]
                cells.append(f"{e['auroc']:.3f} [{lo:.2f},{hi:.2f}]{'*' if row[g]['seen'] else ' '}")
            print(f"{name:<{width}}" + "".join(f"{c:>22}" for c in cells))
        seen_cells = [r[g][agg]["auroc"] for r in matrix.values() for g in columns if r[g]["seen"]]
        other_cells = [r[g][agg]["auroc"] for r in matrix.values() for g in columns if not r[g]["seen"]]
        summary[agg] = {
            "seen_mean_auroc": float(np.mean(seen_cells)) if seen_cells else None,
            "unseen_mean_auroc": float(np.mean(other_cells)) if other_cells else None,
        }
        if seen_cells and other_cells:
            print(f"mean AUROC: trained-on cells {summary[agg]['seen_mean_auroc']:.3f}, "
                  f"transfer cells {summary[agg]['unseen_mean_auroc']:.3f}")
    print(f"\nshare of synthetic songs flagged at {args.fpr:g} human FPR (min / mean)")
    print(f"{'probe':<{width}}" + "".join(f"{g:>22}" for g in columns))
    for name, row in matrix.items():
        cells = [f"{row[g]['min']['flagged_at_fpr']:.2f} / {row[g]['mean']['flagged_at_fpr']:.2f}"
                 f"{'*' if row[g]['seen'] else ' '}" for g in columns]
        print(f"{name:<{width}}" + "".join(f"{c:>22}" for c in cells))

    codec_checks: dict = {}
    sources = [em.HUMAN, *columns]
    for spec, check in zip(args.codec_check, checks):
        chain = check + base
        root_c, rows_c, scores_c = condition(chain)
        entry: dict = {"prepended": chain_name(check), "chain": chain_name(chain),
                       "eval_cache": str(root_c), "probes": {}}
        for name in probes:
            table_b, table_c = tables[name], song_scores(rows_c, scores_c[name], exclude)
            entry["probes"][name] = {}
            for source in sources:
                if source not in table_c:
                    continue
                entry["probes"][name][source] = {
                    agg: codec_shift(
                        {"songs": table_b[source]["songs"], "value": table_b[source][agg]},
                        {"songs": table_c[source]["songs"], "value": table_c[source][agg]},
                        float(np.quantile(table_b[em.HUMAN][agg], args.fpr)),
                    )
                    for agg in AGGREGATES
                }
        codec_checks[spec] = entry
        print(f"\ncodec check: {chain_name(check)} before "
              f"{chain_name(base) or 'nothing (original audio)'}, same songs; song score = min")
        print("cell = shift AUROC / flagged re-encoded (flagged as-is). Shift 0.5: the probe ignores "
              "the re-encoding; above 0.5: re-encoded songs look more synthetic")
        print(f"{'probe':<{width}}" + "".join(f"{s:>22}" for s in sources))
        for name, per_source in entry["probes"].items():
            cells = []
            for source in sources:
                r = per_source.get(source, {}).get("min")
                cells.append("—" if r is None else
                             f"{r['shift_auroc']:.2f} / {r['flagged_at_fpr']:.0%} ({r['flagged_as_is']:.0%})")
            print(f"{name:<{width}}" + "".join(f"{c:>22}" for c in cells))

    if dropped:
        print(f"\n{dropped} evaluation songs were in a training cache and were left out of every table")

    report = {
        "probes": {
            name: {"checkpoint": str(path), "epoch": checkpoint["epoch"],
                   "train_generators": checkpoint["train_generators"],
                   "trained_codec": checkpoint.get("codec")}
            for name, (_, checkpoint, path) in probes.items()
        },
        "human_split": args.human_split,
        "codec": chain_name(base),
        "eval_cache": str(root),
        "songs": {s: len(e["songs"]) for s, e in some.items()},
        "human_works": len(set(groups)),
        "eval_songs_dropped_as_trained_on": dropped,
        "bootstrap": {"reps": args.bootstrap, "human_unit": unit},
        "fpr": args.fpr,
        "matrix": matrix,
        "summary": summary,
        "codec_checks": codec_checks,
    }
    common = Path(os.path.commonpath([str(p.resolve().parent) for p in args.checkpoints]))
    out = args.out or common / (f"eval_{args.human_split}" + (f"_{chain_slug(base)}" if base else "") + ".json")
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
