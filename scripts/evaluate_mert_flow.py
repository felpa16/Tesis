#!/usr/bin/env python3
"""Score MERT-only flows on test50 vs. MoM with the codec taken out of the comparison.

Why. The flow never trains on synthetic music, but a human-only density can
still find mp3 atypical. Every MoM file is mp3, each generator with its own
settings; every SHS100K file is YouTube opus or AAC. If the flow's density
drops on mp3 artifacts, part of its AUROC (0.856 pooled, experiments.md
section 11) measures the codec and not generation. The supervised probes were
cleared of this with a common codec and a counterfactual (section 13). This
script runs the same two controls on the flow (section 15).

Conditions. Songs are sampled, cut and pooled exactly as evaluate_mom.py does
it (300 per source, 4 windows of 20 s, seed 0 by default). Each condition is
one cached MERT pass over them (track_mom.build_eval_cache, all 25 layers,
{eval-dir}/eval-{human-split}[-{chain}]). The caches are the ones
evaluate_mert_probe.py builds, so a condition the probes already scored costs
no MERT pass.

  training chain   the audio as the flow was trained on it: original audio for
                   a flow trained on cache_pooled_mert.py's default cache, or
                   the chain of a --codec train cache (train_mert_flow.py
                   records it). For an original-audio flow this reproduces
                   evaluate_mom.py's numbers, up to fp16 storage.
  --codec CHAIN    every window of both classes re-encoded through CHAIN
                   before MERT. The last codec is then the same on both sides,
                   so its settings carry no label.

Reported per flow and condition, for the flow and its Gaussian baseline, song
score = min and mean over windows:
  * AUROC per generator and pooled ("all"), with a bootstrap 95 % interval
    (human songs resampled by work, since the songs of one work are not
    independent; synthetic songs by song);
  * the share of synthetic songs flagged at 5 % human FPR;
  * for every condition other than the flow's training chain, the change in
    AUROC against it, with a paired bootstrap interval (the same resampled
    songs on both sides). Every condition scores the same songs: a song that
    failed to load under any condition is left out of all of them.

--codec-check SPEC ... is the counterfactual, under every condition: the same
songs again with SPEC applied first. The music is identical; only the codec
history differs. Reported per flow and source:
  shift AUROC  P(the as-is version scores higher than the re-encoded one).
               0.5: the re-encoding does not move the flow. Toward 1: it
               makes songs look synthetic. Toward 0: human.
  flagged      share of re-encoded songs below the as-is human 5 % FPR
               threshold. For human songs, that is the false-positive rate the
               codec alone induces.
Pass each generator's own settings (scripts/codec_report.py measures them) to
ask whether human music in that generator's mp3 looks synthetic, and YouTube's
(e.g. opus:137k:48000:2) to ask whether generator songs in YouTube's codec look
human. Under the training chain of an original-audio flow this measures the
headline number's codec exposure directly. Under --codec it measures what
survives the common chain (traces of the earlier codec under the new one).

Reading.
  * Checks clean under the training chain (human shift near 0.5 and flagged
    near 5 %; generator songs unmoved by YouTube's spec): the flow does not
    read the codec, and the headline stands. A drop under --codec is then
    music the common chain destroys, or audio the flow was never trained on,
    not codec bias removed.
  * Checks not clean under the training chain: the headline includes codec
    bias. The --codec AUROC is the codec-free number, provided the checks
    under --codec are clean.
  * The cleanest number comes from a flow trained on a --codec cache and scored
    under that chain: human training audio, human test audio and synthetic test
    audio then share the last codec, and the flow is not scored off its own
    training distribution. Compare it with a flow trained on the same tracks as
    original audio (experiments.md section 15, "Run order").

Writes {common checkpoint dir}/eval_codec_{human-split}.json (or --out).

Example:
    python scripts/evaluate_mert_flow.py \\
        --checkpoints checkpoints/mert-flow-unfrozen/mert_flow_last.pt \\
                      checkpoints/mert-flow-frozen/mert_flow_last.pt \\
        --codec mp3:64k --codec-check mp3:128k:44100:2 opus:137k:48000:2 \\
        --eval-dir $CACHE --data-root $DATA --bucket $BUCKET
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

import evaluate_mert_probe as emp  # noqa: E402
import evaluate_mom as em  # noqa: E402
import track_mom as tm  # noqa: E402
from shs100k_meta import DEFAULT_DATA_ROOT  # noqa: E402
from src.data.codec import Chain, chain_name, chain_slug, check_encoders, parse_chain  # noqa: E402
from src.training import pick_device  # noqa: E402

AGGREGATES = emp.AGGREGATES  # ("min", "mean")
VIEWS = ("flow", "gaussian")
ALL = "all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                        help="train_mert_flow.py checkpoints (mert_flow_last.pt)")
    parser.add_argument("--codec", nargs="+", default=["mp3:64k"],
                        help="common chains, each applied to both classes as one condition "
                             "(default: mp3:64k, section 13's chain; 'none' = original audio)")
    parser.add_argument("--codec-check", nargs="*", default=[],
                        help="chains applied before each condition's chain to the same songs, "
                             "e.g. mp3:128k:44100:2 opus:137k:48000:2")
    parser.add_argument("--eval-dir", type=Path, default=None,
                        help="where the evaluation caches live (default: next to the first "
                             "flow's train cache; pass the probes' cache dir to reuse theirs)")
    parser.add_argument("--human-split", default="test50")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="human audio and manifests (the manifest gives the works)")
    parser.add_argument("--bucket", default="songs-and-mert-features")
    parser.add_argument("--region", default=None)
    parser.add_argument("--mom-prefix", default="mom")
    parser.add_argument("--mom-root", type=Path, default=None)
    parser.add_argument("--generators", nargs="+", default=list(em.GENERATORS))
    parser.add_argument("--per-source", type=int, default=300)
    parser.add_argument("--windows-per-song", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="song sampling seed")
    parser.add_argument("--fpr", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=1000, help="resamples for the intervals")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_flows(paths: list[Path], device: torch.device) -> dict[str, tuple]:
    """name -> (model, config, checkpoint, path); the name is the run directory."""
    from train_mert_flow import load_mert_flow

    flows: dict[str, tuple] = {}
    for path in paths:
        model, config, checkpoint = load_mert_flow(path, device)
        name = path.parent.name
        if name in flows:
            name = str(path.parent)
        flows[name] = (model, config, checkpoint, path)
    return flows


def label(chain: Chain) -> str:
    return chain_name(chain) or "original audio"


@torch.no_grad()
def window_scores(cache: tm.EvalCache, flows: dict, batch_size: int = 512) -> dict[str, dict]:
    """flow name -> {"flow": (n,), "gaussian": (n,)} log-densities, one per cached window."""
    out: dict[str, dict] = {}
    for name, (model, _, _, _) in flows.items():
        parts: dict[str, list] = {v: [] for v in VIEWS}
        for chunk in cache.pooled.split(batch_size):
            terms = model.log_probs(chunk.float())
            for v in VIEWS:
                parts[v].append(terms[v].float().cpu())
        out[name] = {v: torch.cat(parts[v]).numpy() for v in VIEWS}
    return out


def align(table: dict, keep: dict[str, list[str]]) -> dict:
    """Each source's song arrays restricted to, and ordered by, keep[source]."""
    out = {}
    for source, songs in keep.items():
        entry = table[source]
        index = {s: i for i, s in enumerate(entry["songs"])}
        picked = [index[s] for s in songs]
        out[source] = {"songs": songs, **{agg: entry[agg][picked] for agg in AGGREGATES}}
    return out


def synthetic_of(table: dict, target: str, generators: list[str], agg: str) -> np.ndarray:
    if target == ALL:
        return np.concatenate([table[g][agg] for g in generators])
    return table[target][agg]


def paired_change_interval(
    human: tuple[np.ndarray, np.ndarray], synthetic: tuple[np.ndarray, np.ndarray],
    groups: list, reps: int, rng: np.random.Generator,
) -> list[float]:
    """95 % interval of AUROC(new) - AUROC(reference) on the same resampled songs.

    `human` and `synthetic` are (reference, new) pairs of song-aligned arrays.
    """
    _, inverse = np.unique(np.array(groups, dtype=str), return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]
    values = []
    for _ in range(reps):
        h = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        s = rng.integers(0, len(synthetic[0]), len(synthetic[0]))
        values.append(em.auroc(human[1][h], synthetic[1][s]) - em.auroc(human[0][h], synthetic[0][s]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    flows = load_flows(args.checkpoints, device)
    first_config = next(iter(flows.values()))[1]
    for name, (_, config, _, _) in flows.items():
        if config.mert.model_name != first_config.mert.model_name:
            raise SystemExit(f"{name} was trained on {config.mert.model_name}, "
                             f"another flow on {first_config.mert.model_name}")
        if float(config.data.window_seconds) != float(first_config.data.window_seconds):
            raise SystemExit("the flows were trained on different window lengths")

    trained = {name: parse_chain(c.get("codec")) for name, (_, _, c, _) in flows.items()}
    conditions: list[Chain] = []
    for chain in [*trained.values(), *(parse_chain(spec) for spec in args.codec)]:
        if chain not in conditions:
            conditions.append(chain)
    checks = [parse_chain(spec) for spec in args.codec_check]
    if any(not c for c in checks):
        raise SystemExit("--codec-check needs real chains, not 'none'")
    check_encoders(tuple(s for c in conditions + checks for s in c))

    first_checkpoint = next(iter(flows.values()))[2]
    eval_dir = args.eval_dir or Path(first_checkpoint["train_cache"]).parent
    settings = {
        "human_split": args.human_split,
        "per_source": args.per_source,
        "windows_per_song": args.windows_per_song,
        "window_seconds": float(first_config.data.window_seconds),
        "seed": args.seed,
        "generators": list(args.generators),
        "model_name": first_config.mert.model_name,
    }
    scored: dict[str, tuple[Path, dict]] = {}

    def condition(chain: Chain) -> tuple[Path, dict]:
        """Build or load the evaluation cache under `chain` -> (root, flow -> view -> song table)."""
        key = chain_name(chain) or ""
        if key in scored:
            return scored[key]
        root = eval_dir / (f"eval-{args.human_split}" + (f"-{chain_slug(chain)}" if chain else ""))
        if not tm.cache_matches(root, dict(settings, codec=chain_name(chain))):
            tm.build_eval_cache(
                root,
                data_root=args.data_root,
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
                mert_config=first_config.mert,
                device=device,
                codec=chain,
            )
        cache = tm.EvalCache(root, device)
        print(f"{label(chain)}: {len(cache.rows)} windows from {root}")
        scores = window_scores(cache, flows)
        tables = {name: {v: emp.song_scores(cache.rows, scores[name][v], set()) for v in VIEWS}
                  for name in flows}
        del cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
        scored[key] = (root, tables)
        return scored[key]

    raw = {label(chain): condition(chain) for chain in conditions}

    # one song set for every condition, so the changes are paired
    sources = [em.HUMAN, *args.generators]
    any_tables = [t[next(iter(flows))]["flow"] for _, t in raw.values()]
    missing = [s for s in sources if any(s not in t for t in any_tables)]
    if em.HUMAN in missing:
        raise SystemExit(f"no {em.HUMAN} songs in one of the evaluation caches")
    if missing:
        print(f"warning: no songs of {', '.join(missing)} under some condition; left out")
    generators = [g for g in args.generators if g not in missing]
    keep = {s: sorted(set.intersection(*(set(t[s]["songs"]) for t in any_tables)))
            for s in [em.HUMAN, *generators]}
    dropped = {s: max(len(t[s]["songs"]) for t in any_tables) - len(keep[s]) for s in keep}
    tables = {cond: {name: {v: align(t[name][v], keep) for v in VIEWS} for name in flows}
              for cond, (_, t) in raw.items()}

    works = emp.work_of(args.data_root, args.human_split)
    human_songs = keep[em.HUMAN]
    groups = [works.get(s, s) for s in human_songs] if works else human_songs
    unit = "work" if works else "song (no manifest found; intervals are too narrow)"
    targets = [*generators, ALL]
    rng = np.random.default_rng(0)

    results: dict[str, dict] = {}
    for name in flows:
        reference = label(trained[name])
        results[name] = {}
        for cond in tables:
            results[name][cond] = {}
            for v in VIEWS:
                t, r = tables[cond][name][v], tables[reference][name][v]
                results[name][cond][v] = {}
                for agg in AGGREGATES:
                    human = t[em.HUMAN][agg]
                    threshold = em.finite_quantiles(human, [args.fpr])[0]
                    cells = {}
                    for target in targets:
                        synthetic = synthetic_of(t, target, generators, agg)
                        cell = {
                            "auroc": em.auroc(human, synthetic),
                            "ci95": emp.bootstrap_interval(human, groups, synthetic, args.bootstrap, rng),
                            "flagged_at_fpr": float(np.mean(synthetic < threshold)),
                        }
                        if cond != reference:
                            ref_synthetic = synthetic_of(r, target, generators, agg)
                            cell["change"] = {
                                "auroc": cell["auroc"] - em.auroc(r[em.HUMAN][agg], ref_synthetic),
                                "ci95": paired_change_interval(
                                    (r[em.HUMAN][agg], human), (ref_synthetic, synthetic),
                                    groups, args.bootstrap, rng),
                            }
                        cells[target] = cell
                    results[name][cond][v][agg] = cells

    print(f"\nhuman = {args.human_split}: {len(human_songs)} songs, {len(set(groups))} works; "
          + ", ".join(f"{g} {len(keep[g])}" for g in generators)
          + (f"; left out because they failed under some condition: "
             + ", ".join(f"{s} {n}" for s, n in dropped.items() if n) if any(dropped.values()) else ""))
    cw, lw = 22, 32  # cell and condition-label widths
    for name, (_, _, checkpoint, _) in flows.items():
        reference = label(trained[name])
        print(f"\n=== {name} (epoch {checkpoint.get('epoch')}, trained on {reference}) ===")
        for agg in AGGREGATES:
            for v in VIEWS:
                print(f"\nAUROC [95% CI], {v}, song score = {agg} of windows")
                print(f"{'condition':<{lw}}" + "".join(f"{t:>{cw}}" for t in targets))
                for cond, per_view in results[name].items():
                    cells = per_view[v][agg]
                    tag = " (training)" if cond == reference else ""
                    print(f"{(cond + tag)[:lw - 1]:<{lw}}" + "".join(
                        f"{cells[t]['auroc']:.3f} [{cells[t]['ci95'][0]:.2f},{cells[t]['ci95'][1]:.2f}]"
                        .rjust(cw) for t in targets))
                    if "change" in cells[targets[0]]:
                        print(f"{'  change vs training':<{lw}}" + "".join(
                            f"{cells[t]['change']['auroc']:+.3f} [{cells[t]['change']['ci95'][0]:+.2f},"
                            f"{cells[t]['change']['ci95'][1]:+.2f}]".rjust(cw) for t in targets))
            print(f"\nshare of synthetic songs flagged at {args.fpr:g} human FPR, flow, song score = {agg}")
            print(f"{'condition':<{lw}}" + "".join(f"{t:>{cw}}" for t in targets))
            for cond, per_view in results[name].items():
                cells = per_view["flow"][agg]
                print(f"{cond[:lw - 1]:<{lw}}" + "".join(f"{cells[t]['flagged_at_fpr']:.0%}".rjust(cw)
                                                   for t in targets))

    codec_checks: dict = {}
    for base in conditions:
        for check in checks:
            chain = check + base
            root_c, tables_c = condition(chain)
            entry: dict = {"condition": label(base), "prepended": chain_name(check),
                           "chain": chain_name(chain), "eval_cache": str(root_c), "flows": {}}
            for name in flows:
                entry["flows"][name] = {}
                for v in VIEWS:
                    base_table = tables[label(base)][name][v]
                    entry["flows"][name][v] = {}
                    for source in [em.HUMAN, *generators]:
                        if source not in tables_c[name][v]:
                            continue
                        entry["flows"][name][v][source] = {
                            agg: emp.codec_shift(
                                {"songs": base_table[source]["songs"], "value": base_table[source][agg]},
                                {"songs": tables_c[name][v][source]["songs"],
                                 "value": tables_c[name][v][source][agg]},
                                em.finite_quantiles(base_table[em.HUMAN][agg], [args.fpr])[0],
                            )
                            for agg in AGGREGATES
                        }
            codec_checks[f"{label(base)} <- {chain_name(check)}"] = entry
            print(f"\ncodec check: {chain_name(check)} before {label(base)}, same songs; "
                  "flow, song score = min")
            print("cell = shift AUROC / flagged re-encoded (flagged as-is). Shift 0.5: the flow ignores "
                  "the re-encoding; above 0.5: re-encoded songs look more synthetic")
            width = max(len(n) for n in flows) + 2
            print(f"{'flow':<{width}}" + "".join(f"{s:>{cw}}" for s in [em.HUMAN, *generators]))
            for name, per_view in entry["flows"].items():
                cells = []
                for source in [em.HUMAN, *generators]:
                    r = per_view["flow"].get(source, {}).get("min")
                    cells.append("—" if r is None else
                                 f"{r['shift_auroc']:.2f} / {r['flagged_at_fpr']:.0%} ({r['flagged_as_is']:.0%})")
                print(f"{name:<{width}}" + "".join(f"{c:>{cw}}" for c in cells))

    report = {
        "flows": {
            name: {"checkpoint": str(path), "epoch": checkpoint.get("epoch"), "arm": checkpoint.get("arm"),
                   "trained_codec": checkpoint.get("codec"), "train_cache": checkpoint.get("train_cache")}
            for name, (_, _, checkpoint, path) in flows.items()
        },
        "human_split": args.human_split,
        "conditions": {label(c): {"codec": chain_name(c), "eval_cache": str(raw[label(c)][0])}
                       for c in conditions},
        "songs": {s: len(v) for s, v in keep.items()},
        "songs_left_out": dropped,
        "human_works": len(set(groups)),
        "bootstrap": {"reps": args.bootstrap, "human_unit": unit},
        "fpr": args.fpr,
        "results": results,
        "codec_checks": codec_checks,
    }
    common = Path(os.path.commonpath([str(p.resolve().parent) for p in args.checkpoints]))
    out = args.out or common / f"eval_codec_{args.human_split}.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
