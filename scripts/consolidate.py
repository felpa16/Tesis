#!/usr/bin/env python3
"""Fix the MERT-only detector's protocol on val50, then score it once on unseen songs.

Why (experiments.md section 24). Sections 16-23 tried levers on the MERT-only
flow's features and score. Three cheap choices are left, all small, all still
open:
  * the song aggregate: the mean over windows beat the min on AUROC (sections
    15 and 18), while the min flagged more at 5 % human FPR (section 18);
  * a Gaussian refitted without the training recordings of large τ²: +0.011
    at the deepest trim tried, maybe not peaked (section 20);
  * the operating point: the ROC curve is steep, 30 % of synthetic songs
    caught at 5 % human FPR and 54 % at 10 % (section 20).
This script fixes all three on val50 with the dev generators, freezes them in
a protocol file, and scores the frozen protocol once on songs no earlier
choice has seen.

Stage 1: select (val50).
  Human: every val50 recording in the flow's val50 pooled cache (~1,670
  recordings, 4 windows each), not only the 300 sampled ones. That is ~5x the
  human songs behind each AUROC and each threshold.
  Synthetic: the 300 songs per generator of the val50 evaluation cache, the
  songs sections 16-23 scored.
  Candidates, every combination of
    detector   the checkpoint's flow; its Gaussian; and the Gaussian refitted
               on the training recordings whose geometric-mean τ² ≤ t, for each
               --thresholds t (trim_tau.py's recording level, same units)
    aggregate  song score = min or mean of its windows' log densities
  The rule, fixed before the run:
    1. Take the candidate with the highest song AUROC over the --dev
       generators pooled (riffusion and diffrhythm). The others (udio, yue)
       are printed and never used.
    2. On that candidate, take the human FPR f that maximises
       TPR_dev(f) - --fp-weight * f, searched on a 1 % grid up to --max-fpr.
       At weight 1 this is Youden's J. --operating-fpr fixes f by hand
       instead.
    3. The threshold is the val50 human quantile at f, kept as a fixed log
       density. A song is flagged as synthetic when its score is below it.
  Writes {out-dir}/protocol.json, plus protocol_gaussian.pt for a refitted
  Gaussian. That is the frozen protocol the test stage reads. It refuses to
  overwrite a protocol that has already been tested (--force overrides).

Stage 2: test (test50, once).
  Human: every test50 recording except the 300 that sections 10-15 scored
  (--exclude-seen, evaluate_mom.py's seed-0 sample).
  Synthetic: MoM songs that cache_pooled_mom.py cached after the first 300 of
  each generator's deterministic order. No evaluation has scored them. The
  script refuses any MoM song the selection used.
  Reported:
    * the protocol's song AUROC per generator, and pooled over three groups:
      the dev generators (tuned on, but these are new songs), the others
      (never tuned on), and all. Each comes with a bootstrap 95 % interval:
      human songs resampled by work, synthetic songs by song.
    * at the frozen threshold, the human FPR it actually reaches on test50, and
      the TPR per generator and pool. The same for the val50 thresholds at 5,
      10 and 20 %, to show how the calibration transfers.
    * the share flagged at test50's own 5 % human quantile, comparable with the
      earlier tables.
    * the paired change against fixed comparison rows: {flow, Gaussian} x {min,
      mean}. Section 11's protocol, flow | min, is among them.
  It refuses to run twice; --again overrides and labels the result.

Nothing here runs MERT. Inputs:
  select  --checkpoint   train_mert_flow.py checkpoint (its mix, flow and Gaussian)
          --train-cache  the flow's training cache (pooled-MERT layout, ~10 GB)
          --human-cache  val50, pooled-MERT layout (default: the checkpoint's --val-cache)
          --mom-cache    the val50 evaluation cache (track_mom.py layout)
  test    --protocol     select's protocol.json
          --human-cache  test50, pooled-MERT layout (cache_pooled_mert.py --split test50)
          --mom-cache    cache_pooled_mom.py --skip 300 output

Examples:
    python scripts/consolidate.py select --checkpoint $FLOW --train-cache $CACHE/train \\
        --human-cache $CACHE/val50 --mom-cache $CACHE/eval-val50 --data-root $DATA
    python scripts/consolidate.py test \\
        --protocol checkpoints/mert-flow-unfrozen/consolidate/protocol.json \\
        --human-cache $CACHE/test50 --mom-cache $CACHE/mom-fresh --data-root $DATA
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

import dilution_diagnostic as dd  # noqa: E402
import evaluate_mert_flow as emf  # noqa: E402
import evaluate_mert_probe as emp  # noqa: E402
import evaluate_mom as em  # noqa: E402
import trim_tau as tt  # noqa: E402
from src.data.codec import chain_name, parse_chain  # noqa: E402
from src.data.pooled_cache import PooledCache  # noqa: E402
from src.training import pick_device  # noqa: E402

DEV = ("riffusion", "diffrhythm")
AGGREGATES = emp.AGGREGATES  # ("min", "mean")
THRESHOLDS = (1.5, 1.25, 1.1, 1.0, 0.9)
REPORT_FPRS = (0.05, 0.10, 0.20)
CURVE_FPRS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
REFERENCE = ("flow", "min")  # section 11's headline protocol
COMPARISONS = (("flow", "min"), ("flow", "mean"), ("gaussian", "min"), ("gaussian", "mean"))
GAUSS_BUFFERS = ("feat_mean", "feat_std", "gauss_chol")
PROTOCOL = "protocol.json"
GAUSSIAN_STATE = "protocol_gaussian.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--human-cache", type=Path, default=None,
                       help="pooled-MERT cache of a human split, or a track_mom.py evaluation cache")
        p.add_argument("--mom-cache", type=Path, default=None,
                       help="track_mom.py evaluation cache or cache_pooled_mom.py cache (its generators are used)")
        p.add_argument("--data-root", type=Path, default=None,
                       help="for the human split's manifest (track -> work); default: the human cache's")
        p.add_argument("--bootstrap", type=int, default=1000, help="resamples for every interval")
        p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
        p.add_argument("--batch-size", type=int, default=512)
        p.add_argument("--device", default="auto")

    s = sub.add_parser("select", help="fix the protocol on val50")
    common(s)
    s.add_argument("--checkpoint", type=Path, required=True, help="train_mert_flow.py checkpoint")
    s.add_argument("--train-cache", type=Path, default=None,
                   help="the flow's training cache (default: the path the checkpoint records)")
    s.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS),
                   help="recording-level τ² trims of the Gaussian, in training-mean units (trim_tau.py)")
    s.add_argument("--dev", nargs="+", default=list(DEV), help="generators the protocol is chosen on")
    s.add_argument("--fp-weight", type=float, default=1.0,
                   help="operating point maximises TPR_dev - w * FPR (1 = Youden's J)")
    s.add_argument("--max-fpr", type=float, default=0.30, help="largest human FPR the rule may pick")
    s.add_argument("--operating-fpr", type=float, default=None,
                   help="fix the operating FPR by hand instead of the rule")
    s.add_argument("--out-dir", type=Path, default=None, help="default: {checkpoint dir}/consolidate")
    s.add_argument("--force", action="store_true",
                   help="overwrite a protocol although a test result for it exists")

    t = sub.add_parser("test", help="score the frozen protocol once")
    common(t)
    t.add_argument("--protocol", type=Path, required=True, help="select's protocol.json")
    t.add_argument("--checkpoint", type=Path, default=None,
                   help="the protocol's checkpoint, if it moved (its sha256 must match)")
    t.add_argument("--exclude-seen", type=int, default=300,
                   help="leave out the first N human songs of evaluate_mom.py's sample (scored in "
                        "sections 10-15); 0 keeps them")
    t.add_argument("--seen-seed", type=int, default=0, help="the sampling seed those evaluations used")
    t.add_argument("--again", action="store_true", help="score although a test result exists (labelled)")
    return parser.parse_args()


# --------------------------------------------------------------------------
# Caches
# --------------------------------------------------------------------------


def open_cache(root: Path) -> tuple[dict, list[dict], list[np.ndarray]]:
    """(meta, rows as {"source", "song", "start"}, row arrays) of a cache in either layout.

    A track_mom.py evaluation cache (pooled.npy + windows.jsonl) holds a human
    split and the generators. A pooled-MERT cache (shards) holds either one
    human split (cache_pooled_mert.py, rows {"key", "start"}) or MoM songs
    (cache_pooled_mom.py, rows with "source" and "song").
    """
    if (root / "pooled.npy").exists():
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        with open(root / "windows.jsonl", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        arrays = [np.load(root / "pooled.npy", mmap_mode="r")]
        meta = {**meta, "layout": "evaluation"}
    else:
        cache = PooledCache(root)
        rows = [{"source": r.get("source", em.HUMAN), "song": r.get("song", r["key"]), "start": r["start"]}
                for r in cache.rows()]
        arrays = cache.arrays
        meta = {**cache.meta, "layout": "pooled"}
        meta.setdefault("windows_per_song", meta.get("windows_per_track"))
    if sum(len(a) for a in arrays) != len(rows):
        raise SystemExit(f"{root}: {sum(len(a) for a in arrays)} rows but {len(rows)} index lines")
    return meta, rows, arrays


def human_split_of(meta: dict) -> str | None:
    return meta.get("human_split") if meta["layout"] == "evaluation" else meta.get("split")


@torch.no_grad()
def load_side(root: Path, side: str, model, config, trained, device: torch.device, batch: int
              ) -> tuple[torch.Tensor, list[dict], dict]:
    """One side of a cache, em.HUMAN or "generators" -> (layer mixes (n, F) fp32 on the device, rows, meta).

    The mix is the checkpoint's, before standardization. It is the same for the
    flow and for every refitted Gaussian, so it is computed once.
    """
    if not root.exists():
        raise SystemExit(f"no cache at {root}")
    meta, rows, arrays = open_cache(root)
    if meta.get("model_name") != config.mert.model_name:
        raise SystemExit(f"{root} was cached with {meta.get('model_name')}, the flow trained on "
                         f"{config.mert.model_name}")
    if float(meta["window_seconds"]) != float(config.data.window_seconds):
        raise SystemExit(f"{root} holds {meta['window_seconds']} s windows, the flow trained on "
                         f"{config.data.window_seconds} s")
    if parse_chain(meta.get("codec")) != trained:
        raise SystemExit(f"{root} is under codec {meta.get('codec')}, the flow trained under "
                         f"{chain_name(trained) or 'none'}")
    sources = np.array([r["source"] for r in rows])
    keep = sources == em.HUMAN if side == em.HUMAN else sources != em.HUMAN
    if not keep.any():
        raise SystemExit(f"{root} holds no {side} windows")
    out, offset = [], 0
    for array in arrays:
        for start in range(0, len(array), batch):
            stop = min(start + batch, len(array))
            mask = keep[offset + start:offset + stop]
            if mask.any():
                chunk = np.asarray(array[start:stop], dtype=np.float32)[mask]
                out.append(model.mixes(torch.from_numpy(chunk).to(device)))
        offset += len(array)
    return torch.cat(out), [r for r, k in zip(rows, keep) if k], meta


def incomplete_songs(rows: list[dict], windows: int) -> set[tuple[str, str]]:
    """Songs without exactly `windows` cached windows: a min over 3 is not a min over 4."""
    counts: dict[tuple[str, str], int] = {}
    for r in rows:
        key = (r["source"], r["song"])
        counts[key] = counts.get(key, 0) + 1
    return {key for key, n in counts.items() if n != windows}


def work_groups(songs: list[str], work_of: dict | None) -> list[str]:
    return [str((work_of or {}).get(s, s.split("_")[0])) for s in songs]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------


@torch.no_grad()
def flow_scores(model, mix: torch.Tensor, batch: int) -> np.ndarray:
    return torch.cat([model.flow_log_prob((m - model.feat_mean) / model.feat_std).double().cpu()
                      for m in mix.split(batch)]).numpy()


@torch.no_grad()
def gaussian_scores(gauss, mix: torch.Tensor, batch: int) -> np.ndarray:
    """Window log densities under `gauss`'s standardization and Gaussian (z units of that fit)."""
    return torch.cat([gauss.gaussian_log_prob((m - gauss.feat_mean) / gauss.feat_std).double().cpu()
                      for m in mix.split(batch)]).numpy()


def gaussian_state(model) -> dict[str, torch.Tensor]:
    return {k: getattr(model, k).detach().cpu().clone() for k in GAUSS_BUFFERS}


@torch.no_grad()
def load_gaussian_state(scratch, state: dict[str, torch.Tensor]) -> None:
    for k in GAUSS_BUFFERS:
        getattr(scratch, k).copy_(state[k].to(getattr(scratch, k).device))


def detector_name(trim: float | None) -> str:
    return "gaussian" if trim is None else f"gaussian τ²≤{trim:g}"


def candidate_name(detector: str, agg: str) -> str:
    return f"{detector} | {agg}"


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def synthetic(table: dict, members: list[str], agg: str) -> np.ndarray:
    return np.concatenate([table[g][agg] for g in members])


def share_below(values: np.ndarray, threshold: float) -> float:
    return float(np.mean(values < threshold))


def tpr_at(human: np.ndarray, synth: np.ndarray, fpr: float) -> float:
    return share_below(synth, float(np.quantile(human, fpr)))


def metrics(table: dict, agg: str, generators: list[str], pools: dict[str, list[str]],
            fprs=REPORT_FPRS) -> dict:
    """AUROC per generator and pool; TPR per pool at each human FPR (threshold from these humans)."""
    human = table[em.HUMAN][agg]
    auroc = {g: em.auroc(human, table[g][agg]) for g in generators}
    auroc.update({name: em.auroc(human, synthetic(table, m, agg)) for name, m in pools.items()})
    tpr = {f"{f:g}": {name: tpr_at(human, synthetic(table, m, agg), f) for name, m in pools.items()}
           for f in fprs}
    return {"auroc": auroc, "tpr": tpr}


def paired_delta(tables: dict, new: tuple[str, str], ref: tuple[str, str], members: list[str],
                 groups: list[str], reps: int, seed: int) -> list[float]:
    """95 % interval of AUROC(new) - AUROC(ref) on the same resampled songs; the same resamples every call."""
    (nd, na), (rd, ra) = new, ref
    return emf.paired_change_interval(
        (tables[rd][em.HUMAN][ra], tables[nd][em.HUMAN][na]),
        (synthetic(tables[rd], members, ra), synthetic(tables[nd], members, na)),
        groups, reps, np.random.default_rng(seed))


def share_interval(values: np.ndarray, threshold: float, members: list[np.ndarray] | None,
                   reps: int, rng: np.random.Generator) -> list[float]:
    """95 % interval of the share below `threshold`, resampled by group (`members`) or by item."""
    out = []
    for _ in range(reps):
        if members is None:
            idx = rng.integers(0, len(values), len(values))
        else:
            idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        out.append(share_below(values[idx], threshold))
    return dd.interval(out)


def fpr_grid(max_fpr: float) -> list[float]:
    return [round(k / 100, 2) for k in range(1, int(round(max_fpr * 100)) + 1)]


def ci(value: float, interval: list[float], signed: bool = False) -> str:
    lo, hi = interval
    if signed:
        return f"{value:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    return f"{value:.3f} [{lo:.3f}, {hi:.3f}]"


# --------------------------------------------------------------------------
# Stage 1: select
# --------------------------------------------------------------------------


def select(args: argparse.Namespace) -> None:
    started = time.time()
    device = pick_device(args.device)
    from train_mert_flow import load_mert_flow

    model, config, checkpoint = load_mert_flow(args.checkpoint, device)
    trained = parse_chain(checkpoint.get("codec"))
    ckpt_args = checkpoint.get("args", {})
    shrinkage = float(ckpt_args.get("gaussian_shrinkage", 0.01))
    if ckpt_args.get("stats_windows"):
        print("note: the checkpoint's moments were fitted on a subset of the training windows; the trims "
              "are refitted on all of them")
    train_root = args.train_cache or Path(checkpoint["train_cache"])
    val_cache = ckpt_args.get("val_cache")
    human_root = args.human_cache or (Path(val_cache) if val_cache not in (None, "None")
                                      else train_root.parent / "val50")
    mom_root = args.mom_cache or train_root.parent / "eval-val50"
    out_dir = args.out_dir or args.checkpoint.parent / "consolidate"
    tested = sorted(out_dir.glob("test_*.json"))
    if tested and not args.force:
        raise SystemExit(f"{tested[0]} exists: this protocol has been scored on test already. Choosing again "
                         "after looking at test would make test a selection set. --force overrides.")
    if args.operating_fpr is not None and not 0 < args.operating_fpr < 1:
        raise SystemExit("--operating-fpr must be in (0, 1)")

    # ---------------- evaluation windows ----------------
    human_mix, human_rows, human_meta = load_side(human_root, em.HUMAN, model, config, trained, device,
                                                  args.batch_size)
    split = human_split_of(human_meta)
    if split in ("test50", "train"):
        raise SystemExit(f"{human_root} is {split}: selection runs on val50")
    if split != "val50":
        print(f"warning: the human cache is split {split}, not val50")
    mom_mix, mom_rows, mom_meta = load_side(mom_root, "generators", model, config, trained, device,
                                            args.batch_size)
    windows = int(human_meta["windows_per_song"])
    if int(mom_meta["windows_per_song"]) != windows:
        raise SystemExit(f"{human_root} has {windows} windows per song, {mom_root} "
                         f"{mom_meta['windows_per_song']}")
    mix = torch.cat([human_mix, mom_mix])
    rows = human_rows + mom_rows
    present = {r["source"] for r in mom_rows}
    generators = [g for g in mom_meta.get("generators", em.GENERATORS) if g in present]
    pools = {"dev": [g for g in generators if g in args.dev],
             "others": [g for g in generators if g not in args.dev],
             "all": generators}
    if not pools["dev"]:
        raise SystemExit(f"none of --dev {args.dev} is in {mom_root}")
    pools = {k: v for k, v in pools.items() if v}
    exclude = incomplete_songs(rows, windows)

    data_root = args.data_root or Path(human_meta.get("data_root") or ".")
    work_of = emp.work_of(data_root, split)
    if work_of is None:
        print(f"note: no {split} manifest under {data_root}; works taken from the track-key prefix")

    # ---------------- τ² of the training recordings ----------------
    train = PooledCache(train_root)
    if train.meta["model_name"] != config.mert.model_name or \
            float(train.meta["window_seconds"]) != float(config.data.window_seconds) or \
            parse_chain(train.meta.get("codec")) != trained:
        raise SystemExit(f"{train_root} is not the flow's training cache (model, window or codec differ)")
    axes_np, lam = dd.principal_axes(model)
    axes32 = torch.from_numpy(axes_np / np.sqrt(lam)).to(device).float()
    z = tt.training_features(train, model, device, args.batch_size)
    keys = np.array([row["key"] for row in train.rows()], dtype=str)
    if len(keys) != len(z):
        raise SystemExit(f"{len(z)} training features against {len(keys)} index rows")
    tau2 = tt.squared_norms(z, axes32)
    tau2 /= tau2.mean()
    rec_tau2, rec_tau2_rows = tt.geometric_mean_per_unit(keys, tau2)

    # ---------------- detectors ----------------
    scores = {"flow": flow_scores(model, mix, args.batch_size),
              "gaussian": gaussian_scores(model, mix, args.batch_size)}
    states: dict[str, dict | None] = {"flow": None, "gaussian": None}
    trims: dict[str, float | None] = {"flow": None, "gaussian": None}
    kept: dict[str, dict] = {}
    scratch = copy.deepcopy(model)
    mean_all, cov_all = tt.moments(z, torch.ones(len(z), dtype=torch.bool, device=device))
    tt.refit(scratch, model, mean_all, cov_all, shrinkage)
    drift = float(np.abs(gaussian_scores(scratch, mix, args.batch_size) - scores["gaussian"]).max())
    del mean_all, cov_all
    for t in sorted(args.thresholds, reverse=True):
        keep_np = rec_tau2_rows <= t
        if keep_np.sum() <= z.shape[1]:
            print(f"skip: τ² ≤ {t:g} keeps {int(keep_np.sum())} windows, too few for a covariance")
            continue
        mean_k, cov_k = tt.moments(z, torch.from_numpy(keep_np).to(device))
        tt.refit(scratch, model, mean_k, cov_k, shrinkage)
        name = detector_name(t)
        scores[name] = gaussian_scores(scratch, mix, args.batch_size)
        states[name] = gaussian_state(scratch)
        trims[name] = t
        kept[name] = {"windows": float(keep_np.mean()), "recordings": float((rec_tau2 <= t).mean())}
        del mean_k, cov_k
    del z
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tables = {d: emp.song_scores(rows, s, exclude) for d, s in scores.items()}
    human_songs = tables["flow"][em.HUMAN]["songs"]
    groups = work_groups(human_songs, work_of)
    candidates = [(d, a) for d in scores for a in AGGREGATES]
    results: dict[str, dict] = {}
    for d, a in candidates:
        r = metrics(tables[d], a, generators, pools)
        r["detector"], r["aggregate"], r["trim"] = d, a, trims[d]
        if (d, a) != REFERENCE:
            r["delta_dev_vs_reference"] = paired_delta(tables, (d, a), REFERENCE, pools["dev"], groups,
                                                       args.bootstrap, args.seed)
        results[candidate_name(d, a)] = r

    # ---------------- the rule ----------------
    ranked = sorted(candidates, key=lambda c: -results[candidate_name(*c)]["auroc"]["dev"])
    chosen = ranked[0]
    runner_up = ranked[1]
    margin = paired_delta(tables, chosen, runner_up, pools["dev"], groups, args.bootstrap, args.seed)
    det, agg = chosen
    table = tables[det]
    human = table[em.HUMAN][agg]
    dev_synth = synthetic(table, pools["dev"], agg)
    others_synth = synthetic(table, pools["others"], agg) if "others" in pools else None
    curve = []
    for f in sorted(set(fpr_grid(max(args.max_fpr, max(CURVE_FPRS)))) | set(CURVE_FPRS)):
        tpr_dev = tpr_at(human, dev_synth, f)
        curve.append({"fpr": f, "threshold": float(np.quantile(human, f)), "tpr_dev": tpr_dev,
                      "tpr_others": tpr_at(human, others_synth, f) if others_synth is not None else None,
                      "utility": tpr_dev - args.fp_weight * f})
    if args.operating_fpr is not None:
        operating = args.operating_fpr
        rule = f"fixed by hand at {operating:g}"
    else:
        eligible = [c for c in curve if c["fpr"] <= args.max_fpr + 1e-9]
        operating = max(eligible, key=lambda c: c["utility"])["fpr"]  # first (smallest f) on ties
        rule = f"max TPR_dev - {args.fp_weight:g} * FPR, FPR <= {args.max_fpr:g}, 1 % grid"
    threshold = float(np.quantile(human, operating))

    # ---------------- check against section 18 (the evaluation cache's own 300 val50 songs) ----------------
    check = None
    if mom_meta["layout"] == "evaluation":
        try:
            c_mix, c_rows, _ = load_side(mom_root, em.HUMAN, model, config, trained, device, args.batch_size)
        except SystemExit:
            c_mix = None
        if c_mix is not None:
            both_rows = c_rows + mom_rows
            both_mix = torch.cat([c_mix, mom_mix])
            check = {}
            for d, fn in (("flow", flow_scores), ("gaussian", gaussian_scores)):
                tab = emp.song_scores(both_rows, fn(model, both_mix, args.batch_size),
                                      incomplete_songs(both_rows, windows))
                check[d] = em.auroc(tab[em.HUMAN]["min"], synthetic(tab, generators, "min"))

    # ---------------- report ----------------
    n_human = len(human_songs)
    print(f"\nflow {args.checkpoint} (arm {checkpoint.get('arm')}, epoch {checkpoint.get('epoch')}); "
          f"Gaussian shrinkage s = {shrinkage:g}")
    print(f"human: {human_root} ({split}): {n_human} songs from {len(set(groups))} works")
    print(f"synthetic: {mom_root}: " + ", ".join(f"{g} {len(table[g]['songs'])}" for g in generators)
          + f"; dev = {', '.join(pools['dev'])}")
    print(f"songs left out for a missing window: {len(exclude)}")
    print(f"training: {len(keys)} windows, {len(rec_tau2)} recordings; recording τ² quantiles "
          + ", ".join(f"{p:.0%} {np.quantile(rec_tau2, p):.2f}" for p in (0.25, 0.5, 0.75, 0.9, 0.95)))
    print(f"check: the Gaussian refitted on all training windows vs. the checkpoint's: max |Δ log p| = "
          f"{drift:.3g} nats (section 20: 0.0095)")
    if check is not None:
        print(f"check: on the evaluation cache's own 300 val50 songs, song min, all generators: flow "
              f"{check['flow']:.3f}, Gaussian {check['gaussian']:.3f} (section 18: 0.823, 0.814)")

    print("\n=== Trims: share of training data each refitted Gaussian keeps ===")
    for name, k in kept.items():
        print(f"{name:<22} windows {k['windows']:.3f}  recordings {k['recordings']:.3f}")

    cols = generators + list(pools)
    header = [f"{g}*" if g in pools["dev"] else g for g in generators] + list(pools)
    print(f"\n=== Candidates, sorted by dev AUROC (song level; {split}, {n_human} songs) ===")
    print("* = dev generator. Δ dev = against flow | min (section 11's protocol), paired 95 %. "
          "TPR dev at 5 / 10 / 20 % human FPR.")
    print(f"{'candidate':<30}" + "".join(f"{h:>12}" for h in header) + f"{'Δ dev':>28}{'TPR dev':>21}")
    for d, a in ranked:
        r = results[candidate_name(d, a)]
        delta = (ci(r["auroc"]["dev"] - results[candidate_name(*REFERENCE)]["auroc"]["dev"],
                    r["delta_dev_vs_reference"], signed=True) if "delta_dev_vs_reference" in r else "reference")
        tprs = " / ".join(f"{r['tpr'][f'{f:g}']['dev']:.2f}" for f in REPORT_FPRS)
        mark = "  <- chosen" if (d, a) == chosen else ""
        print(f"{candidate_name(d, a):<30}" + "".join(f"{r['auroc'][c]:>12.3f}" for c in cols)
              + f"{delta:>28}{tprs:>21}{mark}")
    winner = results[candidate_name(*chosen)]
    print(f"\nchosen: {candidate_name(*chosen)}, dev AUROC {winner['auroc']['dev']:.3f}; against the runner-up "
          f"{candidate_name(*runner_up)}: {ci(winner['auroc']['dev'] - results[candidate_name(*runner_up)]['auroc']['dev'], margin, signed=True)}")

    print(f"\n=== Operating point on {candidate_name(*chosen)} (val50 human quantiles) ===")
    print(f"rule: {rule}")
    print(f"{'human FPR':>10}{'threshold':>14}{'TPR dev':>10}{'TPR others':>12}{'TPR - w FPR':>13}")
    for c in curve:
        if c["fpr"] in CURVE_FPRS or c["fpr"] == operating:
            others = f"{c['tpr_others']:>12.3f}" if c["tpr_others"] is not None else f"{'-':>12}"
            mark = "  <- operating" if abs(c["fpr"] - operating) < 1e-9 else ""
            print(f"{c['fpr']:>10.2f}{c['threshold']:>14.2f}{c['tpr_dev']:>10.3f}{others}"
                  f"{c['utility']:>13.3f}{mark}")
    if args.operating_fpr is not None and not any(abs(c["fpr"] - operating) < 1e-9 for c in curve):
        print(f"{operating:>10.3f}{threshold:>14.2f}{tpr_at(human, dev_synth, operating):>10.3f}  <- operating")
    print(f"threshold: flag a song as synthetic when its {agg} window log density is below {threshold:.3f}")

    # ---------------- protocol ----------------
    out_dir.mkdir(parents=True, exist_ok=True)
    gaussian_file = None
    if states.get(det) is not None:
        gaussian_file = GAUSSIAN_STATE
        torch.save(states[det], out_dir / GAUSSIAN_STATE)
    protocol = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "arm": checkpoint.get("arm"), "epoch": checkpoint.get("epoch"),
        "detector": det, "trim": trims[det], "aggregate": agg, "gaussian_state": gaussian_file,
        "windows_per_song": windows, "generators": generators, "pools": pools,
        "operating": {
            "rule": rule, "fp_weight": args.fp_weight, "max_fpr": args.max_fpr, "fpr": operating,
            "threshold": threshold,
            "val_thresholds": {f"{f:g}": float(np.quantile(human, f)) for f in REPORT_FPRS},
        },
        "selection": {
            "human_cache": str(human_root), "mom_cache": str(mom_root), "human_split": split,
            "human_songs": n_human, "works": len(set(groups)), "left_out_incomplete": len(exclude),
            "mom_songs": {g: list(table[g]["songs"]) for g in generators},
            "train_cache": str(train_root), "shrinkage": shrinkage,
            "thresholds": sorted(args.thresholds, reverse=True), "kept": kept,
            "bootstrap": args.bootstrap, "seed": args.seed,
        },
        "chosen_val": winner,
        "runner_up": {"candidate": candidate_name(*runner_up), "delta_dev_ci95": margin},
        "candidates": results,
        "curve": curve,
        "check": {"refit_drift_nats": drift, "eval_cache_300_min_all": check},
    }
    path = out_dir / PROTOCOL
    path.write_text(json.dumps(protocol, indent=1) + "\n", encoding="utf-8")
    print(f"\n-> {path}" + (f" and {out_dir / GAUSSIAN_STATE}" if gaussian_file else "")
          + f" ({time.time() - started:.0f}s)")


# --------------------------------------------------------------------------
# Stage 2: test
# --------------------------------------------------------------------------


def test(args: argparse.Namespace) -> None:
    started = time.time()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    out_dir = args.protocol.parent
    earlier = sorted(out_dir.glob("test_*.json"))
    if earlier and not args.again:
        raise SystemExit(f"{earlier[0]} exists: the protocol has been scored on test. The test is scored "
                         "once; --again overrides, and the result is labelled as a repeat.")
    if args.human_cache is None or args.mom_cache is None:
        raise SystemExit("test needs --human-cache (test50) and --mom-cache (cache_pooled_mom.py --skip 300)")
    ckpt_path = args.checkpoint or Path(protocol["checkpoint"])
    if file_sha256(ckpt_path) != protocol["checkpoint_sha256"]:
        raise SystemExit(f"{ckpt_path} is not the checkpoint the protocol was selected with (sha256 differs)")

    device = pick_device(args.device)
    from train_mert_flow import load_mert_flow

    model, config, checkpoint = load_mert_flow(ckpt_path, device)
    trained = parse_chain(checkpoint.get("codec"))
    windows = int(protocol["windows_per_song"])

    # ---------------- songs ----------------
    human_mix, human_rows, human_meta = load_side(args.human_cache, em.HUMAN, model, config, trained, device,
                                                  args.batch_size)
    split = human_split_of(human_meta)
    if split in (protocol["selection"]["human_split"], "train"):
        raise SystemExit(f"{args.human_cache} is {split}: the protocol was selected on "
                         f"{protocol['selection']['human_split']}; test on a held-out split")
    mom_mix, mom_rows, mom_meta = load_side(args.mom_cache, "generators", model, config, trained, device,
                                            args.batch_size)
    for root, meta in ((args.human_cache, human_meta), (args.mom_cache, mom_meta)):
        if int(meta["windows_per_song"]) != windows:
            raise SystemExit(f"{root} has {meta['windows_per_song']} windows per song; the protocol, {windows}")

    data_root = args.data_root or Path(human_meta.get("data_root") or ".")
    seen: set[str] = set()
    if args.exclude_seen > 0:
        sample = em.deterministic_sample(
            em.human_songs(data_root, split, float(config.data.window_seconds)), args.exclude_seen,
            args.seen_seed)
        seen = {s.song_id for s in sample}
    used = {g: set(s) for g, s in protocol["selection"]["mom_songs"].items()}
    overlap = {g: len({r["song"] for r in mom_rows if r["source"] == g} & used.get(g, set()))
               for g in protocol["generators"]}
    if any(overlap.values()):
        raise SystemExit("the MoM cache holds songs the selection used: "
                         + ", ".join(f"{g} {n}" for g, n in overlap.items() if n)
                         + ". Build it with cache_pooled_mom.py --skip 300 (the default) and seed 0.")
    present = {r["source"] for r in mom_rows}
    missing = [g for g in protocol["generators"] if g not in present]
    if missing:
        raise SystemExit(f"{args.mom_cache} has no songs of {', '.join(missing)}")
    generators = list(protocol["generators"])
    pools = {name: list(m) for name, m in protocol["pools"].items()}
    mom_keep = [r["source"] in generators for r in mom_rows]
    mom_mix = mom_mix[torch.from_numpy(np.array(mom_keep)).to(device)]
    mom_rows = [r for r, k in zip(mom_rows, mom_keep) if k]

    mix = torch.cat([human_mix, mom_mix])
    rows = human_rows + mom_rows
    incomplete = incomplete_songs(rows, windows)
    human_all = {r["song"] for r in human_rows}
    excluded_seen = {(em.HUMAN, s) for s in human_all & seen}
    exclude = incomplete | excluded_seen

    # ---------------- detectors ----------------
    det, agg = protocol["detector"], protocol["aggregate"]
    scores = {"flow": flow_scores(model, mix, args.batch_size),
              "gaussian": gaussian_scores(model, mix, args.batch_size)}
    if det not in scores:
        if not protocol.get("gaussian_state"):
            raise SystemExit(f"the protocol's detector {det} needs {GAUSSIAN_STATE}, which it does not name")
        scratch = copy.deepcopy(model)
        load_gaussian_state(scratch, torch.load(out_dir / protocol["gaussian_state"], map_location="cpu"))
        scores[det] = gaussian_scores(scratch, mix, args.batch_size)
        del scratch
    tables = {d: emp.song_scores(rows, s, exclude) for d, s in scores.items()}
    table = tables[det]
    human = table[em.HUMAN][agg]
    human_songs = table[em.HUMAN]["songs"]
    work_of = emp.work_of(data_root, split)
    if work_of is None:
        print(f"note: no {split} manifest under {data_root}; works taken from the track-key prefix")
    groups = work_groups(human_songs, work_of)
    members = tt.unit_members(groups)
    rng = np.random.default_rng(args.seed)

    # ---------------- the protocol ----------------
    targets = generators + list(pools)

    def target_synth(t: dict, target: str, a: str) -> np.ndarray:
        return synthetic(t, pools[target], a) if target in pools else t[target][a]

    auroc = {x: {"value": em.auroc(human, target_synth(table, x, agg)),
                 "ci95": emp.bootstrap_interval(human, groups, target_synth(table, x, agg), args.bootstrap, rng)}
             for x in targets}
    op = protocol["operating"]
    thresholds = {"operating": (op["fpr"], op["threshold"]),
                  **{f"val {f}": (float(f), v) for f, v in op["val_thresholds"].items()}}
    at_threshold = {}
    for label, (f_val, theta) in thresholds.items():
        at_threshold[label] = {
            "val_fpr": f_val, "threshold": theta,
            "fpr": share_below(human, theta),
            "fpr_ci95": share_interval(human, theta, members, args.bootstrap, rng),
            "tpr": {x: share_below(target_synth(table, x, agg), theta) for x in targets},
            "tpr_ci95": {x: share_interval(target_synth(table, x, agg), theta, None, args.bootstrap, rng)
                         for x in targets},
        }
    own5 = {x: tpr_at(human, target_synth(table, x, agg), 0.05) for x in targets}

    # ---------------- comparison rows ----------------
    comparisons = {}
    for d, a in COMPARISONS:
        if (d, a) == (det, agg):
            continue
        h = tables[d][em.HUMAN][a]
        comparisons[candidate_name(d, a)] = {
            "auroc": {x: em.auroc(h, target_synth(tables[d], x, a)) for x in targets},
            "flagged_own_5pct": {x: tpr_at(h, target_synth(tables[d], x, a), 0.05) for x in targets},
            "delta_protocol_minus_this": {
                name: paired_delta(tables, (det, agg), (d, a), m, groups, args.bootstrap, args.seed)
                for name, m in pools.items()},
        }

    # ---------------- report ----------------
    name = candidate_name(det, agg)
    repeat = " (REPEAT: --again)" if earlier else ""
    print(f"\nprotocol {args.protocol}: {name}, operating FPR {op['fpr']:g} ({op['rule']}), "
          f"threshold {op['threshold']:.3f}{repeat}")
    print(f"selected on {protocol['selection']['human_split']}: dev AUROC "
          f"{protocol['chosen_val']['auroc']['dev']:.3f}, all {protocol['chosen_val']['auroc']['all']:.3f}")
    n_seen = len(excluded_seen)
    print(f"human: {args.human_cache} ({split}): {len(human_songs)} songs from {len(set(groups))} works "
          f"({n_seen} left out as seen in sections 10-15, "
          f"{sum(1 for s, _ in incomplete if s == em.HUMAN)} for a missing window)")
    print(f"synthetic: {args.mom_cache} (skip {mom_meta.get('skip', 'n/a')}, seed {mom_meta.get('seed', 'n/a')}): "
          + ", ".join(f"{g} {len(table[g]['songs'])}" for g in generators)
          + "; none used by the selection")
    if args.exclude_seen > 0 and n_seen == 0:
        print("warning: none of the seen test50 songs is in the human cache; is the manifest the cache's?")

    header = [f"{g}*" if g in pools.get("dev", []) else g for g in generators] + list(pools)
    print("\n=== The protocol on unseen songs: song AUROC, bootstrap 95 % (human by work, synthetic by song) ===")
    print("* = dev generator (tuned on, new songs); others = never tuned on")
    for x, h in zip(targets, header):
        print(f"{h:<14}{ci(auroc[x]['value'], auroc[x]['ci95'])}")

    print("\n=== Flagged as synthetic at fixed val50 thresholds ===")
    print(f"{'threshold':<16}{'val FPR':>8}{'test FPR':>24}" + "".join(f"{h:>12}" for h in header))
    for label, r in at_threshold.items():
        print(f"{label:<16}{r['val_fpr']:>8.2f}{ci(r['fpr'], r['fpr_ci95']):>24}"
              + "".join(f"{r['tpr'][x]:>12.3f}" for x in targets))
    r = at_threshold["operating"]
    print("operating point, TPR intervals: " + "; ".join(
        f"{h} {ci(r['tpr'][x], r['tpr_ci95'][x])}" for x, h in zip(targets, header)))
    print("at test50's own 5 % human quantile (comparable with earlier tables): "
          + ", ".join(f"{h} {own5[x]:.3f}" for x, h in zip(targets, header)))

    print("\n=== Against fixed comparison rows (same songs): AUROC, and protocol minus row, paired 95 % ===")
    print(f"{'row':<20}" + "".join(f"{h:>12}" for h in header)
          + "".join(f"{'Δ ' + p:>28}" for p in pools) + f"{'flagged @5%':>13}")
    print(f"{name:<20}" + "".join(f"{auroc[x]['value']:>12.3f}" for x in targets)
          + "".join(f"{'protocol':>28}" for _ in pools) + f"{own5['all']:>13.3f}")
    for row, c in comparisons.items():
        deltas = "".join(
            f"{ci(auroc[p]['value'] - c['auroc'][p], c['delta_protocol_minus_this'][p], signed=True):>28}"
            for p in pools)
        print(f"{row:<20}" + "".join(f"{c['auroc'][x]:>12.3f}" for x in targets) + deltas
              + f"{c['flagged_own_5pct']['all']:>13.3f}")

    result = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "repeat": bool(earlier), "protocol": str(args.protocol), "protocol_content": protocol,
        "human_cache": str(args.human_cache), "mom_cache": str(args.mom_cache), "human_split": split,
        "human_songs": len(human_songs), "works": len(set(groups)), "left_out_seen": n_seen,
        "left_out_incomplete": len(incomplete),
        "mom_songs": {g: len(table[g]["songs"]) for g in generators},
        "mom_cache_meta": {k: mom_meta.get(k) for k in ("skip", "seed", "per_generator", "per_source")},
        "auroc": auroc, "at_threshold": at_threshold, "flagged_own_5pct": own5,
        "comparisons": comparisons, "bootstrap": args.bootstrap, "seed": args.seed,
    }
    path = out_dir / (f"test_{split}.json" if not earlier else f"test_{split}_repeat{len(earlier)}.json")
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"\n-> {path} ({time.time() - started:.0f}s)")


def main() -> None:
    args = parse_args()
    if args.stage == "select":
        select(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
