#!/usr/bin/env python3
"""Trimmed Gaussian: refit the baseline without the training windows of large τ².

Why (experiments.md section 20). Listening to the human tails of τ
(listen_tails.py) put noise, an amateur phone recording and worn tape among the
most atypical val50 recordings, next to studio metal, electronic and ambient
music. The proposal (the author's): drop the training windows whose τ² is above
a threshold, so that the density is fitted to the bulk of human music, and see
whether fitting less junk gains more AUROC than the rarer genres then lose. The
Gaussian is refitted, not the flow: it is closed form, takes seconds per
threshold, and lands within 0.01 AUROC of the flow (section 11).

What.
  1. Every window of the flow's training cache goes through the checkpoint's
     layer mix and standardization. Its τ² is ‖w‖² under the checkpoint's
     Gaussian, over the training mean of ‖w‖², so 1 is an average training
     window. This is in-sample. listen_tails.py normalises by the val50 mean
     instead; the header prints both means, so a cut-off can be converted.
  2. For each --level and --threshold t, the windows kept are
       window     those with τ² ≤ t
       recording  every window of the recordings whose geometric-mean τ² is
                  ≤ t (the statistic listen_tails.py ranks recordings by)
     and the Gaussian is refitted on them exactly as train_mert_flow.py fits
     it: mean and covariance of the mix, per-dimension standardization, and
     the correlation shrunk toward I by the checkpoint's --gaussian-shrinkage
     (MertFlow.set_moments itself is called).
  3. Each refit scores the evaluation cache (the human split and the MoM
     generators, as dilution_diagnostic.py), and is compared with the refit on
     all training windows ("none"). Reported: the song-level AUROC (song = min
     over its windows, the headline) per generator, pooled over all
     generators, over the --dev generators and over the others; the pooled
     window-level AUROC; and the share of synthetic songs flagged at 5 %
     human FPR. Δ against "none" comes with a paired bootstrap 95 % interval:
     human songs resampled by work, generator songs by song, the same
     resamples for every threshold.
  4. The author's operating point (section 20, second follow-up). The
     untrimmed refit's threshold at --fpr human FPR is kept as a fixed log
     density and applied to every refit. For this, log p is taken of the mix
     itself rather than of the standardized features, so that one value means
     the same density in every refit. Reported per refit:
       * the human FPR and the synthetic TPR it reaches at that threshold;
       * the untrimmed refit's TPR at that same FPR. A trim that only moves
         along the untrimmed ROC curve gives the same number;
       * the held-out human songs it newly flags, by key. Titles and links
         are in listen_tails.py's recordings.csv.
     The untrimmed refit's own curve (TPR at 5, 10, 20 and 30 % FPR) is
     printed too. That is the same trade made without any trim.

Prediction (section 20, written before the run). Under section 16's scale
model w = τ g, with g spherically symmetric and independent of τ, trimming on
‖w‖² changes τ's distribution and almost nothing else. The kept windows'
covariance is k Σ for one scalar k and their mean does not move. A Gaussian
refitted on them ranks every window as the untrimmed one does, so ΔAUROC = 0 at
every threshold. Section 18 found the scale model holds for PCs 257-1,024 and
fails for the top 128, so any change has to come from there. The script prints
what says whether it does:
  k        variance of the kept windows over that of all windows, summed over
           the PCs of the checkpoint's Gaussian (≈ the kept windows' mean τ²)
  band/k   the same ratio per band of PCs, over k. 1 in every band for a pure
           scale. Above 1: the trim shrank that band less than the rest, so
           the refit is relatively looser along it. Below 1: stricter.
  shift    ‖mean of the kept windows - mean of all‖² in whitened units, 0 for
           a pure scale
  ρ        Spearman correlation of the refit's evaluation scores with the
           untrimmed refit's. 1 = the same ranking, hence the same AUROC
  >t       share of held-out human windows (window level) or songs (recording
           level) above the same cut-off: the trim does not remove these, and
           they are where the false positives come from
Under the same model, the operating point of item 4 lies on the untrimmed
curve: the refit's TPR at the fixed threshold equals the untrimmed TPR at the
FPR it reaches.

Selection. A threshold chosen by looking at the AUROC is fitted to these
generators. Choose it on the --dev generators (default riffusion and
diffrhythm; section 12, "Selection leakage") and read the others as untouched.
Human = val50, not test50, for the same reason.

Input. A train_mert_flow.py checkpoint, its training cache (pooled-MERT layout,
~10 GB, ~203 k windows) and an evaluation cache in track_mom.py's layout, by
default {train cache}/../eval-{human-split}[-{chain}] as dilution_diagnostic.py
looks for it. Nothing here runs MERT. The training features are held on the
device as fp32 (~0.8 GB for 203 k windows), next to a copy of the model.

Writes {checkpoint dir}/trim_tau_{human-split}.json (or --out).

Example:
    python scripts/trim_tau.py \\
        --checkpoint checkpoints/mert-flow-unfrozen/mert_flow_last.pt \\
        --train-cache $CACHE/train --eval-cache $CACHE/eval-val50 \\
        --thresholds 1.25 1.5 2 3
"""

from __future__ import annotations

import argparse
import copy
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
import evaluate_mert_probe as emp  # noqa: E402
import evaluate_mom as em  # noqa: E402
import track_mom as tm  # noqa: E402
from src.data.codec import chain_name, chain_slug, parse_chain  # noqa: E402
from src.data.pooled_cache import PooledCache  # noqa: E402
from src.training import pick_device  # noqa: E402

LEVELS = ("window", "recording")
DEV = ("riffusion", "diffrhythm")
QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999)
CURVE = (0.05, 0.10, 0.20, 0.30)  # human FPRs for the untrimmed refit's operating curve
CHUNK = 8192  # rows per matrix product over the training features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="train_mert_flow.py checkpoint (its mix and Gaussian)")
    parser.add_argument("--train-cache", type=Path, default=None,
                        help="the flow's training cache (default: the path the checkpoint records)")
    parser.add_argument("--eval-cache", type=Path, default=None,
                        help="track_mom.py-layout cache (default: {train cache}/../eval-{human-split}[-chain])")
    parser.add_argument("--human-split", default="val50", help="for the default --eval-cache")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="for the human split's manifest (track -> work); default: the cache's")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[1.25, 1.5, 2.0, 3.0],
                        help="τ² cut-offs, in units of the training mean of ‖w‖²")
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS),
                        help="trim single windows, or whole recordings by their geometric-mean τ²")
    parser.add_argument("--dev", nargs="+", default=list(DEV),
                        help="generators a threshold may be chosen on; the others are read untouched")
    parser.add_argument("--fpr", type=float, default=0.05,
                        help="human FPR of the threshold: flagged share, and the fixed log density of item 4")
    parser.add_argument("--bands", type=int, default=8, help="equal-size groups of PCs for band/k")
    parser.add_argument("--bootstrap", type=int, default=1000, help="paired resamples for the Δ intervals")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512, help="cache rows per forward")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


# --------------------------------------------------------------------------
# Features and moments
# --------------------------------------------------------------------------


@torch.no_grad()
def training_features(cache: PooledCache, model, device: torch.device, batch_size: int) -> torch.Tensor:
    """Every training window -> the checkpoint's standardized mix z (n, F), fp32 on the device."""
    out = []
    started = time.time()
    for array in cache.shards():
        for start in range(0, len(array), batch_size):
            chunk = torch.from_numpy(array[start:start + batch_size]).to(device)
            out.append(model.features(chunk))
    z = torch.cat(out)
    print(f"training features: {len(z)} windows in {time.time() - started:.0f}s")
    return z


@torch.no_grad()
def squared_norms(z: torch.Tensor, rotation: torch.Tensor) -> np.ndarray:
    """‖w‖² of every row, w = z @ rotation (the checkpoint Gaussian's whitened coordinates)."""
    return torch.cat([((z[s:s + CHUNK] @ rotation) ** 2).sum(1) for s in range(0, len(z), CHUNK)]
                     ).double().cpu().numpy()


@torch.no_grad()
def moments(z: torch.Tensor, keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean (F,) and covariance (F, F), fp64, of the kept rows of z.

    Two passes, as train_mert_flow.fit_layer_stats: fp32 products on centred
    data, summed in fp64, normalised by n - 1.
    """
    n = int(keep.sum())
    total = torch.zeros(z.shape[1], dtype=torch.float64, device=z.device)
    for s in range(0, len(z), CHUNK):
        total += z[s:s + CHUNK][keep[s:s + CHUNK]].sum(0, dtype=torch.float64)
    mean = total / n
    centre = mean.float()
    acc = torch.zeros(z.shape[1], z.shape[1], dtype=torch.float64, device=z.device)
    for s in range(0, len(z), CHUNK):
        x = z[s:s + CHUNK][keep[s:s + CHUNK]] - centre
        acc += (x.T @ x).double()
    return mean, acc / max(n - 1, 1)


def refit(scratch, model, mean_z: torch.Tensor, cov_z: torch.Tensor, shrinkage: float) -> None:
    """Set scratch's Gaussian to these moments, given in the checkpoint's standardized units.

    z = (mix - feat_mean) / feat_std, so the mix moments are feat_mean +
    feat_std * mean_z and feat_std cov_z feat_std. set_moments then
    re-standardizes and shrinks exactly as training did.
    """
    std = model.feat_std.double()
    mean = model.feat_mean.double() + std * mean_z
    cov = cov_z * std[:, None] * std[None, :]
    scratch.set_moments(mean.float(), cov.float(), shrinkage)


@torch.no_grad()
def eval_terms(cache: tm.EvalCache, model, rotation: torch.Tensor, batch_size: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """Evaluation windows -> (Gaussian log p under `model`, ‖w‖² in rotation's coordinates)."""
    logp, q = [], []
    for chunk in cache.pooled.split(batch_size):
        z = model.features(chunk.float())
        logp.append(model.gaussian_log_prob(z).double().cpu())
        q.append(((z @ rotation) ** 2).sum(1).double().cpu())
    return torch.cat(logp).numpy(), torch.cat(q).numpy()


@torch.no_grad()
def eval_logp(cache: tm.EvalCache, model, batch_size: int) -> np.ndarray:
    return torch.cat([model.gaussian_log_prob(model.features(c.float())).double().cpu()
                      for c in cache.pooled.split(batch_size)]).numpy()


def scale_diagnostics(mean_k, cov_k, mean_a, cov_a, axes: torch.Tensor, edges: np.ndarray) -> dict:
    """How far the trim is from a pure scale, in the checkpoint Gaussian's whitened coordinates.

    `axes` is the whitening rotation (F, F) in fp64: column i is PC i over √λ_i.
    """
    var_k = ((cov_k @ axes) * axes).sum(0)
    var_a = ((cov_a @ axes) * axes).sum(0)
    k = float(var_k.sum() / var_a.sum())
    bands = [float(var_k[lo:hi].sum() / var_a[lo:hi].sum()) / k for lo, hi in zip(edges[:-1], edges[1:])]
    shift = float((((mean_k - mean_a) @ axes) ** 2).sum())
    return {"k": k, "band_over_k": bands, "shift": shift}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def geometric_mean_per_unit(units: np.ndarray, tau2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(per-unit geometric mean of τ², each row's unit's value)."""
    _, inverse = np.unique(units, return_inverse=True)
    per_unit = np.exp(np.bincount(inverse, np.log(tau2)) / np.bincount(inverse))
    return per_unit, per_unit[inverse]


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def detection(rows: list[dict], scores: np.ndarray, generators: list[str], pools: dict[str, list[str]],
              fpr: float) -> tuple[dict, dict]:
    """AUROCs (song = min over windows) per generator and pool, window AUROC, flagged at `fpr`."""
    table = emp.song_scores(rows, scores, set())
    human = table[em.HUMAN]["min"]
    out: dict[str, float] = {g: em.auroc(human, table[g]["min"]) for g in generators}
    for name, members in pools.items():
        out[name] = em.auroc(human, np.concatenate([table[g]["min"] for g in members]))
    out["flagged"] = tpr_at_fpr(table, generators, fpr)
    sources = np.array([row["source"] for row in rows])
    out["window"] = em.auroc(scores[sources == em.HUMAN], scores[np.isin(sources, generators)])
    return out, table


def log_det_standardization(model) -> float:
    """-Σ log feat_std.

    Added to the Gaussian's log p of the standardized features z, it gives the
    log density of the mix itself. The mix is the same space for every refit,
    so the same value means the same density in all of them.
    """
    return -float(model.feat_std.double().log().sum())


def share_below(table: dict, sources: list[str], threshold: float) -> float:
    """Share of the songs of `sources` whose score (min over windows) is below `threshold`."""
    return float(np.mean(np.concatenate([table[s]["min"] for s in sources]) < threshold))


def tpr_at_fpr(table: dict, generators: list[str], fpr: float) -> float:
    return share_below(table, generators, float(np.quantile(table[em.HUMAN]["min"], fpr)))


def fixed_threshold(base: dict, table: dict, base_offset: float, offset: float,
                    pools: dict[str, list[str]], fpr: float) -> dict:
    """The refit at the untrimmed refit's log-density threshold, against the untrimmed one at the same FPR."""
    threshold = float(np.quantile(base[em.HUMAN]["min"], fpr))  # untrimmed refit, units of its z
    moved = threshold + base_offset - offset  # the same log density, in the refit's units
    human_fpr = share_below(table, [em.HUMAN], moved)
    flagged_now = table[em.HUMAN]["min"] < moved
    flagged_before = base[em.HUMAN]["min"] < threshold
    songs = table[em.HUMAN]["songs"]
    return {
        "fpr": human_fpr,
        "tpr": {name: share_below(table, pool, moved) for name, pool in pools.items()},
        "untrimmed_tpr_same_fpr": {name: tpr_at_fpr(base, pool, human_fpr) for name, pool in pools.items()},
        "newly_flagged_human": [s for s, a, b in zip(songs, flagged_now, flagged_before) if a and not b],
        "no_longer_flagged_human": [s for s, a, b in zip(songs, flagged_now, flagged_before) if b and not a],
    }


def unit_members(units: list[str]) -> list[np.ndarray]:
    _, inverse = np.unique(np.array(units, dtype=str), return_inverse=True)
    return [np.flatnonzero(inverse == k) for k in range(int(inverse.max()) + 1)]


def paired_delta(base: dict, table: dict, members: list[np.ndarray], pools: dict[str, list[str]],
                 reps: int, seed: int) -> dict[str, list[float]]:
    """95 % interval of AUROC(table) - AUROC(base) per pool, both on the same resamples.

    The seed is the same for every call, so every threshold sees the same
    resamples as well.
    """
    rng = np.random.default_rng(seed)
    counts = {g: len(base[g]["min"]) for pool in pools.values() for g in pool}
    deltas: dict[str, list[float]] = {name: [] for name in pools}
    for _ in range(reps):
        h = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        picks = {g: rng.integers(0, n, n) for g, n in counts.items()}
        for name, pool in pools.items():
            values = []
            for t in (table, base):
                synthetic = np.concatenate([t[g]["min"][picks[g]] for g in pool])
                values.append(em.auroc(t[em.HUMAN]["min"][h], synthetic))
            deltas[name].append(values[0] - values[1])
    return {name: dd.interval(v) for name, v in deltas.items()}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    from train_mert_flow import load_mert_flow

    model, config, checkpoint = load_mert_flow(args.checkpoint, device)
    trained = parse_chain(checkpoint.get("codec"))
    ckpt_args = checkpoint.get("args", {})
    shrinkage = float(ckpt_args.get("gaussian_shrinkage", 0.01))
    stats_windows = int(ckpt_args.get("stats_windows", 0) or 0)

    train_root = args.train_cache or Path(checkpoint["train_cache"])
    train = PooledCache(train_root)
    if train.meta["model_name"] != config.mert.model_name:
        raise SystemExit(f"{train_root} was cached with {train.meta['model_name']}, the flow trained on "
                         f"{config.mert.model_name}")
    if float(train.meta["window_seconds"]) != float(config.data.window_seconds):
        raise SystemExit(f"{train_root} holds {train.meta['window_seconds']} s windows, the flow trained on "
                         f"{config.data.window_seconds} s")
    if parse_chain(train.meta.get("codec")) != trained:
        raise SystemExit(f"{train_root} is under codec {train.meta.get('codec')}, the flow trained under "
                         f"{chain_name(trained) or 'none'}: not the flow's training cache")
    if train.meta.get("split") != config.data.train_split:
        print(f"warning: {train_root} is split {train.meta.get('split')}, the flow trained on "
              f"{config.data.train_split}")

    eval_root = args.eval_cache or train_root.parent / (
        f"eval-{args.human_split}" + (f"-{chain_slug(trained)}" if trained else ""))
    if not (eval_root / "pooled.npy").exists():
        raise SystemExit(f"no evaluation cache at {eval_root}. Build it with scripts/evaluate_mert_flow.py "
                         f"--human-split {args.human_split}, or pass --eval-cache")
    cache = tm.EvalCache(eval_root, device)
    meta = cache.meta
    if meta["model_name"] != config.mert.model_name or \
            float(meta["window_seconds"]) != float(config.data.window_seconds):
        raise SystemExit(f"{eval_root} was cached with another model or window length than the flow")
    if parse_chain(meta.get("codec")) != trained:
        print("note: the evaluation cache's codec is not the flow's training chain")
    split = meta["human_split"]
    rows = cache.rows
    sources = np.array([row["source"] for row in rows])
    generators = [g for g in meta["generators"] if (sources == g).any()]
    pools = {"all": generators,
             "dev": [g for g in generators if g in args.dev],
             "others": [g for g in generators if g not in args.dev]}
    pools = {name: members for name, members in pools.items() if members}
    print(f"flow {args.checkpoint} (arm {checkpoint.get('arm')}, epoch {checkpoint.get('epoch')}); "
          f"Gaussian shrinkage s = {shrinkage:g}")
    print(f"train cache {train_root}: {len(train)} windows of {train.meta.get('split')}")
    print(f"eval cache {eval_root}: {len(rows)} windows, {split} vs. {', '.join(generators)}; "
          f"dev = {', '.join(pools.get('dev', [])) or 'none'}")
    if split == "test50":
        print("note: human = test50. Choosing a threshold from this run would be selection on test50")
    if stats_windows:
        print(f"note: the checkpoint's moments were fitted on the first {stats_windows} training windows; "
              "the refits here use all of them, so 'none' differs slightly from the checkpoint")

    # ---------------- τ² of the training windows ----------------
    axes_np, lam = dd.principal_axes(model)
    axes64 = torch.from_numpy(axes_np / np.sqrt(lam)).to(device)
    axes32 = axes64.float()
    z = training_features(train, model, device, args.batch_size)
    keys = np.array([row["key"] for row in train.rows()], dtype=str)
    if len(keys) != len(z):
        raise SystemExit(f"{len(z)} training features against {len(keys)} index rows")
    q = squared_norms(z, axes32)
    q_mean = float(q.mean())
    tau2 = q / q_mean
    rec_tau2, rec_tau2_rows = geometric_mean_per_unit(keys, tau2)
    works = {k.split("_")[0] for k in keys}
    print(f"\ntraining: {len(z)} windows, {len(rec_tau2)} recordings, {len(works)} works; "
          f"mean ‖w‖² = {q_mean:.1f} (in-sample)")
    print("training τ² quantiles, window:    " + ", ".join(f"{p:.1%} {np.quantile(tau2, p):.2f}" for p in QUANTILES))
    print("training τ² quantiles, recording: " + ", ".join(f"{p:.1%} {np.quantile(rec_tau2, p):.2f}" for p in QUANTILES))

    # ---------------- held-out human windows on the same scale ----------------
    ckpt_logp, q_eval = eval_terms(cache, model, axes32, args.batch_size)
    human = sources == em.HUMAN
    eval_tau2 = q_eval / q_mean
    human_songs = np.array([row["song"] for row in rows], dtype=str)[human]
    human_song_tau2, _ = geometric_mean_per_unit(human_songs, eval_tau2[human])
    print(f"held-out human ({split}): {int(human.sum())} windows, mean ‖w‖² = {q_eval[human].mean():.1f}, "
          f"i.e. mean τ² = {eval_tau2[human].mean():.2f} in training units")

    data_root = args.data_root or Path(meta.get("data_root", "."))
    work_of = emp.work_of(data_root, split)
    if work_of is None:
        print(f"note: no {split} manifest under {data_root}; works taken from the track-key prefix")

    # ---------------- untrimmed refit ----------------
    scratch = copy.deepcopy(model)
    keep_all = torch.ones(len(z), dtype=torch.bool, device=device)
    mean_all, cov_all = moments(z, keep_all)
    refit(scratch, model, mean_all, cov_all, shrinkage)
    base_scores = eval_logp(cache, scratch, args.batch_size)
    base_offset = log_det_standardization(scratch)
    base, base_table = detection(rows, base_scores, generators, pools, args.fpr)
    ckpt, _ = detection(rows, ckpt_logp, generators, pools, args.fpr)
    curve = {f: {name: tpr_at_fpr(base_table, pool, f) for name, pool in pools.items()} for f in CURVE}
    drift = float(np.abs(base_scores - ckpt_logp).max())
    print(f"check: refit on all windows vs. the checkpoint's Gaussian: max |Δ log p| = {drift:.3g} nats, "
          f"AUROC all {base['all']:.4f} vs. {ckpt['all']:.4f}")
    if drift > 1.0 and not stats_windows:
        print("warning: the refit does not reproduce the checkpoint's Gaussian; is this the flow's training cache?")
    human_units = [str((work_of or {}).get(song, song.split("_")[0]))
                   for song in base_table[em.HUMAN]["songs"]]
    members = unit_members(human_units)

    # ---------------- trims ----------------
    edges = np.linspace(0, len(lam), args.bands + 1).astype(int)
    configs = [{"level": "none", "threshold": None, "kept_windows": len(z), "kept_recordings": len(rec_tau2),
                "k": 1.0, "band_over_k": [1.0] * args.bands, "shift": 0.0, "spearman": 1.0,
                "held_out_above": 0.0, "auroc": base, "delta_ci95": {}}]
    for level in args.levels:
        stat = tau2 if level == "window" else rec_tau2_rows
        for t in sorted(args.thresholds):
            keep_np = stat <= t
            if keep_np.all():
                print(f"note: {level} t = {t:g} trims nothing")
            if keep_np.sum() <= z.shape[1]:
                print(f"skip: {level} t = {t:g} keeps {int(keep_np.sum())} windows, too few for a covariance")
                continue
            keep = torch.from_numpy(keep_np).to(device)
            mean_k, cov_k = moments(z, keep)
            refit(scratch, model, mean_k, cov_k, shrinkage)
            scores = eval_logp(cache, scratch, args.batch_size)
            offset = log_det_standardization(scratch)
            metrics, table = detection(rows, scores, generators, pools, args.fpr)
            above = (eval_tau2[human] > t) if level == "window" else (human_song_tau2 > t)
            configs.append({
                "level": level, "threshold": t,
                "kept_windows": int(keep_np.sum()),
                "kept_recordings": int(len(np.unique(keys[keep_np]))),
                **scale_diagnostics(mean_k, cov_k, mean_all, cov_all, axes64, edges),
                "spearman": spearman(scores, base_scores),
                "held_out_above": float(above.mean()),
                "auroc": metrics,
                "delta_ci95": paired_delta(base_table, table, members, pools, args.bootstrap, args.seed),
                "fixed_threshold": fixed_threshold(base_table, table, base_offset, offset, pools, args.fpr),
            })
            del keep, mean_k, cov_k

    # ---------------- report ----------------
    def label(c: dict) -> str:
        return "none" if c["threshold"] is None else f"{c['level']} {c['threshold']:g}"

    band_names = [f"{lo + 1}-{hi}" for lo, hi in zip(edges[:-1], edges[1:])]
    print("\n=== What each trim removed, and how far it is from a pure scale ===")
    print("kept = share of training windows / of recordings with any window kept; "
          "band/k = 1 everywhere for a pure scale; "
          "ρ = rank correlation with 'none'; >t = held-out human windows (songs, recording level) above t")
    print(f"{'trim':<16}{'kept win':>9}{'kept rec':>9}{'k':>7}  " + "".join(f"{b:>9}" for b in band_names)
          + f"{'shift':>8}{'ρ':>8}{'>t':>7}")
    for c in configs:
        print(f"{label(c):<16}{c['kept_windows'] / len(z):>9.3f}{c['kept_recordings'] / len(rec_tau2):>9.3f}"
              f"{c['k']:>7.3f}  " + "".join(f"{v:>9.3f}" for v in c["band_over_k"])
              + f"{c['shift']:>8.2f}{c['spearman']:>8.4f}{c['held_out_above']:>7.3f}")

    print(f"\n=== Detection (Gaussian, song = min over windows; {split} vs. MoM) ===")
    cols = generators + list(pools)
    header = [f"{g}*" if g in pools.get("dev", []) else g for g in generators] + list(pools)
    print("* = dev generator. window = pooled window-level AUROC; flagged = synthetic songs below the "
          f"{args.fpr:.0%} human quantile")
    print(f"{'trim':<16}" + "".join(f"{h:>11}" for h in header) + f"{'window':>9}{'flagged':>9}")
    print(f"{'checkpoint':<16}" + "".join(f"{ckpt[c]:>11.3f}" for c in cols)
          + f"{ckpt['window']:>9.3f}{ckpt['flagged']:>9.3f}")
    for c in configs:
        a = c["auroc"]
        print(f"{label(c):<16}" + "".join(f"{a[col]:>11.3f}" for col in cols)
              + f"{a['window']:>9.3f}{a['flagged']:>9.3f}")
    print(f"\nΔ AUROC against 'none', paired bootstrap 95 % ({args.bootstrap} resamples; human by work, "
          "generators by song)")
    print(f"{'trim':<16}" + "".join(f"{name:>26}" for name in pools))
    for c in configs[1:]:
        cells = []
        for name in pools:
            lo, hi = c["delta_ci95"][name]
            cells.append(f"{c['auroc'][name] - base[name]:+.3f} [{lo:+.3f},{hi:+.3f}]".rjust(26))
        print(f"{label(c):<16}" + "".join(cells))

    print("\n=== Operating points: synthetic songs flagged (TPR) against human songs flagged (FPR) ===")
    print("untrimmed refit, by moving its threshold (no trim): "
          + "; ".join(f"FPR {f:.0%} → TPR " + ", ".join(f"{name} {v:.3f}" for name, v in tprs.items())
                      for f, tprs in curve.items()))
    print(f"each trim at the untrimmed refit's {args.fpr:.0%}-FPR threshold, kept as a fixed log density of the "
          "mix. 'same FPR' = the untrimmed refit's TPR at the FPR the trim reaches; equal = the trim only "
          "moved the threshold")
    print(f"{'trim':<16}{'human FPR':>10}" + "".join(f"{'TPR ' + name:>12}{'same FPR':>10}" for name in pools)
          + f"{'human songs newly / no longer flagged':>40}")
    for c in configs[1:]:
        f = c["fixed_threshold"]
        print(f"{label(c):<16}{f['fpr']:>10.3f}"
              + "".join(f"{f['tpr'][name]:>12.3f}{f['untrimmed_tpr_same_fpr'][name]:>10.3f}" for name in pools)
              + f"{len(f['newly_flagged_human']):>32} / {len(f['no_longer_flagged_human'])}")
    print("(the newly flagged human keys are in the JSON; their titles and links in listen_tails.py's recordings.csv)")

    report = {
        "checkpoint": str(args.checkpoint), "arm": checkpoint.get("arm"), "epoch": checkpoint.get("epoch"),
        "train_cache": str(train_root), "eval_cache": str(eval_root), "human_split": split,
        "trained_codec": chain_name(trained), "shrinkage": shrinkage, "generators": generators, "pools": pools,
        "train": {"windows": len(z), "recordings": len(rec_tau2), "works": len(works), "mean_q": q_mean,
                  "tau2_quantiles": {str(p): float(np.quantile(tau2, p)) for p in QUANTILES},
                  "recording_tau2_quantiles": {str(p): float(np.quantile(rec_tau2, p)) for p in QUANTILES}},
        "held_out_human": {"windows": int(human.sum()), "mean_q": float(q_eval[human].mean())},
        "check": {"max_abs_logp": drift, "checkpoint_auroc": ckpt},
        "fpr": args.fpr, "untrimmed_curve": {str(f): tprs for f, tprs in curve.items()},
        "bands": band_names, "bootstrap": args.bootstrap, "seed": args.seed,
        "configs": configs,
    }
    out = args.out or args.checkpoint.parent / f"trim_tau_{split}.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
