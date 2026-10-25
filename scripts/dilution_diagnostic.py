#!/usr/bin/env python3
"""Dilution diagnostic: is the flow's detection signal swamped or missing?

Why (experiments.md sections 13 and 14). On the pooled MERT features the flow
sees, a linear probe separates every MoM generator from human music almost
perfectly (0.999-1.000). The human-only flow reaches 0.77-0.94. Section 14
reads this as *swamped*: the information is there, but the likelihood adds it
to the noise of every direction along which human music varies for reasons
unrelated to being human. That reading rests on a toy (whitened human N(0, I)
in d dimensions, generator shifted by Δ) solved backwards from the observed
AUROCs. This script measures Δ directly, from caches only, and checks the toy
against the data.

Coordinates. Everything is computed where the Gaussian baseline works: the
checkpoint's standardized layer mix x, and the baseline's shrunk covariance
C = (1 - s) R + s I. Rotating onto C's eigenvectors (the human principal
components, PCs, in descending variance λ_i) and dividing by √λ_i gives
whitened coordinates w. The baseline's score is exactly const - ½‖w‖². The
training-set human mean is 0 by construction.

Shrinkage. The toy assumes every whitened direction carries one unit of human
noise (d = F = 1024). Under C, PC i carries r_i = λ_R / λ_C of it on the
training set: 1 for the large components, less where the floor s dominates.
The noise that dilutes is then d_eff = Σ r_i² rather than F, and every
prediction below uses r. "toy d=F" is the section 14 formula as written, for
comparison.

Part 1, per generator: Δ against the observed AUROC.
  Δ             Mahalanobis distance between the generator's mean and the
                human training mean under C. It is estimated without the
                upward bias of a plug-in mean: an unbiased U-statistic over
                pairs of distinct songs (works, for human music), so the four
                correlated windows of one song do not count as four draws.
  probe(Δ)      Φ(Δ/√2), the best linear probe for that shift, to set against
                section 13's diagonal.
  pred          the window-level AUROC of the Gaussian if the generator were
                the human training distribution shifted by Δ (normal
                approximation, Gaussian human noise r).
  observed      the Gaussian's window-level AUROC on the same windows:
                held-out human vs. the generator. Window level because the toy
                predicts windows. Song-level AUROCs (min and mean over windows,
                as evaluate_mom.py reports) are printed separately.
  obs - pred    with a bootstrap 95 % interval: human resampled by work,
                generator by song, and Δ re-estimated in every resample.
  The moments of ‖w‖² say why observed and pred differ:
    mean - Σr ≈ Δ² + spread, up to the sampling noise of the mean.
    spread      Σ_i (variance along PC i - r_i): how much wider (> 0) or more
                concentrated (< 0) the source is than training human music,
                summed over the PCs. The toy assumes 0. For the held-out human
                row, Δ² is held-out human music's own offset from the
                training mean, and the toy assumes that is 0 too.
    sd          against the sd a pure shift would give. A wider human sd means
                human windows are unusual in many directions at once, which
                dilutes more.
    pred(moments)  Φ(gap / √(var_H + var_G)) from the empirical moments: the
                normal approximation with none of the toy's assumptions.

Part 2, per generator: Δ² split over the human PCs.
  c_i = (projection of the generator's mean on PC i)² / λ_i, which sums to Δ².
  Each c_i is a U-statistic too, so a component that carries nothing averages
  0 and can come out slightly negative.
  concentration   how many PCs, ranked by their own share of Δ², hold 50 % and
                  90 % of it. Ranking by a noisy estimate overstates
                  concentration, so the ranking is cross-fitted: songs are
                  split into two halves, one half ranks the PCs, the other
                  measures their share, and then the other way round. An even
                  spread needs about F/2 and 0.9 F.
  bands           equal-size groups of PCs, in order of human variance: share
                  of human variance, share of Δ², spread inside the band (the
                  generator and held-out human music), and the Gaussian
                  restricted to the band (pred and observed AUROC).
  top-k / minor-k the same for the k largest-variance PCs (a density on the
                  PCA projection) and the k smallest (the PCA-residual score).
                  "even" = k / F, the share an evenly spread shift would put
                  in the top k.

Reading (section 14, "Diagnosing it").
  * The obs - pred interval contains 0: dilution of a pure shift explains the
    Gaussian's AUROC, and swamped is confirmed for that generator.
  * Above: the generator also differs in spread (spread > 0). The density sees
    that and a probe does not.
  * Below: the generator is more concentrated than human music (spread < 0),
    so part of the problem is the one-sided likelihood (consider typicality
    scores). Or held-out human music sits farther from the model, or spreads
    wider, than the toy assumes (the human row).
  * Δ² in a few high-variance PCs: a density on the top-k projection should
    beat the full one (the top-k observed column), and any compression that
    keeps the largest variation also keeps the signal.
  * Δ² in low-variance PCs: anything trained to reconstruct drops it. Watch
    the minor-k column.
  * Δ² spread over hundreds of PCs (concentration near the even reference): no
    projection helps. Removing the nuisance then needs outside knowledge of
    what it is (covers, proposal 1).

Selection. Part 2 uses synthetic labels to locate the shift. That is fine for
diagnosis. A k or a band picked from these tables is a detector choice fitted
to these generators: fix it on a dev generator and report the others
untouched (section 12, "Selection leakage"). For the same reason the default
human side is val50, not test50.

Input. A train_mert_flow.py checkpoint (the mix, the standardization and the
Gaussian) and an evaluation cache in track_mom.py's layout (pooled.npy,
windows.jsonl, meta.json) holding the human split and the generators. The
default cache is where train_mert_flow.py --track-metrics and
evaluate_mert_flow.py put theirs, {train cache}/../eval-{human-split}[-{chain}],
under the flow's training chain. Nothing here runs MERT. Build a missing
cache with evaluate_mert_flow.py --human-split ..., or pass --eval-cache.

Writes {checkpoint dir}/dilution_{human-split}[_{chain}].json (or --out). It
includes the per-PC arrays (λ, r, c and the variance ratios) for plotting.

Example:
    python scripts/dilution_diagnostic.py \\
        --checkpoint checkpoints/mert-flow-unfrozen/mert_flow_last.pt
    python scripts/dilution_diagnostic.py \\
        --checkpoint checkpoints/mert-flow-unfrozen/mert_flow_last.pt \\
        --human-split test50
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

import evaluate_mert_probe as emp  # noqa: E402
import evaluate_mom as em  # noqa: E402
import track_mom as tm  # noqa: E402
from src.data.codec import chain_name, chain_slug, parse_chain  # noqa: E402
from src.training import pick_device  # noqa: E402

NORMAL = NormalDist()
FRACTIONS = (0.5, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="train_mert_flow.py checkpoint (mert_flow_last.pt)")
    parser.add_argument("--eval-cache", type=Path, default=None,
                        help="evaluation cache directory (default: "
                             "{train cache}/../eval-{human-split}[-{training chain}])")
    parser.add_argument("--human-split", default="val50",
                        help="human split of the default cache (default: val50; see Selection)")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="holds the human split's manifest, which gives the works "
                             "(default: the cache's data_root)")
    parser.add_argument("--bands", type=int, default=8, help="equal-size groups of PCs")
    parser.add_argument("--bootstrap", type=int, default=1000, help="resamples for the intervals")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap and cross-fitting seed")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


# --------------------------------------------------------------------------
# Whitened coordinates
# --------------------------------------------------------------------------


def principal_axes(model) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvectors (F, F) and eigenvalues (F,) of the Gaussian's covariance C, descending."""
    chol = model.gauss_chol.double().cpu()
    lam, axes = torch.linalg.eigh(chol @ chol.T)
    return axes.flip(1).numpy(), lam.flip(0).numpy()


def noise_ratios(lam: np.ndarray, shrinkage: float) -> np.ndarray:
    """r_i: training human variance along PC i, in whitened units.

    C = (1 - s) R + s I shares R's eigenvectors, with λ_R = (λ_C - s) / (1 - s).
    Along PC i the training windows have variance λ_R / λ_C after whitening:
    1 where λ_R is large, below 1 where the floor s dominates.
    """
    return np.clip(lam - shrinkage, 0.0, None) / ((1.0 - shrinkage) * lam)


@torch.no_grad()
def window_terms(cache: tm.EvalCache, model, rotation: np.ndarray, batch_size: int = 512):
    """Every cached window -> whitened coordinates (n, F), flow and Gaussian log p (n,)."""
    w, flow, gauss = [], [], []
    for chunk in cache.pooled.split(batch_size):
        x = model.features(chunk.float())
        flow.append(model.flow_log_prob(x).double().cpu().numpy())
        gauss.append(model.gaussian_log_prob(x).double().cpu().numpy())
        w.append(x.double().cpu().numpy() @ rotation)
    return np.concatenate(w), np.concatenate(flow), np.concatenate(gauss)


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------


def unit_means(w: np.ndarray, inverse: np.ndarray, n_units: int) -> np.ndarray:
    """Mean whitened vector of each unit (song or work) -> (n_units, F)."""
    onehot = np.zeros((n_units, len(inverse)))
    onehot[inverse, np.arange(len(inverse))] = 1.0
    return (onehot @ w) / onehot.sum(1, keepdims=True)


def mean_square(means: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Per-component unbiased estimate of μ_i², from unit means.

    The average of a_i b_i over pairs of distinct units a ≠ b, whose
    expectation is μ_i² because distinct units are independent. `weights` are
    bootstrap multiplicities; pairs of two copies of one unit are left out.
    """
    k = np.ones(len(means)) if weights is None else weights
    total = k @ means
    same = (k**2) @ (means**2)
    pairs = k.sum() ** 2 - (k**2).sum()
    return (total**2 - same) / pairs


def shift_auroc(delta2: float, r: np.ndarray, c: np.ndarray) -> float:
    """Window AUROC of ‖w‖² for a pure shift with per-PC components c.

    Human training ‖w‖² (Gaussian): mean Σr, variance 2Σr². Shifted: mean
    Σr + Δ², variance 2Σr² + 4Σ r c. With r = 1 this is the toy's
    Φ(Δ² / (2√(d + Δ²))).
    """
    var = 4.0 * float((r**2).sum()) + 4.0 * max(float(r @ c), 0.0)
    return NORMAL.cdf(max(delta2, 0.0) / math.sqrt(max(var, 1e-12)))


def toy_auroc(delta2: float, d: int) -> float:
    delta2 = max(delta2, 0.0)
    return NORMAL.cdf(delta2 / (2.0 * math.sqrt(d + delta2)))


def probe_auroc(delta2: float) -> float:
    return NORMAL.cdf(math.sqrt(max(delta2, 0.0)) / math.sqrt(2.0))


def members(inverse: np.ndarray, idx: np.ndarray) -> list[np.ndarray]:
    """Window indices of each unit."""
    order = np.argsort(inverse, kind="stable")
    bounds = np.cumsum(np.bincount(inverse))[:-1]
    return np.split(idx[order], bounds)


def interval(values: list[float]) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def bootstrap(q: np.ndarray, human: dict, gen: dict, r: np.ndarray, reps: int,
              rng: np.random.Generator) -> dict:
    """95 % intervals of the observed AUROC, pred, and observed - pred.

    Human units (works) and generator units (songs) are resampled; Δ and its
    components are re-estimated from the resampled songs every time.
    """
    h_members = members(human["inverse"], human["idx"])
    g_members = members(gen["inverse"], gen["idx"])
    observed, predicted, gap = [], [], []
    for _ in range(reps):
        h = np.concatenate([h_members[u] for u in rng.integers(0, len(h_members), len(h_members))])
        picked = rng.integers(0, len(g_members), len(g_members))
        g = np.concatenate([g_members[u] for u in picked])
        c = mean_square(gen["means"], np.bincount(picked, minlength=len(g_members)).astype(np.float64))
        o, p = em.auroc(-q[h], -q[g]), shift_auroc(float(c.sum()), r, c)
        observed.append(o)
        predicted.append(p)
        gap.append(o - p)
    return {"observed": interval(observed), "pred": interval(predicted), "gap": interval(gap)}


def concentration(means: np.ndarray, rng: np.random.Generator) -> dict[str, int | None]:
    """PCs needed for each fraction of Δ², ranked by their own share, cross-fitted."""
    order = rng.permutation(len(means))
    half = len(means) // 2
    c = [mean_square(means[order[:half]]), mean_square(means[order[half:]])]
    curves = []
    for rank_by, measure in ((c[0], c[1]), (c[1], c[0])):
        ranked = measure[np.argsort(-rank_by)]
        if ranked.sum() <= 0:
            return {f"k{round(f * 100)}": None for f in FRACTIONS}
        curves.append(np.cumsum(ranked) / ranked.sum())
    curve = np.mean(curves, axis=0)
    return {f"k{round(f * 100)}": int(np.argmax(curve >= f)) + 1 if (curve >= f).any() else len(curve)
            for f in FRACTIONS}


def reading(gap: list[float]) -> str:
    if gap[0] > 0:
        return "above"
    if gap[1] < 0:
        return "below"
    return "≈ dilution"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    from train_mert_flow import load_mert_flow

    model, config, checkpoint = load_mert_flow(args.checkpoint, device)
    trained = parse_chain(checkpoint.get("codec"))
    root = args.eval_cache or Path(checkpoint["train_cache"]).parent / (
        f"eval-{args.human_split}" + (f"-{chain_slug(trained)}" if trained else ""))
    if not (root / "pooled.npy").exists():
        raise SystemExit(f"no evaluation cache at {root}. Build it with scripts/evaluate_mert_flow.py "
                         f"--human-split {args.human_split}, or pass --eval-cache")
    cache = tm.EvalCache(root, device)
    meta = cache.meta
    if meta["model_name"] != config.mert.model_name:
        raise SystemExit(f"{root} was cached with {meta['model_name']}, the flow trained on "
                         f"{config.mert.model_name}")
    if float(meta["window_seconds"]) != float(config.data.window_seconds):
        raise SystemExit(f"{root} holds {meta['window_seconds']} s windows, the flow trained on "
                         f"{config.data.window_seconds} s")
    scored = parse_chain(meta.get("codec"))
    split = meta["human_split"]
    print(f"flow {args.checkpoint} (arm {checkpoint.get('arm')}, epoch {checkpoint.get('epoch')}, "
          f"trained on {chain_name(trained) or 'original audio'})")
    print(f"cache {root}: {len(cache.rows)} windows, {split} vs. {', '.join(meta['generators'])}, "
          f"codec {chain_name(scored) or 'none (original audio)'}")
    if scored != trained:
        print("note: the cache's codec is not the flow's training chain, so human windows are scored "
              "off the training distribution too")
    if split == "test50":
        print("note: human = test50. Fine for measuring; a k or band chosen from Part 2 would be "
              "selection on test50")

    shrinkage = float(checkpoint.get("args", {}).get("gaussian_shrinkage", 0.01))
    axes, lam = principal_axes(model)
    r = noise_ratios(lam, shrinkage)
    n_feat = len(lam)
    w, flow, gauss = window_terms(cache, model, axes / np.sqrt(lam))
    q = (w**2).sum(1)
    mismatch = float(np.abs(-0.5 * (n_feat * math.log(2 * math.pi) + np.log(lam).sum() + q) - gauss).max())
    if mismatch > 1.0:
        print(f"warning: ‖w‖² and the model's Gaussian disagree by up to {mismatch:.2f} nats")

    lam_r = np.clip(lam - shrinkage, 0.0, None) / (1.0 - shrinkage)
    var_share = np.cumsum(lam_r) / lam_r.sum()
    d_eff = float((r**2).sum())
    print(f"\nGaussian: F = {n_feat} features, shrinkage s = {shrinkage:g}. Human variance: 50 / 90 / 99 % "
          "in the top " + " / ".join(str(int(np.argmax(var_share >= f)) + 1) for f in (0.5, 0.9, 0.99))
          + f" PCs. Human training noise d_eff = Σr² = {d_eff:.0f} (the toy assumes {n_feat}); "
          f"training mean of ‖w‖² Σr = {r.sum():.0f}")

    # units: works for human music (several songs per work are not independent), songs otherwise
    data_root = args.data_root or Path(meta.get("data_root", "."))
    works = emp.work_of(data_root, split)
    if works is None:
        print(f"note: no {split} manifest under {data_root}; human songs stand in for works, "
              "so the intervals are too narrow")
    sources = np.array([row["source"] for row in cache.rows])
    views: dict[str, dict] = {}
    for source in [em.HUMAN, *meta["generators"]]:
        idx = np.flatnonzero(sources == source)
        if len(idx) == 0:
            print(f"warning: no {source} windows in the cache; left out")
            continue
        units = [cache.rows[i]["song"] for i in idx]
        if source == em.HUMAN and works is not None:
            units = [str(works.get(u, u)) for u in units]
        _, inverse = np.unique(np.array(units, dtype=str), return_inverse=True)
        n_units = int(inverse.max()) + 1
        means = unit_means(w[idx], inverse, n_units)
        views[source] = {"idx": idx, "inverse": inverse, "units": n_units, "means": means,
                         "c": mean_square(means), "var": w[idx].var(0),
                         "songs": len({cache.rows[i]["song"] for i in idx})}
    if em.HUMAN not in views:
        raise SystemExit(f"no {em.HUMAN} windows in {root}")
    generators = [g for g in meta["generators"] if g in views]
    human = views[em.HUMAN]
    h_idx = human["idx"]
    rng = np.random.default_rng(args.seed)

    # ---------------- Part 1 ----------------
    song_tables = {name: emp.song_scores(cache.rows, values, set())
                   for name, values in (("gaussian", -q), ("flow", flow))}
    results: dict[str, dict] = {}
    for g in generators:
        view = views[g]
        g_idx, c = view["idx"], view["c"]
        delta2 = float(c.sum())
        observed = em.auroc(-q[h_idx], -q[g_idx])
        pred = shift_auroc(delta2, r, c)
        gap = observed - pred
        intervals = bootstrap(q, human, view, r, args.bootstrap, rng)
        moments_pred = NORMAL.cdf((q[g_idx].mean() - q[h_idx].mean())
                                  / math.sqrt(q[g_idx].var() + q[h_idx].var()))
        results[g] = {
            "windows": len(g_idx), "songs": view["songs"],
            "delta": math.sqrt(max(delta2, 0.0)), "delta2": delta2,
            "probe_auroc": probe_auroc(delta2), "toy_auroc": toy_auroc(delta2, n_feat),
            "pred_auroc": pred, "observed_auroc": observed, "gap": gap,
            "ci95": intervals, "reading": reading(intervals["gap"]),
            "moments": {
                "mean": float(q[g_idx].mean()), "sd": float(q[g_idx].std()),
                "plug_in_shift": float((w[g_idx].mean(0) ** 2).sum()),
                "spread": float(view["var"].sum() - r.sum()),
                "sd_if_pure_shift": math.sqrt(2 * d_eff + 4 * max(float(r @ c), 0.0)),
                "pred_auroc": moments_pred,
            },
            "observed": {
                name: {"window": em.auroc((-q if name == "gaussian" else flow)[h_idx],
                                          (-q if name == "gaussian" else flow)[g_idx]),
                       **{f"song_{agg}": em.auroc(t[em.HUMAN][agg], t[g][agg]) for agg in ("min", "mean")}}
                for name, t in song_tables.items()
            },
        }
    human_delta2 = float(human["c"].sum())
    human_report = {
        "windows": len(h_idx), "songs": human["songs"], "units": human["units"],
        "unit": "work" if works is not None else "song",
        "mean": float(q[h_idx].mean()), "sd": float(q[h_idx].std()),
        "delta2": human_delta2, "delta": math.sqrt(max(human_delta2, 0.0)),
        "spread": float(human["var"].sum() - r.sum()),
        "train_mean": float(r.sum()), "train_sd": math.sqrt(2 * d_eff),
    }

    print("\n=== Part 1: Δ against the observed AUROC (Gaussian baseline, window level) ===")
    print(f"human = {split}, held out: {len(h_idx)} windows, {human['songs']} songs, "
          f"{human['units']} {human_report['unit']}s")
    print(f"{'generator':<12}{'windows':>8}{'songs':>7}{'Δ':>7}{'probe(Δ)':>10}{'toy d=F':>9}{'pred':>7}"
          f"{'observed':>10}{'obs - pred [95% CI]':>25}   reading")
    for g, res in results.items():
        lo, hi = res["ci95"]["gap"]
        print(f"{g:<12}{res['windows']:>8}{res['songs']:>7}{res['delta']:>7.2f}{res['probe_auroc']:>10.3f}"
              f"{res['toy_auroc']:>9.3f}{res['pred_auroc']:>7.3f}{res['observed_auroc']:>10.3f}"
              + f"{res['gap']:+.3f} [{lo:+.3f},{hi:+.3f}]".rjust(25) + f"   {res['reading']}")

    print(f"\nmoments of ‖w‖² (the Gaussian's score is const - ½‖w‖²). Training human music: "
          f"mean Σr = {r.sum():.0f}, sd √(2Σr²) = {math.sqrt(2 * d_eff):.1f} if Gaussian")
    print(f"{'source':<12}{'mean':>9}{'- Σr':>9}{'≈ Δ²':>9}{'+ spread':>10}{'sd':>8}"
          f"{'sd if shift':>13}{'pred(moments)':>15}")
    hr = human_report
    print(f"{'human':<12}{hr['mean']:>9.1f}{hr['mean'] - r.sum():>+9.1f}{hr['delta2']:>9.1f}"
          f"{hr['spread']:>+10.1f}{hr['sd']:>8.1f}{hr['train_sd']:>13.1f}{'—':>15}")
    for g, res in results.items():
        m = res["moments"]
        print(f"{g:<12}{m['mean']:>9.1f}{m['mean'] - r.sum():>+9.1f}{res['delta2']:>9.1f}"
              f"{m['spread']:>+10.1f}{m['sd']:>8.1f}{m['sd_if_pure_shift']:>13.1f}{m['pred_auroc']:>15.3f}")

    print("\nobserved AUROC (song = min / mean over its windows, as evaluate_mom.py)")
    print(f"{'generator':<12}{'Gaussian window':>16}{'song min':>10}{'song mean':>11}"
          f"{'flow window':>13}{'song min':>10}{'song mean':>11}")
    for g, res in results.items():
        o = res["observed"]
        print(f"{g:<12}{o['gaussian']['window']:>16.3f}{o['gaussian']['song_min']:>10.3f}"
              f"{o['gaussian']['song_mean']:>11.3f}{o['flow']['window']:>13.3f}"
              f"{o['flow']['song_min']:>10.3f}{o['flow']['song_mean']:>11.3f}")

    # ---------------- Part 2 ----------------
    cum = np.cumsum(w**2, axis=1)

    def subset(g_idx: np.ndarray, c: np.ndarray, lo: int, hi: int) -> dict:
        """The Gaussian restricted to PCs lo..hi-1: Δ² share, pred and observed AUROC."""
        score = cum[:, hi - 1] - (cum[:, lo - 1] if lo else 0.0)
        d2, total = float(c[lo:hi].sum()), float(c.sum())
        return {"pcs": [lo + 1, hi], "delta2": d2, "share": d2 / total if total > 0 else float("nan"),
                "pred_auroc": shift_auroc(d2, r[lo:hi], c[lo:hi]),
                "observed_auroc": em.auroc(-score[h_idx], -score[g_idx])}

    edges = np.linspace(0, n_feat, args.bands + 1).astype(int)
    ks = sorted({2**i for i in range(int(math.log2(n_feat)) + 1)} | {n_feat})
    print("\n=== Part 2: Δ² split over the human principal components (uses synthetic labels: "
          "diagnosis only) ===")
    for g in generators:
        view, res = views[g], results[g]
        g_idx, c = view["idx"], view["c"]
        res["concentration"] = concentration(view["means"], rng)
        res["bands"] = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            band = subset(g_idx, c, int(lo), int(hi))
            band["human_variance_share"] = float(lam_r[lo:hi].sum() / lam_r.sum())
            band["spread"] = float((view["var"][lo:hi] - r[lo:hi]).sum())
            band["human_spread"] = float((human["var"][lo:hi] - r[lo:hi]).sum())
            res["bands"].append(band)
        res["top_k"] = [dict(subset(g_idx, c, 0, k), k=k) for k in ks]
        res["minor_k"] = [dict(subset(g_idx, c, n_feat - k, n_feat), k=k) for k in ks]
        res["components"] = c.tolist()
        res["variance_ratio"] = view["var"].tolist()

        conc = res["concentration"]
        conc_text = ("not measurable (Δ² ≤ 0 in a half)" if conc["k50"] is None else
                     f"50 % of it in {conc['k50']} PCs, 90 % in {conc['k90']} "
                     f"(cross-fitted; an even spread needs ~{n_feat // 2} and ~{int(0.9 * n_feat)})")
        print(f"\n--- {g}: Δ² = {res['delta2']:.1f} over {n_feat} PCs; {conc_text} ---")
        print(f"{'PCs':<12}{'human var':>10}{'Δ² share':>10}{'spread':>9}{'human spread':>14}"
              f"{'pred':>7}{'observed':>10}")
        for band in res["bands"]:
            pcs = f"{band['pcs'][0]}-{band['pcs'][1]}"
            print(f"{pcs:<12}{band['human_variance_share']:>10.3f}{band['share']:>10.3f}"
                  f"{band['spread']:>+9.1f}{band['human_spread']:>+14.1f}"
                  f"{band['pred_auroc']:>7.3f}{band['observed_auroc']:>10.3f}")
        print(f"{'k':>6}{'top: Δ² share':>15}{'even':>7}{'pred':>7}{'observed':>10}"
              f"{'minor: Δ² share':>18}{'pred':>7}{'observed':>10}")
        for top, minor in zip(res["top_k"], res["minor_k"]):
            print(f"{top['k']:>6}{top['share']:>15.3f}{top['k'] / n_feat:>7.3f}{top['pred_auroc']:>7.3f}"
                  f"{top['observed_auroc']:>10.3f}{minor['share']:>18.3f}{minor['pred_auroc']:>7.3f}"
                  f"{minor['observed_auroc']:>10.3f}")

    report = {
        "checkpoint": str(args.checkpoint), "arm": checkpoint.get("arm"), "epoch": checkpoint.get("epoch"),
        "trained_codec": chain_name(trained), "eval_cache": str(root), "codec": chain_name(scored),
        "human_split": split, "features": n_feat, "shrinkage": shrinkage, "d_eff": d_eff,
        "gaussian_check_max_abs_nats": mismatch, "bootstrap": args.bootstrap, "seed": args.seed,
        "eigenvalues": lam.tolist(), "noise_ratio": r.tolist(),
        "human": dict(human_report, components=human["c"].tolist(), variance_ratio=human["var"].tolist()),
        "generators": results,
    }
    suffix = f"_{chain_slug(scored)}" if scored else ""
    out = args.out or args.checkpoint.parent / f"dilution_{split}{suffix}.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
