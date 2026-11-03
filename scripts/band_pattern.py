#!/usr/bin/env python3
"""Is the common factor a pure scale, and does a score that cancels it detect?

Why (experiments.md sections 16 and 17, items 2 and 3 of section 16's "Next").
The Gaussian baseline scores a window by ‖w‖², and what swamps the
generators' shift there is one common factor: how atypical a human window is
overall. As a scale mixture, w = τ g, every band of principal components
carries the same τ², so the band norms are near copies of one another
(section 16, points 3 and 5). Section 17 found τ spread over works,
recordings and windows alike, so no targeted fix removes most of it. A ratio
of band norms cancels τ wherever it comes from: τ² S_a / τ² S_b = S_a / S_b.
The generators differ in exactly that pattern: narrower than human music along
the top PCs, wider elsewhere (section 16, point 6).

Coordinates as in dilution_diagnostic.py: whitened w under the Gaussian
baseline, PCs in descending human variance, cut into K equal bands (default 8
of 128: section 16's bands). B_a = Σ_{i in a} w_i², Q = ‖w‖² = Σ_a B_a.

Part 1 (item 2): is the factor a pure scale? On human windows.
  Model: w_i = τ √v_i g_i, the g_i independent N(0, 1), τ independent of them,
  E τ² = 1, Var τ² = c². v_i is the measured mean of w_i² (r_i on the training
  windows). With μ_a = Σ_{i in a} v_i and q_a = Σ_{i in a} v_i²:
    Cov(B_a, B_b) = c² μ_a μ_b                            (a ≠ b)
    Var(B_a)      = c² μ_a² + (1 + c²) · 2 q_a
  c²            two estimates that agree under the model: from the
                off-diagonal, the mean over pairs of Cov(B_a, B_b) / (μ_a μ_b);
                and from the total, (Var Q - 2 Σv²) / ((Σv)² + 2 Σv²), as
                section 16 did.
  correlation   the observed K × K correlation of the band norms, against the
                model's c² μ_a μ_b / √(Var B_a · Var B_b) with the off-diagonal
                c² (section 16 predicted ≈ 0.93 for two bands of 128).
  per band      coupling: the mean over b ≠ a of Cov(B_a, B_b) / (μ_a μ_b),
                c² for every band under the model; a band with a factor of its
                own couples less. Excess: Var(B_a) / μ_a² minus the model's
                c² + (1 + c²) · 2 q_a / μ_a², 0 under the model. Cancelled: the
                share of Var(log B_a) that the ratio removes, 1 - Var(u_a) /
                Var(log B_a), with u as in Part 2; near 1 if τ is most of it.
  quantiles     of Q / E Q, which is τ² up to the noise of S (sd √(2Σv²) / Σv,
                ~0.05): observed; without a common factor (the toy); and for a
                lognormal τ² with the observed sd of log Q. An upper tail
                heavier than the lognormal's means a few windows atypical
                everywhere.
  Measured on the fit windows (training, in-sample for the whitening) and on
  the held-out human windows of the evaluation cache.

Part 2 (item 3): the pattern score.
  u_a = log B_a - mean_b log B_b, the centred log-ratio of the band norms:
  exactly τ-free, K - 1 degrees of freedom (u sums to 0).
  pattern   log N(u; m, S), a Gaussian fitted on the training windows' u. Any
            K - 1 coordinates give the same Mahalanobis distance. The score
            item 3 proposed, fixed in advance.
  joint     log N(log B; m, S) on all K log band norms: the pattern plus the
            common direction, weighted by its own (large) human variance. A
            DoSE-style multi-statistic test, reported for comparison.
  Gaussian  -½ ‖w‖², the baseline, and the flow, on the same windows. Their
            window-level rows reproduce section 16 on the same cache.
  AUROC per generator and pooled, for windows and for songs (min and mean over
  windows, as evaluate_mom.py). For the pooled song AUROC: a 95 % interval
  (human resampled by work, generators by song) and the share flagged at the
  5 % human threshold. Last, the mean u per band and source, in held-out human
  sds from held-out human music: where each generator's pattern departs.

Selection. The bands (K equal, in PC order) and the pattern score are fixed in
advance, so all four generators are scored untouched. The joint score and the
per-band tables are labelled reporting. Any change they suggest (the band
count, uneven edges, dropping a band) is a choice fitted to these generators:
make it on a dev generator and report the others untouched (section 12,
"Selection leakage"). For the same reason the default human side is val50.

Input. A train_mert_flow.py checkpoint; its training cache for the fit
(--train-cache, default the one it was trained on; --fit-windows samples it);
and an evaluation cache in track_mom.py's layout, located as
dilution_diagnostic.py locates it. The work of a human song is read off its
track key, {work}_{version}. Nothing here runs MERT.

Writes {checkpoint dir}/band_pattern_{human-split}[_{chain}].json (or --out).

Example:
    python scripts/band_pattern.py \\
        --checkpoint checkpoints/mert-flow-unfrozen/mert_flow_last.pt
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

import dilution_diagnostic as dd  # noqa: E402
import evaluate_mert_probe as emp  # noqa: E402
import evaluate_mom as em  # noqa: E402
import track_mom as tm  # noqa: E402
from src.data.codec import chain_name, chain_slug, parse_chain  # noqa: E402
from src.data.pooled_cache import PooledCache  # noqa: E402
from src.training import pick_device  # noqa: E402

NORMAL = NormalDist()
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
FPR = 0.05


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
    parser.add_argument("--train-cache", type=Path, default=None,
                        help="pooled-MERT cache the pattern Gaussians are fitted on "
                             "(default: the flow's training cache)")
    parser.add_argument("--fit-windows", type=int, default=50000,
                        help="training windows sampled for the fit (0 = all)")
    parser.add_argument("--bands", type=int, default=8, help="equal-size groups of PCs, fixed in advance")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap", type=int, default=1000, help="resamples for the intervals")
    parser.add_argument("--seed", type=int, default=0, help="fit sample and bootstrap seed")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


# --------------------------------------------------------------------------
# Band norms
# --------------------------------------------------------------------------


@torch.no_grad()
def band_terms(batches, model, rotation: np.ndarray, edges: np.ndarray, device: torch.device, flow: bool):
    """Pooled-layer batches -> band norms (n, K), flow log p (n,) or None, and Σ w_i² per PC (F,)."""
    bands, logp, total = [], [], np.zeros(rotation.shape[1])
    for chunk in batches:
        x = model.features(chunk.to(device).float())
        w2 = (x.cpu().double().numpy() @ rotation) ** 2
        bands.append(np.add.reduceat(w2, edges[:-1], axis=1))
        total += w2.sum(0)
        if flow:
            logp.append(model.flow_log_prob(x).cpu().double().numpy())
    return np.concatenate(bands), (np.concatenate(logp) if flow else None), total


def train_batches(cache: PooledCache, indices: np.ndarray, batch_size: int):
    for start in range(0, len(indices), batch_size):
        yield torch.from_numpy(cache.gather(indices[start : start + batch_size]))


def eval_batches(cache: tm.EvalCache, indices: np.ndarray, batch_size: int):
    rows = torch.as_tensor(indices, device=cache.pooled.device)
    yield from cache.pooled[rows].split(batch_size)


def clr(bands: np.ndarray) -> np.ndarray:
    """Centred log-ratio of the band norms: log B_a - mean_b log B_b, τ-free."""
    log_b = np.log(bands)
    return log_b - log_b.mean(1, keepdims=True)


# --------------------------------------------------------------------------
# Part 1: the scale-mixture check
# --------------------------------------------------------------------------


def scale_check(bands: np.ndarray, v: np.ndarray, edges: np.ndarray) -> dict:
    """The model's moments against the observed ones, on one set of human windows."""
    k = bands.shape[1]
    mu = bands.mean(0)
    q = np.add.reduceat(v**2, edges[:-1])
    cov = np.cov(bands, rowvar=False)
    norm = cov / np.outer(mu, mu)
    off = ~np.eye(k, dtype=bool)
    c2 = float(norm[off].mean())
    total = bands.sum(1)
    sv, sv2 = float(v.sum()), float((v**2).sum())
    c2_total = (float(total.var(ddof=1)) - 2 * sv2) / (sv**2 + 2 * sv2)
    var_model = c2 * mu**2 + (1 + c2) * 2 * q
    corr_model = c2 * np.outer(mu, mu) / np.sqrt(np.outer(var_model, var_model))
    np.fill_diagonal(corr_model, 1.0)
    corr = np.corrcoef(bands, rowvar=False)
    log_b = np.log(bands)
    s = float(np.log(total).std())
    z = np.array([NORMAL.inv_cdf(p) for p in QUANTILES])
    return {
        "windows": len(bands),
        "c2_off_diagonal": c2,
        "c2_pairs_range": [float(norm[off].min()), float(norm[off].max())],
        "c2_total": c2_total,
        "corr": corr.tolist(),
        "corr_model": corr_model.tolist(),
        "corr_off_diagonal": [float(corr[off].mean()), float(corr[off].min()), float(corr[off].max())],
        "corr_model_off_diagonal": [float(corr_model[off].mean()), float(corr_model[off].min()),
                                    float(corr_model[off].max())],
        "band_share": (mu / mu.sum()).tolist(),
        "coupling": ((norm * off).sum(1) / (k - 1)).tolist(),
        "excess": (np.diag(norm) - (c2 + (1 + c2) * 2 * q / mu**2)).tolist(),
        "cancelled": (1 - clr(bands).var(0) / log_b.var(0)).tolist(),
        "sd_log_q": s,
        "quantiles": {
            "p": list(QUANTILES),
            "observed": np.quantile(total / total.mean(), QUANTILES).tolist(),
            "no_factor": (1 + z * math.sqrt(2 * sv2) / sv).tolist(),
            "lognormal": np.exp(s * z - s**2 / 2).tolist(),
        },
    }


# --------------------------------------------------------------------------
# Part 2: the pattern score
# --------------------------------------------------------------------------


def fit_gaussian(x: np.ndarray) -> dict:
    cov = np.cov(x, rowvar=False)
    return {"mean": x.mean(0), "precision": np.linalg.inv(cov), "log_det": float(np.linalg.slogdet(cov)[1])}


def gaussian_log_prob(x: np.ndarray, g: dict) -> np.ndarray:
    d = x - g["mean"]
    maha = np.einsum("ni,ij,nj->n", d, g["precision"], d)
    return -0.5 * (x.shape[1] * math.log(2 * math.pi) + g["log_det"] + maha)


def detection(score: np.ndarray, rows: list[dict], sources: np.ndarray, generators: list[str],
              reps: int, rng: np.random.Generator) -> dict:
    """Window and song AUROCs of one score (higher = more human), per generator and pooled."""
    h = sources == em.HUMAN
    synthetic = np.isin(sources, generators)
    table = emp.song_scores(rows, score, set())
    human = table[em.HUMAN]
    works = [str(song).split("_")[0] for song in human["songs"]]
    out: dict = {"window": {g: em.auroc(score[h], score[sources == g]) for g in generators}}
    out["window"]["all"] = em.auroc(score[h], score[synthetic])
    for agg in ("min", "mean"):
        pooled = np.concatenate([table[g][agg] for g in generators])
        threshold = float(np.quantile(human[agg], FPR))
        out[f"song_{agg}"] = {g: em.auroc(human[agg], table[g][agg]) for g in generators}
        out[f"song_{agg}"]["all"] = em.auroc(human[agg], pooled)
        out[f"song_{agg}"]["ci95"] = emp.bootstrap_interval(human[agg], works, pooled, reps, rng)
        out[f"song_{agg}"]["flagged"] = float(np.mean(pooled < threshold))
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def print_check(name: str, check: dict) -> None:
    lo, hi = check["c2_pairs_range"]
    obs, model = check["corr_off_diagonal"], check["corr_model_off_diagonal"]
    print(f"\n--- {name}: {check['windows']} windows ---")
    print(f"c²: {check['c2_off_diagonal']:.3f} from the off-diagonal (pairs {lo:.3f} to {hi:.3f}), "
          f"{check['c2_total']:.3f} from the total (c = {math.sqrt(max(check['c2_off_diagonal'], 0)):.2f} "
          f"and {math.sqrt(max(check['c2_total'], 0)):.2f})")
    print(f"band-norm correlation, off-diagonal: observed {obs[0]:.3f} ({obs[1]:.3f} to {obs[2]:.3f}), "
          f"pure scale {model[0]:.3f} ({model[1]:.3f} to {model[2]:.3f})")
    print(f"{'band':<6}{'share of Q':>11}{'coupling':>10}{'excess':>9}{'cancelled':>11}")
    per_band = zip(check["band_share"], check["coupling"], check["excess"], check["cancelled"])
    for a, (share, coupling, excess, cancelled) in enumerate(per_band):
        print(f"{a + 1:<6}{share:>11.3f}{coupling:>10.3f}{excess:>+9.3f}{cancelled:>11.3f}")


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
    train_root = args.train_cache or Path(checkpoint["train_cache"])
    train = PooledCache(train_root)
    print(f"flow {args.checkpoint} (arm {checkpoint.get('arm')}, epoch {checkpoint.get('epoch')}, "
          f"trained on {chain_name(trained) or 'original audio'})")
    print(f"eval cache {root}: {len(cache.rows)} windows, {split} vs. {', '.join(meta['generators'])}, "
          f"codec {chain_name(scored) or 'none (original audio)'}")
    print(f"fit cache {train_root}: {len(train)} windows of {train.meta['split']}, codec "
          f"{chain_name(parse_chain(train.meta.get('codec'))) or 'none (original audio)'}")
    if scored != trained:
        print("note: the eval cache's codec is not the flow's training chain, so human windows are "
              "scored off the training distribution too")
    if split == "test50":
        print("note: human = test50. Fine for measuring; a band choice made from these tables would be "
              "selection on test50")

    axes, lam = dd.principal_axes(model)
    rotation = axes / np.sqrt(lam)
    n_feat = len(lam)
    if not 2 <= args.bands <= n_feat:
        raise SystemExit(f"--bands must lie in 2..{n_feat}")
    edges = np.linspace(0, n_feat, args.bands + 1).astype(int)

    rng = np.random.default_rng(args.seed)
    n_fit = len(train) if args.fit_windows <= 0 else min(args.fit_windows, len(train))
    fit_idx = np.sort(rng.choice(len(train), n_fit, replace=False))
    fit_bands, _, fit_v = band_terms(train_batches(train, fit_idx, args.batch_size),
                                     model, rotation, edges, device, flow=False)
    fit_v /= n_fit

    sources = np.array([row["source"] for row in cache.rows])
    generators = [g for g in meta["generators"] if (sources == g).any()]
    h_idx = np.flatnonzero(sources == em.HUMAN)
    g_idx = np.flatnonzero(np.isin(sources, generators))
    if len(h_idx) == 0 or len(g_idx) == 0:
        raise SystemExit(f"{root} needs both {em.HUMAN} and generator windows")
    bands = np.full((len(sources), args.bands), np.nan)
    flow = np.full(len(sources), np.nan)
    b, f, v = band_terms(eval_batches(cache, h_idx, args.batch_size), model, rotation, edges, device, flow=True)
    bands[h_idx], flow[h_idx], held_v = b, f, v / len(h_idx)
    b, f, _ = band_terms(eval_batches(cache, g_idx, args.batch_size), model, rotation, edges, device, flow=True)
    bands[g_idx], flow[g_idx] = b, f

    # ---------------- Part 1 ----------------
    checks = {"fit": scale_check(fit_bands, fit_v, edges), "held_out": scale_check(bands[h_idx], held_v, edges)}
    width = n_feat // args.bands
    print(f"\n=== Part 1: is the common factor a pure scale? ({args.bands} bands of {width} PCs) ===")
    print_check(f"fit windows ({train.meta['split']}, in-sample for the whitening)", checks["fit"])
    print_check(f"held-out human ({split})", checks["held_out"])
    print(f"\nobserved band-norm correlation, held-out human (pure scale predicts "
          f"{checks['held_out']['corr_model_off_diagonal'][0]:.2f} off the diagonal):")
    for line in checks["held_out"]["corr"]:
        print("  " + " ".join(f"{x:5.2f}" for x in line))
    print("\nquantiles of ‖w‖² / E‖w‖² (≈ τ²)")
    print(f"{'':<26}" + "".join(f"{p:>8.2f}" for p in QUANTILES))
    rows_q = (("observed, fit", checks["fit"]["quantiles"]["observed"]),
              (f"observed, {split}", checks["held_out"]["quantiles"]["observed"]),
              ("no common factor (toy)", checks["held_out"]["quantiles"]["no_factor"]),
              (f"lognormal, sd log = {checks['held_out']['sd_log_q']:.2f}",
               checks["held_out"]["quantiles"]["lognormal"]))
    for label, values in rows_q:
        print(f"{label:<26}" + "".join(f"{x:>8.3f}" for x in values))

    # ---------------- Part 2 ----------------
    fit_u = clr(fit_bands)
    pattern = fit_gaussian(fit_u[:, :-1])
    joint = fit_gaussian(np.log(fit_bands))
    u = clr(bands)
    scores = {
        "flow": flow,
        "Gaussian": -0.5 * bands.sum(1),
        "pattern": gaussian_log_prob(u[:, :-1], pattern),
        "joint": gaussian_log_prob(np.log(bands), joint),
    }
    results = {name: detection(s, cache.rows, sources, generators, args.bootstrap, rng)
               for name, s in scores.items()}

    print(f"\n=== Part 2: the pattern score (AUROC; human = {split}, held out) ===")
    header = f"{'score':<10}" + "".join(f"{g:>12}" for g in generators) + f"{'all':>8}"
    print("\nwindow level")
    print(header)
    for name, res in results.items():
        print(f"{name:<10}" + "".join(f"{res['window'][g]:>12.3f}" for g in generators)
              + f"{res['window']['all']:>8.3f}")
    for agg in ("min", "mean"):
        print(f"\nsong = {agg} over windows")
        print(header + f"{'[95% CI]':>16}{'flagged @ 5%':>14}")
        for name, res in results.items():
            r = res[f"song_{agg}"]
            print(f"{name:<10}" + "".join(f"{r[g]:>12.3f}" for g in generators) + f"{r['all']:>8.3f}"
                  + f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]".rjust(16) + f"{r['flagged']:>13.0%}")

    h_u = u[h_idx]
    h_mean, h_sd = h_u.mean(0), h_u.std(0)
    shifts = {"fit": ((fit_u.mean(0) - h_mean) / h_sd).tolist()}
    shifts.update({g: ((u[sources == g].mean(0) - h_mean) / h_sd).tolist() for g in generators})
    print(f"\nmean pattern u per band, in {split} human sds from {split} human music "
          "(labelled reporting: see Selection)")
    print(f"{'source':<12}" + "".join(f"{'b' + str(a + 1):>8}" for a in range(args.bands)))
    for source, values in shifts.items():
        label = "train (fit)" if source == "fit" else source
        print(f"{label:<12}" + "".join(f"{x:>+8.2f}" for x in values))

    report = {
        "checkpoint": str(args.checkpoint), "arm": checkpoint.get("arm"), "epoch": checkpoint.get("epoch"),
        "trained_codec": chain_name(trained), "eval_cache": str(root), "codec": chain_name(scored),
        "human_split": split, "fit_cache": str(train_root), "fit_windows": n_fit,
        "bands": args.bands, "edges": edges.tolist(), "bootstrap": args.bootstrap, "seed": args.seed,
        "scale_check": checks, "detection": results, "pattern_shift_sd": shifts,
    }
    suffix = f"_{chain_slug(scored)}" if scored else ""
    out = args.out or args.checkpoint.parent / f"band_pattern_{split}{suffix}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
