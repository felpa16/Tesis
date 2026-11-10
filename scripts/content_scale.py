#!/usr/bin/env python3
"""Content-conditional scale: score a song's atypicality against what its content predicts.

Why (experiments.md section 21). The human-only density ranks windows by
‖w‖² = τ² S, and τ is mostly human music of rare styles (section 20). 36 % of
the variance of log τ² lies between works (section 17): every cover of
*Deathcrush* is far from the centre. A density that expects that, given the
content, forgives metal for sounding like metal. This is p(style | content) to
first order, the thesis's own hypothesis, with the cheapest possible
conditioner: x | c ~ N(0, τ̂²(c) C), the checkpoint's Gaussian with a scale
predicted from content. A conditional mean would not do: τ is far from the
centre in every direction at once, so only a predicted scale can absorb it.

What.
  1. Training windows go through the checkpoint's mix and Gaussian, as in
     trim_tau.py: standardized z, whitened w, τ² = ‖w‖² / training mean.
  2. Content directions: an LDA with the training works as classes, in the
     whitened coordinates. Direction j has η_j, the share of its variance that
     lies between works; the rest lies between covers of one work and between
     windows. The top --dims directions, scaled to unit within-work variance,
     give c.
  3. Scale-free content: c / ‖w‖ per window, averaged over the song (a
     recording). Under the scale model w = τ g this does not depend on τ. A
     window's own atypicality cannot leak into what is expected of it; only
     the direction its content points in can.
  4. Expected atypicality: each training work is a centroid in that space,
     with the mean log τ² of its recordings as target. A song's log τ̂² is the
     mean target of its --k nearest works, calibrated by least squares on the
     training recordings with each recording's own work left out.
  5. Two scores per window, with τ̂² of its song, against the unconditional
     Gaussian ("none"):
       density  log N(x; 0, τ̂² C) = Gaussian − ½‖w‖² (1/τ̂² − 1) − (F/2) log τ̂²
       ratio    −‖w‖² / τ̂², atypicality relative to what the content predicts

Reported for every --dims × --k:
  η            between-work share of the first and the last direction kept
  b            calibration slope (0 = content predicts nothing)
  r² train     recordings, own work left out
  r² held out  val50 human songs. The ceiling is an oracle that predicts each
               song by the mean of the other songs of its work, on the same
               songs; the header prints it, and the training analogue too.
  τ̂ excuse     mean log τ̂² of each generator's songs minus that of human songs.
               Above 0, the content model expects that generator to be
               atypical and forgives part of its distance. Next to it, the
               actual difference in log τ², for scale.
  detection    as trim_tau.py: song-min AUROC per generator, pooled over all,
               the --dev generators and the others; pooled window AUROC;
               flagged at --fpr; Δ against "none" with a paired bootstrap 95 %
               interval (human by work, generators by song).

Prediction (section 21, written before the run).
  * The content directions recover less than half of the oracle's r² on
    val50.
  * ΔAUROC (song min, all) lies in [−0.01, +0.03] for every setting. None
    reaches section 17's ceiling of about +0.05.
Reading. A gain counts if it shows on the dev generators and the others alike
and comes with a held-out r² clearly above 0. The τ̂ excuse column says how
much of a generator's distance was forgiven along with the humans'.

Selection. --dims, --k and the score are chosen on the dev generators (default
riffusion and diffrhythm; section 12, "Selection leakage"), with human =
val50. test50 is scored once, afterwards.

Input. As trim_tau.py: a train_mert_flow.py checkpoint, its training cache
(work ids come from the track keys, "work_track") and section 16's evaluation
cache. Nothing here runs MERT.

Writes {checkpoint dir}/content_scale_{human-split}.json (or --out).

Example:
    python scripts/content_scale.py \\
        --checkpoint checkpoints/mert-flow-unfrozen/mert_flow_last.pt \\
        --train-cache $CACHE/train --eval-cache $CACHE/eval-val50
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

import dilution_diagnostic as dd  # noqa: E402
import evaluate_mert_probe as emp  # noqa: E402
import evaluate_mom as em  # noqa: E402
import track_mom as tm  # noqa: E402
import trim_tau as tt  # noqa: E402
from src.data.codec import chain_name, chain_slug, parse_chain  # noqa: E402
from src.data.pooled_cache import PooledCache  # noqa: E402
from src.training import pick_device  # noqa: E402

SCORES = ("density", "ratio")
FLOOR = 1e-3  # relative eigenvalue floor when inverting the total covariance for the LDA
QUERY_CHUNK = 4096


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
    parser.add_argument("--dims", type=int, nargs="+", default=[8, 32, 128],
                        help="numbers of LDA content directions to try")
    parser.add_argument("--k", type=int, nargs="+", default=[20],
                        help="numbers of nearest training works to average")
    parser.add_argument("--dev", nargs="+", default=list(tt.DEV),
                        help="generators a setting may be chosen on; the others are read untouched")
    parser.add_argument("--fpr", type=float, default=0.05, help="human FPR for the flagged share")
    parser.add_argument("--bootstrap", type=int, default=1000, help="paired resamples for the Δ intervals")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512, help="cache rows per forward")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------


@torch.no_grad()
def group_means(x: torch.Tensor, index: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Row means per group, fp64 (n_groups, d)."""
    sums = torch.zeros(n_groups, x.shape[1], dtype=torch.float64, device=x.device)
    for s in range(0, len(x), tt.CHUNK):
        sums.index_add_(0, index[s:s + tt.CHUNK], x[s:s + tt.CHUNK].double())
    counts = torch.bincount(index, minlength=n_groups).double().clamp_min(1.0)
    return sums / counts[:, None]


def numpy_group_means(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    return np.bincount(index, values) / np.bincount(index)


def leave_one_out_r2(values: np.ndarray, groups: np.ndarray) -> float:
    """r² of each value against the mean of the other members of its group (groups of ≥ 2)."""
    names, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, values)
    counts = np.bincount(inverse)
    ok = counts[inverse] >= 2
    pred = (sums[inverse][ok] - values[ok]) / (counts[inverse][ok] - 1)
    return r2(pred, values[ok])


def r2(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 3 or np.std(pred) == 0 or np.std(target) == 0:
        return float("nan")
    return float(np.corrcoef(pred, target)[0, 1] ** 2)


# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------


@torch.no_grad()
def lda(z: torch.Tensor, rotation: torch.Tensor, work_index: torch.Tensor, n_works: int, dims: int
        ) -> tuple[torch.Tensor, np.ndarray, float]:
    """Content projection (F, dims) applied to z, η of every direction, between-work share of the trace.

    Works in the whitened coordinates w = z @ rotation. With T the total and B
    the between-work covariance of w, the directions are the eigenvectors of
    T^-1/2 B T^-1/2, whose eigenvalues η are the between-work share of each
    direction's variance. Each kept direction is scaled to unit within-work
    variance (1 - η), so distances in c count within-work standard deviations.
    """
    everything = torch.ones(len(z), dtype=torch.bool, device=z.device)
    mean_z, cov_z = tt.moments(z, everything)
    total = rotation.T @ cov_z @ rotation
    counts = torch.bincount(work_index, minlength=n_works).double()
    centred = (group_means(z, work_index, n_works) - mean_z) @ rotation
    between = (centred * counts[:, None]).T @ centred / counts.sum()
    evals, evecs = torch.linalg.eigh(total)
    inv_sqrt = evecs @ torch.diag(evals.clamp_min(FLOOR * float(evals.max())).rsqrt()) @ evecs.T
    ratio = inv_sqrt @ between @ inv_sqrt
    eta, v = torch.linalg.eigh(0.5 * (ratio + ratio.T))
    eta, v = eta.flip(0), v.flip(1)
    within = (1.0 - eta[:dims]).clamp_min(1e-6)
    projection = rotation @ inv_sqrt @ v[:, :dims] / within.sqrt()
    share = float(torch.trace(between) / torch.trace(total))
    return projection, eta.cpu().numpy(), share


@torch.no_grad()
def scale_free_content(z: torch.Tensor, projection: torch.Tensor, q: np.ndarray) -> torch.Tensor:
    """(z @ projection) / ‖w‖ per row, fp32: the direction of a window's content, not its size."""
    c = torch.cat([(z[s:s + tt.CHUNK].double() @ projection).float() for s in range(0, len(z), tt.CHUNK)])
    return c / torch.from_numpy(np.sqrt(q)).float().to(c.device)[:, None]


@torch.no_grad()
def knn_mean(queries: torch.Tensor, centroids: torch.Tensor, targets: torch.Tensor, k: int,
             exclude: torch.Tensor | None) -> np.ndarray:
    """Mean target of each query's k nearest centroids; `exclude` = a centroid per query to skip."""
    out = []
    for s in range(0, len(queries), QUERY_CHUNK):
        d = torch.cdist(queries[s:s + QUERY_CHUNK].float(), centroids.float())
        if exclude is not None:
            d[torch.arange(len(d), device=d.device), exclude[s:s + QUERY_CHUNK]] = float("inf")
        nearest = d.topk(k, dim=1, largest=False).indices
        out.append(targets[nearest].mean(1))
    return torch.cat(out).double().cpu().numpy()


def calibrate(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Least-squares a, b in target ≈ a + b pred."""
    var = float(np.var(pred, ddof=1))
    b = float(np.cov(pred, target)[0, 1] / var) if var > 0 else 0.0
    return float(target.mean() - b * pred.mean()), b


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    from train_mert_flow import load_mert_flow

    model, config, checkpoint = load_mert_flow(args.checkpoint, device)
    trained = parse_chain(checkpoint.get("codec"))
    train_root = args.train_cache or Path(checkpoint["train_cache"])
    train = PooledCache(train_root)
    if train.meta["model_name"] != config.mert.model_name or \
            float(train.meta["window_seconds"]) != float(config.data.window_seconds) or \
            parse_chain(train.meta.get("codec")) != trained:
        raise SystemExit(f"{train_root} is not this flow's training cache (model, window length or codec differ)")
    eval_root = args.eval_cache or train_root.parent / (
        f"eval-{args.human_split}" + (f"-{chain_slug(trained)}" if trained else ""))
    if not (eval_root / "pooled.npy").exists():
        raise SystemExit(f"no evaluation cache at {eval_root}; pass --eval-cache (experiments.md section 20 "
                         "says where to find or how to rebuild it)")
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
    print(f"flow {args.checkpoint} (arm {checkpoint.get('arm')}, epoch {checkpoint.get('epoch')})")
    print(f"train cache {train_root}: {len(train)} windows; eval cache {eval_root}: {len(rows)} windows, "
          f"{split} vs. {', '.join(generators)}; dev = {', '.join(pools.get('dev', [])) or 'none'}")
    if split == "test50":
        print("note: human = test50. Choosing --dims, --k or the score from this run would be selection on test50")

    # ---------------- training: τ², works, content directions ----------------
    axes_np, lam = dd.principal_axes(model)
    axes64 = torch.from_numpy(axes_np / np.sqrt(lam)).to(device)
    axes32 = axes64.float()
    n_feat = len(lam)
    z = tt.training_features(train, model, device, args.batch_size)
    keys = np.array([row["key"] for row in train.rows()], dtype=str)
    if len(keys) != len(z):
        raise SystemExit(f"{len(z)} training features against {len(keys)} index rows")
    q = tt.squared_norms(z, axes32)
    q_mean = float(q.mean())
    log_tau2 = np.log(q / q_mean)

    rec_names, rec_index = np.unique(keys, return_inverse=True)
    work_names, rec_work = np.unique(np.array([k.split("_")[0] for k in rec_names]), return_inverse=True)
    window_work = torch.from_numpy(rec_work[rec_index]).to(device)
    dims = sorted({min(d, len(work_names) - 1, n_feat) for d in args.dims})
    projection, eta, between_share = lda(z, axes64, window_work, len(work_names), max(dims))
    content = scale_free_content(z, projection, q)
    del z
    rec_content = group_means(content, torch.from_numpy(rec_index).to(device), len(rec_names))
    del content
    rec_work_t = torch.from_numpy(rec_work).to(device)
    work_content = group_means(rec_content, rec_work_t, len(work_names))
    rec_log = numpy_group_means(log_tau2, rec_index)
    work_target = torch.from_numpy(numpy_group_means(rec_log, rec_work)).to(device)
    train_oracle = leave_one_out_r2(rec_log, rec_work)
    print(f"\ntraining: {len(keys)} windows, {len(rec_names)} recordings, {len(work_names)} works; "
          f"mean ‖w‖² = {q_mean:.1f}")
    print(f"LDA over works: {between_share:.3f} of the whitened variance lies between works; η of directions "
          + ", ".join(f"{i + 1}: {eta[i]:.3f}" for i in sorted({0, 7, 31, 127, max(dims) - 1}) if i < len(eta)))
    print(f"oracle r² (a recording's log τ² from the other recordings of its work): training {train_oracle:.3f}")

    # ---------------- evaluation windows and songs ----------------
    with torch.no_grad():
        z_eval = torch.cat([model.features(c.float()) for c in cache.pooled.split(args.batch_size)])
        base_scores = model.gaussian_log_prob(z_eval).double().cpu().numpy()
        q_eval = ((z_eval @ axes32) ** 2).sum(1).double().cpu().numpy()
    eval_content = scale_free_content(z_eval, projection, q_eval)
    song_keys = np.array([f"{row['source']}\t{row['song']}" for row in rows])
    song_names, song_index = np.unique(song_keys, return_inverse=True)
    song_source = np.array([name.split("\t")[0] for name in song_names])
    song_id = np.array([name.split("\t")[1] for name in song_names])
    song_content = group_means(eval_content, torch.from_numpy(song_index).to(device), len(song_names))
    song_log = numpy_group_means(np.log(q_eval / q_mean), song_index)
    human_song = song_source == em.HUMAN
    held_oracle = leave_one_out_r2(song_log[human_song],
                                   np.array([s.split("_")[0] for s in song_id[human_song]]))
    print(f"held out ({split}): {int(human_song.sum())} human songs; oracle r² {held_oracle:.3f} "
          "(the ceiling for 'r² held out')")
    actual = {g: float(song_log[song_source == g].mean() - song_log[human_song].mean()) for g in generators}

    base, base_table = tt.detection(rows, base_scores, generators, pools, args.fpr)
    data_root = Path(meta.get("data_root", "."))
    work_of = emp.work_of(data_root, split)
    members = tt.unit_members([str((work_of or {}).get(song, song.split("_")[0]))
                               for song in base_table[em.HUMAN]["songs"]])

    # ---------------- each setting ----------------
    configs = []
    for d in dims:
        for k in sorted({min(k, len(work_names) - 1) for k in args.k}):
            centroids = work_content[:, :d]
            raw_train = knn_mean(rec_content[:, :d], centroids, work_target, k, rec_work_t)
            a, b = calibrate(raw_train, rec_log)
            pred_songs = a + b * knn_mean(song_content[:, :d], centroids, work_target, k, None)
            log_hat = pred_songs[song_index]
            excuse = {g: float(pred_songs[song_source == g].mean() - pred_songs[human_song].mean())
                      for g in generators}
            scores = {
                "density": base_scores - 0.5 * q_eval * (np.exp(-log_hat) - 1.0) - 0.5 * n_feat * log_hat,
                "ratio": -q_eval * np.exp(-log_hat),
            }
            for name in SCORES:
                metrics, table = tt.detection(rows, scores[name], generators, pools, args.fpr)
                configs.append({
                    "dims": d, "k": k, "score": name, "eta_first": float(eta[0]), "eta_last": float(eta[d - 1]),
                    "a": a, "b": b, "r2_train": r2(a + b * raw_train, rec_log),
                    "r2_held_out": r2(pred_songs[human_song], song_log[human_song]),
                    "excuse": excuse, "auroc": metrics,
                    "delta_ci95": tt.paired_delta(base_table, table, members, pools, args.bootstrap, args.seed),
                })

    # ---------------- report ----------------
    def label(c: dict) -> str:
        return f"d={c['dims']} k={c['k']} {c['score']}"

    print("\n=== How well content predicts atypicality ===")
    print("b = calibration slope; r² train with each recording's own work left out; r² held out on "
          f"{split} human songs (oracle {held_oracle:.3f}); excuse = mean log τ̂² of a generator's songs minus "
          "the humans' (actual difference in log τ² in brackets)")
    print(f"{'setting':<14}{'η first':>9}{'η last':>8}{'b':>7}{'r² train':>10}{'r² held':>9}  "
          + "".join(f"{g:>20}" for g in generators))
    for c in configs:
        if c["score"] != SCORES[0]:
            continue
        print(f"{'d=' + str(c['dims']) + ' k=' + str(c['k']):<14}{c['eta_first']:>9.3f}{c['eta_last']:>8.3f}"
              f"{c['b']:>7.2f}{c['r2_train']:>10.3f}{c['r2_held_out']:>9.3f}  "
              + "".join(f"{c['excuse'][g]:+8.3f} [{actual[g]:+.3f}]".rjust(20) for g in generators))

    print(f"\n=== Detection (song = min over windows; {split} vs. MoM) ===")
    cols = generators + list(pools)
    header = [f"{g}*" if g in pools.get("dev", []) else g for g in generators] + list(pools)
    print("* = dev generator. window = pooled window-level AUROC; flagged = synthetic songs below the "
          f"{args.fpr:.0%} human quantile")
    print(f"{'setting':<24}" + "".join(f"{h:>11}" for h in header) + f"{'window':>9}{'flagged':>9}")
    print(f"{'none':<24}" + "".join(f"{base[col]:>11.3f}" for col in cols)
          + f"{base['window']:>9.3f}{base['flagged']:>9.3f}")
    for c in configs:
        m = c["auroc"]
        print(f"{label(c):<24}" + "".join(f"{m[col]:>11.3f}" for col in cols)
              + f"{m['window']:>9.3f}{m['flagged']:>9.3f}")
    print(f"\nΔ AUROC against 'none', paired bootstrap 95 % ({args.bootstrap} resamples; human by work, "
          "generators by song)")
    print(f"{'setting':<24}" + "".join(f"{name:>26}" for name in pools))
    for c in configs:
        cells = []
        for name in pools:
            lo, hi = c["delta_ci95"][name]
            cells.append(f"{c['auroc'][name] - base[name]:+.3f} [{lo:+.3f},{hi:+.3f}]".rjust(26))
        print(f"{label(c):<24}" + "".join(cells))

    report = {
        "checkpoint": str(args.checkpoint), "arm": checkpoint.get("arm"), "epoch": checkpoint.get("epoch"),
        "train_cache": str(train_root), "eval_cache": str(eval_root), "human_split": split,
        "trained_codec": chain_name(trained), "generators": generators, "pools": pools,
        "train": {"windows": len(keys), "recordings": len(rec_names), "works": len(work_names), "mean_q": q_mean,
                  "between_share": between_share, "eta": eta[:max(dims)].tolist(), "oracle_r2": train_oracle},
        "held_out": {"human_songs": int(human_song.sum()), "oracle_r2": held_oracle,
                     "actual_log_tau2_vs_human": actual},
        "none": base, "fpr": args.fpr, "bootstrap": args.bootstrap, "seed": args.seed,
        "configs": configs,
    }
    out = args.out or args.checkpoint.parent / f"content_scale_{split}.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
