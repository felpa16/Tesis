#!/usr/bin/env python3
"""Where the common factor lives: human variance of the window scores, by level.

Why (experiments.md section 16, "Next", item 1). The dilution diagnostic found
that what swamps the generators' shift is one common factor: how atypical a
human window is overall (‖w‖² = τ² S, τ² varying with a coefficient of
variation of ~0.54). Which lever removes it depends on where τ comes from.
SHS100K is nested, work → recording → window, so the human variance of
log ‖w‖² splits into three parts:
  works        between compositions: content. Conditioning on content
               (proposal 1, section 14) can remove this part.
  recordings   between covers of one work: performance and production (live,
               amateur, recording quality). Proposal 1 does not remove it, and
               the SHS100K-vs-MoM dataset confound lives here.
  windows      between windows of one recording: the passage (an intro, a
               quiet section). Window choice and the song aggregate act here;
               a mean over K windows divides this part by K.

Statistics, per window, in the Gaussian baseline's whitened coordinates w (as
dilution_diagnostic.py):
  log ‖w‖²      the factor itself: log τ² + log S, and log S hardly varies
                (sd ~0.05, against ~0.5 for log τ²).
  ‖w‖²          the Gaussian's score, up to -½ and a constant.
  flow -log p   the flow's score.
Shares do not depend on the sign or the constant, so the last two say where
each detector's own human noise sits.

Estimator. No grand mean is needed. For one window from each of two distinct
units, ½ E (x_a - x_b)² is the variance of every level at or below the one
that separates them:
  D_win    two windows of one recording       = windows
  D_rec    two recordings of one work         = windows + recordings
  D_work   two distinct works                 = windows + recordings + works
Each is averaged over pairs of distinct units, every unit weighted equally at
its own level. Under the nested random-effects model this is unbiased even
with unequal counts. The three parts are the successive differences, so a part
that is ~0 can come out slightly negative. Intervals: works resampled with
replacement, pairs of two copies of one work left out, the same resamples for
every statistic.

Window position. Each window is centred in one of K equal slices of the track,
never at second 0 or flush with the end (evaluate_mom.window_starts), so slot 1
is the early body of the track, not its intro. Over recordings with all K
windows: each slot's mean deviation from its recording's mean, and the share
of the within-recording variance the slot explains.

Input. A train_mert_flow.py checkpoint and a pooled-MERT cache of one human
split (cache_pooled_mert.py layout). The work is read off the track key,
{work}_{version}, as build_manifest.py does, so no manifest is needed. The
default is the checkpoint's --val-cache: val50, held out, every track of the
split. The training cache (--cache $CACHE/train) gives ~20× the works but is
in-sample: the Gaussian and the flow were fitted on those windows. Nothing
here runs MERT.

Reading.
  * works dominant: content. Proposal 1 has room.
  * recordings dominant: performance and production. Proposal 1 cannot remove
    it, and a production-matched human set (section 13) is the next test.
  * windows dominant: the passage. Look at window choice and the aggregate
    (section 15: mean over windows beat min by +0.022, not adopted).

Writes {checkpoint dir}/variance_levels_{split}[_{chain}].json (or --out).

Example:
    python scripts/variance_levels.py \\
        --checkpoint checkpoints/mert-flow-unfrozen/mert_flow_last.pt
    python scripts/variance_levels.py \\
        --checkpoint checkpoints/mert-flow-unfrozen/mert_flow_last.pt --cache $CACHE/train
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
from src.data.codec import chain_name, chain_slug, parse_chain  # noqa: E402
from src.data.pooled_cache import PooledCache  # noqa: E402
from src.training import pick_device  # noqa: E402

LEVELS = ("works", "recordings", "windows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="train_mert_flow.py checkpoint (mert_flow_last.pt)")
    parser.add_argument("--cache", type=Path, default=None,
                        help="pooled-MERT cache of one human split (default: the checkpoint's --val-cache)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap", type=int, default=1000, help="resamples of the works")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


@torch.no_grad()
def window_statistics(cache: PooledCache, model, rotation: np.ndarray,
                      batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    """Every cached window, in the order of cache.rows() -> {statistic: (n,)}."""
    q, flow = [], []
    for shard in cache.shards():
        for chunk in torch.from_numpy(shard).split(batch_size):
            x = model.features(chunk.to(device))
            q.append(((x.cpu().double().numpy() @ rotation) ** 2).sum(1))
            flow.append(model.flow_log_prob(x).cpu().double().numpy())
    q = np.concatenate(q)
    return {"log ‖w‖²": np.log(q), "‖w‖²": q, "flow -log p": -np.concatenate(flow)}


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------


def unit_moments(x: np.ndarray, inverse: np.ndarray, n_units: int):
    """Per unit: window count, mean and variance (ddof 0) of x."""
    n = np.bincount(inverse, minlength=n_units).astype(np.float64)
    mean = np.bincount(inverse, weights=x, minlength=n_units) / n
    var = np.bincount(inverse, weights=x * x, minlength=n_units) / n - mean**2
    return n, mean, np.clip(var, 0.0, None)


def pair_dispersion(mean: np.ndarray, var: np.ndarray, k: np.ndarray) -> float:
    """½ E (x_a - x_b)² for one window from each of two distinct units a ≠ b.

    Per pair it is ½ (v_a + v_b) + ½ (m_a - m_b)², with v the units' ddof-0
    variances. Pairs are averaged with weights k_a k_b (bootstrap
    multiplicities), so pairs of two copies of one unit are left out.
    """
    total = k.sum()
    km = k * mean
    within = (k * var) @ (total - k)
    between = (km * mean) @ (total - k) - (km.sum() ** 2 - (km**2).sum())
    return float((within + between) / (total**2 - (k**2).sum()))


def level_pieces(x: np.ndarray, rec: np.ndarray, n_rec: int, work_of_rec: np.ndarray, n_work: int) -> dict:
    """Per-work pieces of D_win, D_rec and D_work, so a bootstrap over works only reweights them."""
    n_r, m_r, v_r = unit_moments(x, rec, n_rec)
    _, m_w, v_w = unit_moments(x, work_of_rec[rec], n_work)
    multi = n_r >= 2
    sample_var = np.where(multi, v_r * n_r / np.maximum(n_r - 1, 1), 0.0)
    rec_disp = np.full(n_work, np.nan)
    for w, members in enumerate(dd.members(work_of_rec, np.arange(n_rec))):
        if len(members) >= 2:
            rec_disp[w] = pair_dispersion(m_r[members], v_r[members], np.ones(len(members)))
    return {
        "win_sum": np.bincount(work_of_rec, weights=sample_var, minlength=n_work),
        "win_count": np.bincount(work_of_rec, weights=multi.astype(np.float64), minlength=n_work),
        "rec_disp": rec_disp,
        "work_mean": m_w,
        "work_var": v_w,
    }


def components(pieces: dict, k: np.ndarray) -> dict[str, float]:
    """The three parts and their total, with works weighted by k."""
    d_win = float(k @ pieces["win_sum"] / (k @ pieces["win_count"]))
    has = ~np.isnan(pieces["rec_disp"])
    d_rec = float(k[has] @ pieces["rec_disp"][has] / k[has].sum())
    d_work = pair_dispersion(pieces["work_mean"], pieces["work_var"], k)
    return {"works": d_work - d_rec, "recordings": d_rec - d_win, "windows": d_win, "total": d_work}


def window_slots(rec: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Rank of each window's start inside its recording (0 = earliest)."""
    order = np.lexsort((starts, rec))
    sorted_rec = rec[order]
    slot = np.empty(len(rec), dtype=np.int64)
    slot[order] = np.arange(len(rec)) - np.searchsorted(sorted_rec, sorted_rec)
    return slot


def slot_effects(x: np.ndarray, rec: np.ndarray, n_rec: int, slot: np.ndarray, k_slots: int) -> dict | None:
    """Mean deviation of each slot from its recording's mean, over recordings with all K windows."""
    n_r, m_r, _ = unit_moments(x, rec, n_rec)
    full = n_r[rec] == k_slots
    if not full.any():
        return None
    dev, s = (x - m_r[rec])[full], slot[full]
    effects = np.array([dev[s == j].mean() for j in range(k_slots)])
    explained = float((np.bincount(s, minlength=k_slots) * effects**2).sum() / (dev**2).sum())
    return {"recordings": int(full.sum()) // k_slots, "effects": effects.tolist(),
            "share_of_within_recording": explained}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    from train_mert_flow import load_mert_flow

    model, config, checkpoint = load_mert_flow(args.checkpoint, device)
    root = args.cache or checkpoint.get("args", {}).get("val_cache")
    if not root or root == "None":
        raise SystemExit(f"{args.checkpoint} was trained without --val-cache; pass --cache")
    root = Path(root)
    cache = PooledCache(root)
    meta = cache.meta
    if meta["model_name"] != config.mert.model_name:
        raise SystemExit(f"{root} was cached with {meta['model_name']}, the flow trained on "
                         f"{config.mert.model_name}")
    if float(meta["window_seconds"]) != float(config.data.window_seconds):
        raise SystemExit(f"{root} holds {meta['window_seconds']} s windows, the flow trained on "
                         f"{config.data.window_seconds} s")
    trained, cached = parse_chain(checkpoint.get("codec")), parse_chain(meta.get("codec"))
    split = meta["split"]
    in_sample = root.resolve() == Path(checkpoint["train_cache"]).resolve()
    print(f"flow {args.checkpoint} (arm {checkpoint.get('arm')}, epoch {checkpoint.get('epoch')}, "
          f"trained on {chain_name(trained) or 'original audio'})")
    print(f"cache {root}: {len(cache)} windows of {split}, codec {chain_name(cached) or 'none (original audio)'}")
    if cached != trained:
        print("note: the cache's codec is not the flow's training chain, so these windows are scored "
              "off the training distribution")
    if in_sample:
        print("note: this is the flow's training cache. The Gaussian and the flow were fitted on these "
              "windows, so the variance is in-sample")

    rows = cache.rows()
    keys = np.array([row["key"] for row in rows], dtype=str)
    rec_keys, rec = np.unique(keys, return_inverse=True)
    _, work_of_rec = np.unique(np.array([k.split("_")[0] for k in rec_keys]), return_inverse=True)
    n_rec, n_work = len(rec_keys), int(work_of_rec.max()) + 1
    rec_per_work = np.bincount(work_of_rec, minlength=n_work)
    win_per_rec = np.bincount(rec, minlength=n_rec)
    k_slots = int(meta["windows_per_track"])
    slot = window_slots(rec, np.array([float(row["start"]) for row in rows]))

    axes, lam = dd.principal_axes(model)
    stats = window_statistics(cache, model, axes / np.sqrt(lam), args.batch_size, device)
    if len(stats["‖w‖²"]) != len(rows):
        raise SystemExit(f"{len(stats['‖w‖²'])} scored windows against {len(rows)} index rows")

    rng = np.random.default_rng(args.seed)
    draws = [np.bincount(rng.integers(0, n_work, n_work), minlength=n_work).astype(np.float64)
             for _ in range(args.bootstrap)]
    results: dict[str, dict] = {}
    for name, x in stats.items():
        x = x - x.mean()  # the shares do not move; the sums of squares stay well conditioned
        pieces = level_pieces(x, rec, n_rec, work_of_rec, n_work)
        point = components(pieces, np.ones(n_work))
        boot = [components(pieces, k) for k in draws]
        results[name] = {
            "variance": point,
            "share": {lv: point[lv] / point["total"] for lv in LEVELS},
            "ci95": {lv: dd.interval([b[lv] / b["total"] for b in boot]) for lv in LEVELS},
            "sample_variance": float(x.var(ddof=1)),
            "slots": slot_effects(x, rec, n_rec, slot, k_slots),
        }

    print(f"\nhuman = {split}: {len(rows)} windows, {n_rec} recordings, {n_work} works "
          f"({int((rec_per_work >= 2).sum())} with ≥ 2 recordings); recordings per work "
          f"median {int(np.median(rec_per_work))} (range {rec_per_work.min()}-{rec_per_work.max()}), "
          f"windows per recording {win_per_rec.mean():.2f}")
    print("\n=== Variance by level: share of the total [95 % CI, works resampled] ===")
    print(f"{'statistic':<14}" + "".join(f"{lv:>22}" for lv in LEVELS) + f"{'sd':>10}{'sd, plain':>11}")
    for name, res in results.items():
        cells = "".join(f"{res['share'][lv]:.3f} [{res['ci95'][lv][0]:.2f}, {res['ci95'][lv][1]:.2f}]".rjust(22)
                        for lv in LEVELS)
        print(f"{name:<14}{cells}{np.sqrt(max(res['variance']['total'], 0.0)):>10.4g}"
              f"{np.sqrt(res['sample_variance']):>11.4g}")
    print("sd: from D_work, every work weighted equally. sd, plain: over all windows.")

    print(f"\n=== Window position: mean deviation from the recording's mean, slot 1 = earliest "
          f"(recordings with all {k_slots} windows) ===")
    print(f"{'statistic':<14}" + "".join(f"{'slot ' + str(j + 1):>9}" for j in range(k_slots))
          + f"{'share of within-recording var':>32}")
    for name, res in results.items():
        s = res["slots"]
        if s is None:
            print(f"{name:<14}no recording has all {k_slots} windows")
            continue
        print(f"{name:<14}" + "".join(f"{e:>+9.3f}" for e in s["effects"])
              + f"{s['share_of_within_recording']:>32.3f}")

    report = {
        "checkpoint": str(args.checkpoint), "arm": checkpoint.get("arm"), "epoch": checkpoint.get("epoch"),
        "trained_codec": chain_name(trained), "cache": str(root), "codec": chain_name(cached),
        "split": split, "in_sample": in_sample, "bootstrap": args.bootstrap, "seed": args.seed,
        "windows": len(rows), "recordings": n_rec, "works": n_work,
        "works_with_2_recordings": int((rec_per_work >= 2).sum()), "windows_per_track": k_slots,
        "statistics": results,
    }
    suffix = f"_{chain_slug(cached)}" if cached else ""
    out = args.out or args.checkpoint.parent / f"variance_levels_{split}{suffix}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
