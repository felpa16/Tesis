#!/usr/bin/env python3
"""Condition the human Gaussian on chroma: does content that is blind to style help?

Why (experiments.md sections 14, 21 and 22). The thesis bets that a human-only
density scores synthetic music better as p(style | content) than as p(window).
Section 21 tried the cheapest version with "content" taken from pooled MERT
itself (LDA directions over works). That content turned out to be style: it
predicted the generators' atypicality along with the humans', and the ratio
score lost 0.15 AUROC. Section 22 found the learned content encoder's training
pairs often do not share content. This script takes content that is blind to
timbre and production by construction, beat-synchronous chroma, and asks the
question without any learned encoder. If conditioning on it does not help,
better pairs and a retrained content encoder will not help the detector either.

What.
  1. Chroma. Training tracks: the archive align_covers.py wrote
     (--chroma-root/{train,val,test}/{key}.npz; $RUN_S3/preprocessing/chroma.tgz).
     Evaluation songs (val50 humans and MoM): computed here with the same
     function (align_covers.beat_sync_chroma, 22.05 kHz, --max-seconds), so
     human and synthetic songs go through identical code. --human-chroma
     archive takes the humans from the archive instead (no audio needed); then
     the two classes may differ by library version, so keep "fresh" for a run
     that counts. With fresh chroma, the human songs that are also in the
     archive are compared with it (the "chroma check" line).
  2. Descriptors of each window (the beats whose mid-time lies in it):
       abs   mean chroma (12) and its second moment (78 upper-triangle terms):
             which pitch classes sound, and which sound together
       rel   the same after rotating to the track's tonic (Krumhansl-Kessler
             key profiles on the whole track's mean chroma), plus the mode: 91
       dftm  2D Fourier transform magnitude of 16-beat chroma patches, median
             over the window's patches (Bertin-Mahieux & Ellis 2012). Invariant
             to transposition and to time shift; 12 x 9 = 108
       all   the three concatenated
     Each is standardized on the training windows.
  3. A joint Gaussian of (z, c), z the checkpoint's standardized layer mix,
     fitted on the training windows that have chroma, with the checkpoint's
     correlation shrinkage applied to the joint. Its z block is exactly the
     unconditional baseline ("none"), refitted on the same windows. Scores:
       mean        log p(z | c): z against the mean its content predicts, under
                   the residual covariance
       joint       log p(z | c) + log p(c) = log p(z, c), alpha = 1 of CLAUDE.md
       scale       section 21's density score with chroma as the content:
                   log N(z; mu, tau^2(c) Sigma), log tau^2 predicted from c by
                   ridge regression on the training windows
       mean+scale  both: the residual of "mean", with its own predicted scale
     and a diagnostic row "content" = log p(c) alone.

Reported per descriptor:
  between-work  share of c's standardized variance between training works. A
                content descriptor should be well above MERT's own (printed on
                the header line, in whitened units).
  explained     share of z's whitened variance the linear content model
                explains, in-sample. 0 = c says nothing about the window.
  PMI           mean log p(z | c) - log p(z) per source. Positive: knowing the
                content made the window more probable. The hypothesis predicts
                a larger PMI for human windows than for synthetic ones; the
                AUROC of "mean" moves with that gap.
  r2 scale      how well c predicts log tau^2: training windows; held-out human
                windows; held-out human songs (song means). Section 21's work
                oracle on the same songs is printed for scale (0.449 there).
  excuse        mean predicted log tau^2 of a generator's songs minus the
                humans', next to the actual difference: how much of each
                generator's distance the content model forgives.
  detection     as trim_tau.py: song-min AUROC per generator, pooled over all,
                over the --dev generators and over the others; pooled window
                AUROC; flagged at --fpr; delta against "none" with a paired
                bootstrap 95 % interval (human by work, generators by song).
Songs with a window that has no chroma (a failed decode, a window past
--max-seconds, under --min-beats beats) are dropped from every score alike, so
all rows compare the same songs. The header says how many.

Prediction (experiments.md section 23, written before the run).
  * between-work: rel and dftm above abs (covers change key), all three above
    MERT's.
  * explained: small, under 5 % of z's whitened variance.
  * PMI: positive for humans, with the generators within 20 % of the human
    value.
  * scale: song-level held-out r2 under 0.15, far below section 21's 0.45-0.56;
    excuses near 0.
  * No score gains more than +0.02 pooled AUROC with an interval above 0 on
    both the dev generators and the others. "content" alone is at or below 0.5.
Reading. A gain counts only if it shows on the dev generators and the others
alike and the PMI gap points the same way. If none does, the conditional
hypothesis is not supported with this content, and section 22's step 4
(re-mined pairs, a retrained encoder) is not worth it for the detector.

Selection. Descriptor and score are chosen on the --dev generators (default
riffusion and diffrhythm; section 12, "Selection leakage"), human = val50.
test50 is scored once, afterwards.

Input. A train_mert_flow.py checkpoint, its training cache (pooled-MERT
layout, ~10 GB, ~203 k windows; keys "work_track" and window starts), section
16's evaluation cache and the chroma archive. MoM audio streams from S3
(--bucket, --mom-prefix) or comes from --mom-root; human audio comes from
--data-root (or --human-audio-key). Nothing here runs MERT. The CPU stages run
before the GPU is touched, in --workers processes, and are cached in
--work-dir, so a re-run skips them:
  {work-dir}/train_descriptors_p{patch}.npz   training window descriptors
  {work-dir}/eval_chroma/{source}/{song}.npz  evaluation chroma

Writes {checkpoint dir}/chroma_condition_{human-split}.json (or --out).

Example:
    python scripts/chroma_condition.py \\
        --checkpoint checkpoints/mert-flow-unfrozen/mert_flow_last.pt \\
        --train-cache $CACHE/train --eval-cache $CACHE/eval-val50 \\
        --data-root $DATA --bucket $BUCKET
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

import align_covers as ac  # noqa: E402
import content_scale as cs  # noqa: E402
import evaluate_mert_probe as emp  # noqa: E402
import evaluate_mom as em  # noqa: E402
import track_mom as tm  # noqa: E402
import trim_tau as tt  # noqa: E402
from src.data.codec import chain_name, chain_slug, parse_chain  # noqa: E402
from src.data.pooled_cache import PooledCache  # noqa: E402

DESCRIPTORS = ("abs", "rel", "dftm", "all")
PARTS = {"abs": ("abs",), "rel": ("rel",), "dftm": ("dftm",), "all": ("abs", "rel", "dftm")}
SCORES = ("mean", "joint", "scale", "mean+scale")
CHUNK = 8192

# Krumhansl-Kessler key profiles, index 0 = tonic.
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
TRIU = np.triu_indices(12)


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
                        help="holds manifests/{human split}/ and the human audio (default: the eval cache's)")
    parser.add_argument("--chroma-root", type=Path, default=None,
                        help="the extracted chroma archive, with train/ val/ test/ (default: {data-root}/chroma)")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="cache of descriptors and evaluation chroma (default: {checkpoint dir}/chroma_condition)")
    parser.add_argument("--human-chroma", choices=("fresh", "archive"), default="fresh",
                        help="compute the human songs' chroma here (default), or take it from the archive")
    parser.add_argument("--human-audio-key", default=None,
                        help="S3 key template for human tracks missing from --data-root, as evaluate_mom.py")
    parser.add_argument("--bucket", default=os.environ.get("BUCKET", "songs-and-mert-features"))
    parser.add_argument("--region", default=None)
    parser.add_argument("--mom-prefix", default="mom", help="S3 prefix of the MoM mirror")
    parser.add_argument("--mom-root", type=Path, default=None, help="read MoM from this local directory instead")
    parser.add_argument("--tmp-dir", type=Path, default=None, help="where S3 audio is staged")
    parser.add_argument("--workers", type=int, default=6,
                        help="processes for the chroma stages (chroma is memory-hungry; ~1-2 GB each)")
    parser.add_argument("--max-seconds", type=float, default=600.0,
                        help="chroma of at most this many seconds per track (align_covers.py's default)")
    parser.add_argument("--patch-beats", type=int, default=16, help="beats per 2D-FTM patch")
    parser.add_argument("--min-beats", type=int, default=8, help="beats a window needs to have a descriptor")
    parser.add_argument("--descriptors", nargs="+", choices=DESCRIPTORS, default=list(DESCRIPTORS))
    parser.add_argument("--dev", nargs="+", default=list(tt.DEV),
                        help="generators a setting may be chosen on; the others are read untouched")
    parser.add_argument("--fpr", type=float, default=0.05, help="human FPR for the flagged share")
    parser.add_argument("--bootstrap", type=int, default=1000, help="paired resamples for the delta intervals")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512, help="cache rows per forward")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


# --------------------------------------------------------------------------
# Descriptors (numpy, CPU)
# --------------------------------------------------------------------------


def estimate_key(mean_chroma: np.ndarray) -> tuple[int, bool]:
    """(tonic pitch class, minor?) by correlation with the 24 rotated key profiles."""
    best, tonic, minor = -np.inf, 0, False
    if np.std(mean_chroma) == 0:
        return tonic, minor
    for is_minor, profile in ((False, MAJOR), (True, MINOR)):
        for t in range(12):
            r = float(np.corrcoef(mean_chroma, np.roll(profile, t))[0, 1])
            if r > best:
                best, tonic, minor = r, t, is_minor
    return tonic, minor


def pitch_class_stats(chroma: np.ndarray) -> np.ndarray:
    """Mean (12) and second moment (78) of a (12, T) beat-chroma window."""
    second = (chroma @ chroma.T) / chroma.shape[1]
    return np.concatenate([chroma.mean(1), second[TRIU]])


def dftm(chroma: np.ndarray, patch: int) -> np.ndarray:
    """Median 2D Fourier transform magnitude over the window's patches, (12 * (patch // 2 + 1),).

    The magnitude ignores circular shifts along pitch (transposition) and along
    time, and is symmetric, so half of the time-frequency axis is kept.
    """
    if chroma.shape[1] < patch:
        chroma = np.tile(chroma, (1, -(-patch // chroma.shape[1])))[:, :patch]
    patches = np.lib.stride_tricks.sliding_window_view(chroma, patch, axis=1)  # (12, n, patch)
    magnitude = np.abs(np.fft.fft2(patches.transpose(1, 0, 2), axes=(1, 2)))[:, :, : patch // 2 + 1]
    return np.median(magnitude, axis=0).ravel()


def dims(patch: int) -> dict[str, int]:
    return {"abs": 90, "rel": 91, "dftm": 12 * (patch // 2 + 1)}


def describe_track(chroma: np.ndarray, times: np.ndarray, starts: list[float], seconds: float,
                   patch: int, min_beats: int) -> dict:
    """Descriptors of each window [start, start + seconds) of one track."""
    tonic, minor = estimate_key(chroma.mean(1))
    size = dims(patch)
    out = {name: np.zeros((len(starts), d), dtype=np.float32) for name, d in size.items()}
    out["ok"] = np.zeros(len(starts), dtype=bool)
    out["tonic"], out["minor"] = tonic, minor
    for i, start in enumerate(starts):
        inside = (times >= start) & (times < start + seconds)
        if int(inside.sum()) < min_beats:
            continue
        window = chroma[:, inside].astype(np.float64)
        out["abs"][i] = pitch_class_stats(window)
        out["rel"][i, :90] = pitch_class_stats(np.roll(window, -tonic, axis=0))
        out["rel"][i, 90] = float(minor)
        out["dftm"][i] = dftm(window, patch)
        out["ok"][i] = True
    return out


def describe_file(task: tuple[str, list[float], float, int, int]) -> tuple[dict | None, str]:
    """Pool worker: (chroma .npz path, window starts, seconds, patch, min beats) -> descriptors."""
    path, starts, seconds, patch, min_beats = task
    try:
        with np.load(path) as data:
            chroma, times = data["chroma"], data["times"]
        return describe_track(chroma, times, starts, seconds, patch, min_beats), ""
    except Exception as exc:  # noqa: BLE001 - reported, the track is left out
        return None, f"{type(exc).__name__}: {exc}"


def archive_index(root: Path) -> dict[str, Path]:
    """Track key -> chroma file, over every split folder of the archive."""
    index: dict[str, Path] = {}
    for path in sorted(root.glob("*/*.npz")):
        index.setdefault(path.stem, path)
    return index


def empty_descriptors(n: int, patch: int) -> dict:
    out = {name: np.zeros((n, d), dtype=np.float32) for name, d in dims(patch).items()}
    out["ok"] = np.zeros(n, dtype=bool)
    out["minor"] = np.zeros(n, dtype=bool)
    return out


def describe_rows(tasks: list[tuple[str, list[int], list[float]]], n_rows: int, seconds: float, patch: int,
                  min_beats: int, workers: int, label: str) -> tuple[dict, list[str]]:
    """Descriptors for every row; tasks = (chroma path, row indices, their starts) per track."""
    out = empty_descriptors(n_rows, patch)
    errors: list[str] = []
    started = time.time()
    jobs = [(path, starts, seconds, patch, min_beats) for path, _, starts in tasks]
    with ProcessPoolExecutor(max_workers=max(workers, 1)) as pool:
        for i, ((path, rows, _), (result, error)) in enumerate(
                zip(tasks, pool.map(describe_file, jobs, chunksize=64)), 1):
            if result is None:
                errors.append(f"{path}: {error}")
                continue
            rows_arr = np.asarray(rows)
            for name in dims(patch):
                out[name][rows_arr] = result[name]
            out["ok"][rows_arr] = result["ok"]
            out["minor"][rows_arr] = result["minor"]
            if i % 5000 == 0:
                print(f"  [{label}] {i}/{len(tasks)} tracks, {time.time() - started:.0f}s", flush=True)
    return out, errors


def training_descriptors(rows: list[dict], index: dict[str, Path], seconds: float, patch: int, min_beats: int,
                         workers: int, cache_path: Path) -> dict:
    keys = np.array([r["key"] for r in rows], dtype=str)
    starts = np.array([float(r["start"]) for r in rows])
    if cache_path.exists():
        with np.load(cache_path) as data:
            if len(data["keys"]) == len(keys) and (data["keys"] == keys).all() and \
                    np.allclose(data["starts"], starts):
                print(f"training descriptors: reused {cache_path}")
                return {name: data[name] for name in (*dims(patch), "ok", "minor")}
        print(f"note: {cache_path} does not match this training cache; recomputing")
    by_key: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        by_key.setdefault(key, []).append(i)
    missing = [k for k in by_key if k not in index]
    tasks = [(str(index[k]), rows_k, [float(starts[i]) for i in rows_k])
             for k, rows_k in by_key.items() if k in index]
    print(f"training descriptors: {len(by_key)} tracks, {len(missing)} not in the chroma archive; "
          f"{workers} workers")
    out, errors = describe_rows(tasks, len(keys), seconds, patch, min_beats, workers, "train")
    for line in errors[:5]:
        print(f"  error: {line}")
    if errors:
        print(f"  {len(errors)} chroma files could not be read")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    partial = cache_path.with_name(cache_path.name + ".part")
    with open(partial, "wb") as f:
        np.savez(f, keys=keys, starts=starts, **out)
    os.replace(partial, cache_path)
    return out


# --------------------------------------------------------------------------
# Evaluation chroma (fresh, CPU)
# --------------------------------------------------------------------------


_WORKER_S3 = None


def safe_name(song_id: str) -> str:
    return song_id.replace("/", "__")


def fresh_chroma_worker(song: em.Song, out_path: str, data_root: str, bucket: str | None, region: str | None,
                        human_key: str | None, mom_local: bool, tmp_dir: str | None,
                        max_seconds: float) -> tuple[str, str]:
    """Pool worker: one evaluation song -> its beat-synchronous chroma, as align_covers.py computes it."""
    global _WORKER_S3
    label = f"{song.source}/{song.song_id}"
    try:
        if _WORKER_S3 is None and bucket:
            from src.s3 import S3
            _WORKER_S3 = S3(bucket, region)
        opener = em.AudioOpener(Path(data_root), _WORKER_S3, human_key, mom_local,
                                Path(tmp_dir) if tmp_dir else None)
        with opener.open(song) as path:
            y = ac.load_audio_ffmpeg(path, ac.CHROMA_SR, max_seconds)
        if len(y) < ac.CHROMA_SR * 5:
            return label, "too short"
        chroma, times = ac.beat_sync_chroma(y, ac.CHROMA_SR)
        if chroma.shape[1] < ac.MIN_BEATS:
            return label, "too few beats"
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        partial = out.with_name(out.name + ".part")
        with open(partial, "wb") as f:
            np.savez_compressed(f, chroma=chroma, times=times)
        os.replace(partial, out)
        return label, ""
    except Exception as exc:  # noqa: BLE001 - reported, the song is left out
        return label, f"{type(exc).__name__}: {exc}"


def eval_song_objects(rows: list[dict], meta: dict, data_root: Path, args: argparse.Namespace,
                      need_human: bool) -> dict[tuple[str, str], em.Song]:
    """(source, song id) of every evaluation row -> the em.Song that locates its audio."""
    wanted: dict[str, set[str]] = {}
    for row in rows:
        wanted.setdefault(row["source"], set()).add(row["song"])
    found: dict[tuple[str, str], em.Song] = {}
    if need_human and em.HUMAN in wanted:
        for song in em.human_songs(data_root, meta["human_split"], float(meta["window_seconds"])):
            if song.song_id in wanted[em.HUMAN]:
                found[(em.HUMAN, song.song_id)] = song
    s3 = None
    for source in wanted:
        if source == em.HUMAN:
            continue
        if args.mom_root is not None:
            _, songs = em.mom_songs_local(args.mom_root, source)
        else:
            if s3 is None:
                from src.s3 import S3
                s3 = S3(args.bucket, args.region)
            _, songs = em.mom_songs_s3(s3, args.mom_prefix, source)
        for song in songs:
            if song.song_id in wanted[source]:
                found[(source, song.song_id)] = song
    lost = [f"{s}/{x}" for s, ids in wanted.items() if s != em.HUMAN or need_human
            for x in ids if (s, x) not in found]
    if lost:
        print(f"warning: {len(lost)} evaluation songs could not be located, e.g. {lost[:3]}")
    return found


def compute_fresh_chroma(songs: dict[tuple[str, str], em.Song], chroma_dir: Path, data_root: Path,
                         args: argparse.Namespace) -> None:
    todo = {k: s for k, s in songs.items()
            if not (chroma_dir / k[0] / f"{safe_name(k[1])}.npz").exists()}
    print(f"evaluation chroma: {len(songs)} songs, {len(todo)} to compute in {chroma_dir} "
          f"({args.workers} workers)")
    if not todo:
        return
    mom_local = args.mom_root is not None
    needs_s3 = not mom_local or bool(args.human_audio_key)
    failures: dict[str, int] = {}
    started = time.time()
    with ProcessPoolExecutor(max_workers=max(args.workers, 1), initializer=ac._numba_private_cache) as pool:
        futures = [pool.submit(fresh_chroma_worker, song,
                               str(chroma_dir / key[0] / f"{safe_name(key[1])}.npz"), str(data_root),
                               args.bucket if needs_s3 else None, args.region, args.human_audio_key,
                               mom_local, str(args.tmp_dir) if args.tmp_dir else None, args.max_seconds)
                   for key, song in todo.items()]
        for i, future in enumerate(as_completed(futures), 1):
            label, error = future.result()
            if error:
                source = label.split("/")[0]
                failures[source] = failures.get(source, 0) + 1
                if sum(failures.values()) <= 10:
                    print(f"  {label}: {error}")
            if i % 100 == 0 or i == len(futures):
                print(f"  [chroma] {i}/{len(futures)} songs, {sum(failures.values())} failed, "
                      f"{time.time() - started:.0f}s", flush=True)
    if failures:
        print("  failed per source: " + ", ".join(f"{s} {n}" for s, n in sorted(failures.items())))


def eval_descriptors(rows: list[dict], paths: dict[tuple[str, str], Path], seconds: float, patch: int,
                     min_beats: int) -> tuple[dict, dict[tuple[str, str], bool]]:
    """Descriptors of every evaluation row, and each song's estimated mode."""
    out = empty_descriptors(len(rows), patch)
    by_song: dict[tuple[str, str], list[int]] = {}
    for i, row in enumerate(rows):
        by_song.setdefault((row["source"], row["song"]), []).append(i)
    minor: dict[tuple[str, str], bool] = {}
    for key, idx in by_song.items():
        path = paths.get(key)
        if path is None or not path.exists():
            continue
        result, _ = describe_file((str(path), [float(rows[i]["start"]) for i in idx], seconds, patch, min_beats))
        if result is None:
            continue
        idx_arr = np.asarray(idx)
        for name in dims(patch):
            out[name][idx_arr] = result[name]
        out["ok"][idx_arr] = result["ok"]
        out["minor"][idx_arr] = result["minor"]
        minor[key] = bool(result["minor"])
    return out, minor


def chroma_check(rows: list[dict], fresh: dict[tuple[str, str], Path], archive: dict[str, Path],
                 seconds: float, patch: int, min_beats: int) -> dict:
    """Fresh against archived chroma on the human songs that have both: window 'abs' correlation, key agreement."""
    by_song: dict[str, list[float]] = {}
    for row in rows:
        if row["source"] == em.HUMAN:
            by_song.setdefault(row["song"], []).append(float(row["start"]))
    corrs, same_key, n = [], [], 0
    for song, starts in by_song.items():
        a_path, b_path = fresh.get((em.HUMAN, song)), archive.get(song)
        if a_path is None or b_path is None or not a_path.exists():
            continue
        a, _ = describe_file((str(a_path), starts, seconds, patch, min_beats))
        b, _ = describe_file((str(b_path), starts, seconds, patch, min_beats))
        if a is None or b is None:
            continue
        n += 1
        same_key.append(a["tonic"] == b["tonic"] and a["minor"] == b["minor"])
        for i in np.flatnonzero(a["ok"] & b["ok"]):
            corrs.append(float(np.corrcoef(a["abs"][i], b["abs"][i])[0, 1]))
    return {"songs": n, "median_window_corr": float(np.median(corrs)) if corrs else float("nan"),
            "p05_window_corr": float(np.quantile(corrs, 0.05)) if corrs else float("nan"),
            "same_key": float(np.mean(same_key)) if same_key else float("nan")}


# --------------------------------------------------------------------------
# Gaussians (torch, fp64)
# --------------------------------------------------------------------------


def shrink(cov: torch.Tensor, s: float) -> torch.Tensor:
    """Correlation shrinkage toward I, as MertFlow.set_moments: D ((1 - s) R + s I) D."""
    std = cov.diagonal().clamp_min(1e-12).sqrt()
    corr = cov / std[:, None] / std[None, :]
    eye = torch.eye(len(cov), dtype=cov.dtype, device=cov.device)
    return ((1 - s) * corr + s * eye) * std[:, None] * std[None, :]


@torch.no_grad()
def joint_moments(parts: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and covariance (fp64, n - 1) of the column-concatenation of `parts`, without building it."""
    n = len(parts[0])
    width = sum(p.shape[1] for p in parts)
    total = torch.zeros(width, dtype=torch.float64, device=parts[0].device)
    for s in range(0, n, CHUNK):
        total += torch.cat([p[s:s + CHUNK].float() for p in parts], 1).sum(0, dtype=torch.float64)
    mean = total / n
    centre = mean.float()
    acc = torch.zeros(width, width, dtype=torch.float64, device=parts[0].device)
    for s in range(0, n, CHUNK):
        x = torch.cat([p[s:s + CHUNK].float() for p in parts], 1) - centre
        acc += (x.T @ x).double()
    return mean, acc / max(n - 1, 1)


def log_norm(chol: torch.Tensor) -> float:
    """log of the Gaussian's normalising constant: -(d/2) log 2 pi - log det L."""
    return float(-0.5 * chol.shape[0] * math.log(2 * math.pi) - chol.diagonal().log().sum())


@torch.no_grad()
def mahalanobis(x: torch.Tensor, mean: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
    """||L^-1 (x - mean)||^2 per row, fp64."""
    out = []
    for s in range(0, len(x), CHUNK):
        y = torch.linalg.solve_triangular(chol, (x[s:s + CHUNK].double() - mean).T, upper=False)
        out.append((y ** 2).sum(0))
    return torch.cat(out)


@dataclass
class JointFit:
    mu_z: torch.Tensor
    mu_c: torch.Tensor
    chol_c: torch.Tensor  # marginal of c
    chol_cond: torch.Tensor  # z given c
    coef: torch.Tensor  # (F, d): E[z | c] = mu_z + coef (c - mu_c)
    explained: float  # share of z's whitened variance explained by c


@torch.no_grad()
def fit_joint(z: torch.Tensor, c: torch.Tensor, shrinkage: float) -> JointFit:
    mean, cov = joint_moments([z, c])
    cov = shrink(cov, shrinkage)
    f = z.shape[1]
    a, cz, d = cov[:f, :f], cov[:f, f:], cov[f:, f:]
    chol_a = torch.linalg.cholesky(a)
    chol_d = torch.linalg.cholesky(d)
    w = torch.linalg.solve_triangular(chol_d, cz.T, upper=False)  # (d, F)
    cond = a - w.T @ w
    chol_cond = torch.linalg.cholesky(0.5 * (cond + cond.T))
    coef = torch.cholesky_solve(cz.T, chol_d).T
    v = torch.linalg.solve_triangular(chol_a, w.T, upper=False)
    return JointFit(mean[:f], mean[f:], chol_d, chol_cond, coef, float((v ** 2).sum() / f))


@torch.no_grad()
def residual_q(z: torch.Tensor, c: torch.Tensor, fit: JointFit) -> torch.Tensor:
    """Mahalanobis distance of z from its content-predicted mean, under the residual covariance."""
    out = []
    for s in range(0, len(z), CHUNK):
        r = z[s:s + CHUNK].double() - fit.mu_z - (c[s:s + CHUNK].double() - fit.mu_c) @ fit.coef.T
        y = torch.linalg.solve_triangular(fit.chol_cond, r.T, upper=False)
        out.append((y ** 2).sum(0))
    return torch.cat(out)


@torch.no_grad()
def fit_scale(c: torch.Tensor, target: np.ndarray, ridge: float) -> tuple[torch.Tensor, float]:
    """Ridge regression target ~ a + c b on standardized c: (b, a)."""
    t = torch.from_numpy(target).to(c.device)
    mean_c = c.double().mean(0)
    gram = torch.zeros(c.shape[1], c.shape[1], dtype=torch.float64, device=c.device)
    cross = torch.zeros(c.shape[1], dtype=torch.float64, device=c.device)
    for s in range(0, len(c), CHUNK):
        x = c[s:s + CHUNK].double() - mean_c
        gram += x.T @ x
        cross += x.T @ (t[s:s + CHUNK] - t.mean())
    n = len(c)
    eye = torch.eye(c.shape[1], dtype=torch.float64, device=c.device)
    b = torch.linalg.solve(gram / n + ridge * eye, cross / n)
    return b, float(t.mean() - mean_c @ b)


@torch.no_grad()
def predict(c: torch.Tensor, b: torch.Tensor, a: float) -> np.ndarray:
    return (c.double() @ b + a).cpu().numpy()


def scaled(logp: np.ndarray, q: np.ndarray, log_hat: np.ndarray, n_feat: int) -> np.ndarray:
    """log N(x; mu, tau^2 Sigma) from log N(x; mu, Sigma) and q, with log tau^2 = log_hat (section 21)."""
    return logp - 0.5 * q * (np.exp(-log_hat) - 1.0) - 0.5 * n_feat * log_hat


@torch.no_grad()
def between_share(x: torch.Tensor, work: torch.Tensor, n_works: int, chol: torch.Tensor | None = None,
                  total: float | None = None) -> float:
    """Share of the variance of x that lies between works; whitened by `chol` if given.

    Whitened, the total is the mean squared Mahalanobis distance of the rows
    (`total`), which shrinkage keeps slightly under the dimension.
    """
    means = cs.group_means(x, work, n_works)  # fp64
    counts = torch.bincount(work, minlength=n_works).double()
    grand = (means * counts[:, None]).sum(0) / counts.sum()
    centred = means - grand
    if chol is not None:
        centred = torch.linalg.solve_triangular(chol, centred.T, upper=False).T
        total = float(total if total is not None else x.shape[1])
    else:
        total = float(x.double().var(0).sum())
    return float((centred ** 2 * counts[:, None]).sum() / counts.sum() / total)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    started = time.time()

    # ---------------- paths and rows, without touching the GPU ----------------
    # The chroma stages fork worker processes; they run before CUDA is initialised.
    checkpoint_cpu = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint_cpu.get("kind") != "mert_flow":
        raise SystemExit(f"{args.checkpoint} is not a train_mert_flow.py checkpoint")
    trained = parse_chain(checkpoint_cpu.get("codec"))
    shrinkage = float(checkpoint_cpu.get("args", {}).get("gaussian_shrinkage", 0.01))
    train_root = args.train_cache or Path(checkpoint_cpu["train_cache"])
    del checkpoint_cpu
    train = PooledCache(train_root)
    train_rows = train.rows()
    eval_root = args.eval_cache or train_root.parent / (
        f"eval-{args.human_split}" + (f"-{chain_slug(trained)}" if trained else ""))
    if not (eval_root / "pooled.npy").exists():
        raise SystemExit(f"no evaluation cache at {eval_root}; pass --eval-cache (experiments.md section 20 "
                         "says where to find or how to rebuild it)")
    meta = json.loads((eval_root / "meta.json").read_text(encoding="utf-8"))
    with open(eval_root / "windows.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    split = meta["human_split"]
    seconds = float(meta["window_seconds"])
    if float(train.meta["window_seconds"]) != seconds:
        raise SystemExit(f"{train_root} holds {train.meta['window_seconds']} s windows, {eval_root} {seconds} s")
    data_root = args.data_root or Path(meta.get("data_root", "."))
    chroma_root = args.chroma_root or data_root / "chroma"
    work_dir = args.work_dir or args.checkpoint.parent / "chroma_condition"
    if not chroma_root.is_dir():
        raise SystemExit(f"no chroma archive at {chroma_root}; extract $RUN_S3/preprocessing/chroma.tgz there "
                         "(experiments.md section 23) or pass --chroma-root")
    index = archive_index(chroma_root)
    print(f"chroma archive {chroma_root}: {len(index)} tracks "
          + ", ".join(f"{d.name} {sum(1 for _ in d.glob('*.npz'))}" for d in sorted(chroma_root.iterdir())
                      if d.is_dir()))

    # ---------------- chroma stages (CPU) ----------------
    patch, min_beats = args.patch_beats, args.min_beats
    train_desc = training_descriptors(train_rows, index, seconds, patch, min_beats, args.workers,
                                      work_dir / f"train_descriptors_p{patch}.npz")
    eval_chroma_dir = work_dir / "eval_chroma"
    need_human = args.human_chroma == "fresh"
    songs = eval_song_objects(rows, meta, data_root, args, need_human)
    compute_fresh_chroma(songs, eval_chroma_dir, data_root, args)
    paths: dict[tuple[str, str], Path] = {}
    for row in rows:
        key = (row["source"], row["song"])
        if key[0] == em.HUMAN and not need_human:
            if key[1] in index:
                paths[key] = index[key[1]]
        else:
            paths[key] = eval_chroma_dir / key[0] / f"{safe_name(key[1])}.npz"
    eval_desc, eval_minor = eval_descriptors(rows, paths, seconds, patch, min_beats)
    check = (chroma_check(rows, paths, index, seconds, patch, min_beats) if need_human
             else {"songs": 0, "median_window_corr": float("nan"), "p05_window_corr": float("nan"),
                   "same_key": float("nan")})
    print(f"chroma stages done in {time.time() - started:.0f}s")

    # ---------------- model and features (GPU) ----------------
    from src.training import pick_device
    from train_mert_flow import load_mert_flow

    device = pick_device(args.device)
    model, config, checkpoint = load_mert_flow(args.checkpoint, device)
    if train.meta["model_name"] != config.mert.model_name or \
            float(train.meta["window_seconds"]) != float(config.data.window_seconds) or \
            parse_chain(train.meta.get("codec")) != trained:
        raise SystemExit(f"{train_root} is not this flow's training cache (model, window length or codec differ)")
    if meta["model_name"] != config.mert.model_name or seconds != float(config.data.window_seconds):
        raise SystemExit(f"{eval_root} was cached with another model or window length than the flow")
    if parse_chain(meta.get("codec")) != trained:
        print("note: the evaluation cache's codec is not the flow's training chain")
    cache = tm.EvalCache(eval_root, device)
    if len(cache.rows) != len(rows):
        raise SystemExit(f"{eval_root}: windows.jsonl changed while running")
    sources = np.array([row["source"] for row in rows])
    generators = [g for g in meta["generators"] if (sources == g).any()]
    pools = {"all": generators,
             "dev": [g for g in generators if g in args.dev],
             "others": [g for g in generators if g not in args.dev]}
    pools = {name: members for name, members in pools.items() if members}
    print(f"flow {args.checkpoint} (arm {checkpoint.get('arm')}, epoch {checkpoint.get('epoch')}); "
          f"shrinkage s = {shrinkage:g}")
    print(f"eval cache {eval_root}: {len(rows)} windows, {split} vs. {', '.join(generators)}; "
          f"dev = {', '.join(pools.get('dev', [])) or 'none'}")
    if split == "test50":
        print("note: human = test50. Choosing a descriptor or a score from this run would be selection on test50")

    # Training windows with chroma.
    z_all = tt.training_features(train, model, device, args.batch_size)
    if len(z_all) != len(train_rows):
        raise SystemExit(f"{len(z_all)} training features against {len(train_rows)} index rows")
    keep_train = np.flatnonzero(train_desc["ok"])
    z = z_all[torch.from_numpy(keep_train).to(device)]
    del z_all
    if device.type == "cuda":
        torch.cuda.empty_cache()
    train_keys = np.array([train_rows[i]["key"] for i in keep_train], dtype=str)
    work_names, work_idx = np.unique(np.array([k.split("_")[0] for k in train_keys]), return_inverse=True)
    work_t = torch.from_numpy(work_idx).to(device)
    n_feat = z.shape[1]
    track_names, track_first = np.unique(train_keys, return_index=True)
    print(f"\ntraining: {len(keep_train)} of {len(train_rows)} windows have chroma "
          f"({len(keep_train) / len(train_rows):.1%}); {len(track_names)} recordings, {len(work_names)} works; "
          f"minor-key share of recordings {train_desc['minor'][keep_train][track_first].mean():.2f}")
    if len(keep_train) < 0.9 * len(train_rows):
        print("warning: more than 10 % of the training windows have no chroma; is --chroma-root the full archive?")

    # Evaluation songs whose every window has chroma.
    song_keys = np.array([f"{row['source']}\t{row['song']}" for row in rows])
    bad_songs = set(song_keys[~eval_desc["ok"]])
    keep_eval = np.flatnonzero(~np.isin(song_keys, list(bad_songs)))
    for source in [em.HUMAN, *generators]:
        total = len(set(song_keys[sources == source]))
        lost = len({s for s in bad_songs if s.split("\t")[0] == source})
        minor_share = np.mean([m for (src, _), m in eval_minor.items() if src == source] or [float("nan")])
        print(f"  {source:<12} {total - lost}/{total} songs kept; minor-key share {minor_share:.2f}")
    if need_human:
        print(f"chroma check, fresh against the archive on {check['songs']} human songs: window 'abs' "
              f"correlation median {check['median_window_corr']:.4f} (5 % quantile {check['p05_window_corr']:.4f}); "
              f"same key {check['same_key']:.2f}")
        if check["songs"] and check["median_window_corr"] < 0.98:
            print("warning: fresh and archived chroma differ; training and evaluation descriptors are not alike")
    rows_k = [rows[i] for i in keep_eval]
    sources_k = sources[keep_eval]
    human_k = sources_k == em.HUMAN
    if not human_k.any() or not np.isin(sources_k, generators).any():
        raise SystemExit("no human or no synthetic song has chroma for every window; see the errors above")

    with torch.no_grad():
        z_eval_all = torch.cat([model.features(c.float()) for c in cache.pooled.split(args.batch_size)])
        ckpt_logp = model.gaussian_log_prob(z_eval_all).double().cpu().numpy()[keep_eval]
    z_eval = z_eval_all[torch.from_numpy(keep_eval).to(device)]
    del z_eval_all

    # ---------------- unconditional baseline on the same windows ----------------
    everything = torch.ones(len(z), dtype=torch.bool, device=device)
    mean_z, cov_z = tt.moments(z, everything)
    chol_none = torch.linalg.cholesky(shrink(cov_z, shrinkage))
    const_none = log_norm(chol_none)
    q_train = mahalanobis(z, mean_z, chol_none).cpu().numpy()
    q_eval = mahalanobis(z_eval, mean_z, chol_none).cpu().numpy()
    logp_none = const_none - 0.5 * q_eval
    q_mean = float(q_train.mean())
    mert_between = between_share(z, work_t, len(work_names), chol_none, q_mean)
    del cov_z

    base, base_table = tt.detection(rows_k, logp_none, generators, pools, args.fpr)
    ckpt, _ = tt.detection(rows_k, ckpt_logp, generators, pools, args.fpr)
    print(f"check: 'none' refitted on these windows, AUROC all {base['all']:.4f}; the checkpoint's Gaussian on the "
          f"same songs {ckpt['all']:.4f}")
    print(f"MERT's between-work share (whitened z): {mert_between:.3f}")
    work_of = emp.work_of(data_root, split)
    members = tt.unit_members([str((work_of or {}).get(song, song.split("_")[0]))
                               for song in base_table[em.HUMAN]["songs"]])

    song_names, song_index = np.unique(np.array([f"{r['source']}\t{r['song']}" for r in rows_k]),
                                       return_inverse=True)
    song_source = np.array([name.split("\t")[0] for name in song_names])
    song_id = np.array([name.split("\t")[1] for name in song_names])
    human_song = song_source == em.HUMAN
    log_tau2_eval = np.log(q_eval / q_mean)
    song_log = cs.numpy_group_means(log_tau2_eval, song_index)
    oracle = cs.leave_one_out_r2(song_log[human_song], np.array([s.split("_")[0] for s in song_id[human_song]]))
    actual = {g: float(song_log[song_source == g].mean() - song_log[human_song].mean()) for g in generators}
    print(f"held-out work oracle r² for log τ² ({int(human_song.sum())} human songs): {oracle:.3f}")

    # ---------------- each descriptor ----------------
    configs: list[dict] = []
    descriptor_rows: list[dict] = []
    for name in args.descriptors:
        parts = PARTS[name]
        c_train_np = np.concatenate([train_desc[p][keep_train] for p in parts], 1).astype(np.float64)
        c_eval_np = np.concatenate([eval_desc[p][keep_eval] for p in parts], 1).astype(np.float64)
        mu, sd = c_train_np.mean(0), c_train_np.std(0)
        live = sd > 1e-8
        c_train = torch.from_numpy(((c_train_np[:, live] - mu[live]) / sd[live]).astype(np.float32)).to(device)
        c_eval = torch.from_numpy(((c_eval_np[:, live] - mu[live]) / sd[live]).astype(np.float32)).to(device)
        del c_train_np, c_eval_np

        fit = fit_joint(z, c_train, shrinkage)
        logp_c = log_norm(fit.chol_c) - 0.5 * mahalanobis(c_eval, fit.mu_c, fit.chol_c).cpu().numpy()
        const_cond = log_norm(fit.chol_cond)
        qr_train = residual_q(z, c_train, fit).cpu().numpy()
        qr_eval = residual_q(z_eval, c_eval, fit).cpu().numpy()
        logp_cond = const_cond - 0.5 * qr_eval
        pmi = logp_cond - logp_none
        pmi_by = {s: float(pmi[sources_k == s].mean()) for s in [em.HUMAN, *generators]}

        b, a = fit_scale(c_train, np.log(q_train / q_mean), shrinkage)
        hat_train, hat_eval = predict(c_train, b, a), predict(c_eval, b, a)
        br, ar = fit_scale(c_train, np.log(qr_train / qr_train.mean()), shrinkage)
        hat_r_eval = predict(c_eval, br, ar)
        song_hat = cs.numpy_group_means(hat_eval, song_index)
        song_hat_r = cs.numpy_group_means(hat_r_eval, song_index)
        song_log_r = cs.numpy_group_means(np.log(qr_eval / qr_train.mean()), song_index)
        excuse = {g: float(song_hat[song_source == g].mean() - song_hat[human_song].mean()) for g in generators}
        info = {
            "descriptor": name, "dims": int(live.sum()),
            "between_work": between_share(c_train, work_t, len(work_names)),
            "explained": fit.explained, "pmi": pmi_by,
            "r2_scale_train": cs.r2(hat_train, np.log(q_train / q_mean)),
            "r2_scale_held_window": cs.r2(hat_eval[human_k], log_tau2_eval[human_k]),
            "r2_scale_held_song": cs.r2(song_hat[human_song], song_log[human_song]),
            "r2_resid_scale_held_song": cs.r2(song_hat_r[human_song], song_log_r[human_song]),
            "excuse": excuse,
        }
        descriptor_rows.append(info)

        scores = {
            "content": logp_c,
            "mean": logp_cond,
            "joint": logp_cond + logp_c,
            "scale": scaled(logp_none, q_eval, hat_eval, n_feat),
            "mean+scale": scaled(logp_cond, qr_eval, hat_r_eval, n_feat),
        }
        for score_name, values in scores.items():
            metrics, table = tt.detection(rows_k, values, generators, pools, args.fpr)
            configs.append({
                "descriptor": name, "score": score_name, "auroc": metrics,
                "delta_ci95": ({} if score_name == "content" else
                               tt.paired_delta(base_table, table, members, pools, args.bootstrap, args.seed)),
            })
        print(f"  {name}: fitted ({info['dims']} dims), {time.time() - started:.0f}s", flush=True)
        del c_train, c_eval, fit

    # ---------------- report ----------------
    print("\n=== The descriptors ===")
    print("between-work = share of c's variance between training works (MERT's, whitened: "
          f"{mert_between:.3f}); explained = share of z's whitened variance c explains (in-sample); "
          "r² of log τ² from c: training windows / held-out human windows / held-out human songs "
          f"(work oracle {oracle:.3f}) / the residual's, held-out songs")
    print(f"{'descriptor':<12}{'dims':>6}{'between':>9}{'explained':>11}{'r² train':>10}{'r² win':>9}"
          f"{'r² song':>9}{'r² resid':>10}")
    for d in descriptor_rows:
        print(f"{d['descriptor']:<12}{d['dims']:>6}{d['between_work']:>9.3f}{d['explained']:>11.4f}"
              f"{d['r2_scale_train']:>10.3f}{d['r2_scale_held_window']:>9.3f}{d['r2_scale_held_song']:>9.3f}"
              f"{d['r2_resid_scale_held_song']:>10.3f}")
    print("\nPMI = mean log p(z | c) - log p(z) per window, nats; generators as the difference from the human "
          "value (below 0 = content explains synthetic windows less: what the hypothesis predicts)")
    print(f"{'descriptor':<12}{'human':>10}" + "".join(f"{g:>13}" for g in generators))
    for d in descriptor_rows:
        h = d["pmi"][em.HUMAN]
        print(f"{d['descriptor']:<12}{h:>10.2f}" + "".join(f"{d['pmi'][g] - h:>+13.2f}" for g in generators))
    print("\nexcuse = mean predicted log τ² of a generator's songs minus the humans' [actual difference]")
    print(f"{'descriptor':<12}" + "".join(f"{g:>20}" for g in generators))
    for d in descriptor_rows:
        print(f"{d['descriptor']:<12}"
              + "".join(f"{d['excuse'][g]:+8.3f} [{actual[g]:+.3f}]".rjust(20) for g in generators))

    print(f"\n=== Detection (song = min over windows; {split} vs. MoM) ===")
    cols = generators + list(pools)
    header = [f"{g}*" if g in pools.get("dev", []) else g for g in generators] + list(pools)
    print("* = dev generator. window = pooled window-level AUROC; flagged = synthetic songs below the "
          f"{args.fpr:.0%} human quantile. 'content' = log p(c) alone, a diagnostic")

    def label(c: dict) -> str:
        return f"{c['descriptor']} {c['score']}"

    print(f"{'setting':<20}" + "".join(f"{h:>11}" for h in header) + f"{'window':>9}{'flagged':>9}")
    print(f"{'checkpoint':<20}" + "".join(f"{ckpt[col]:>11.3f}" for col in cols)
          + f"{ckpt['window']:>9.3f}{ckpt['flagged']:>9.3f}")
    print(f"{'none':<20}" + "".join(f"{base[col]:>11.3f}" for col in cols)
          + f"{base['window']:>9.3f}{base['flagged']:>9.3f}")
    for c in configs:
        m = c["auroc"]
        print(f"{label(c):<20}" + "".join(f"{m[col]:>11.3f}" for col in cols)
              + f"{m['window']:>9.3f}{m['flagged']:>9.3f}")
    print(f"\nΔ AUROC against 'none', paired bootstrap 95 % ({args.bootstrap} resamples; human by work, "
          "generators by song)")
    print(f"{'setting':<20}" + "".join(f"{name:>26}" for name in pools))
    for c in configs:
        if c["score"] == "content":
            continue
        cells = []
        for name in pools:
            lo, hi = c["delta_ci95"][name]
            cells.append(f"{c['auroc'][name] - base[name]:+.3f} [{lo:+.3f},{hi:+.3f}]".rjust(26))
        print(f"{label(c):<20}" + "".join(cells))

    report = {
        "checkpoint": str(args.checkpoint), "arm": checkpoint.get("arm"), "epoch": checkpoint.get("epoch"),
        "train_cache": str(train_root), "eval_cache": str(eval_root), "human_split": split,
        "trained_codec": chain_name(trained), "shrinkage": shrinkage, "generators": generators, "pools": pools,
        "chroma_root": str(chroma_root), "human_chroma": args.human_chroma, "patch_beats": patch,
        "min_beats": min_beats, "max_seconds": args.max_seconds, "chroma_check": check,
        "train": {"windows": len(train_rows), "windows_with_chroma": int(len(keep_train)),
                  "recordings": int(len(track_names)), "works": int(len(work_names)), "mean_q": q_mean,
                  "mert_between_work": mert_between},
        "eval": {"windows": len(rows), "windows_kept": int(len(keep_eval)),
                 "songs_dropped": sorted(bad_songs), "oracle_r2": oracle, "actual_log_tau2_vs_human": actual},
        "none": base, "checkpoint_gaussian": ckpt, "fpr": args.fpr, "bootstrap": args.bootstrap,
        "seed": args.seed, "descriptors": descriptor_rows, "configs": configs,
    }
    out = args.out or args.checkpoint.parent / f"chroma_condition_{split}.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\n-> {out}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
