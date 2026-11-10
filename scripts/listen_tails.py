#!/usr/bin/env python3
"""List the human recordings at both ends of the common factor, for listening.

Why (experiments.md section 18, "Next", item 1). The human noise of the density
score is one common factor τ: some windows are atypical in every direction at
once (sections 16 and 18). 40 % of its variance lies between recordings of one
work (section 17), and its heavy upper tail sets the 5 % FPR threshold. What
those recordings are has never been looked at. If the upper tail is live,
amateur, old, or not music at all, the human spread of τ is a property of
SHS100K rather than of human music, and it lies on the axis of the dataset
confound (section 13).

What. Every window of a human cache (default: the checkpoint's --val-cache, i.e.
val50) is put in the Gaussian baseline's whitened coordinates w, as
dilution_diagnostic.py does. Per window, τ² = ‖w‖² / (mean ‖w‖² over the
cache), so 1 is an average window (section 18's normalisation). A recording's
τ² is the geometric mean over its windows, i.e. its mean log ‖w‖², the
statistic of variance_levels.py. Three groups are listed:
  high     the --n recordings with the largest τ²
  middle   the --n-middle recordings closest to the median: a reference for the ear
  low      the --n recordings with the smallest τ²
With --rank within-work, recordings are ranked by their log τ² minus their
work's mean (works with ≥ 2 recordings only). That isolates the recording level
(40 % of τ) from the works level (36 %), which content explains.

Output in --out:
  index.html       players and metadata, one section per group (open in a browser)
  tails.csv        the listed recordings, with empty `kind` and `notes` columns
                   to fill in while listening
  recordings.csv   every recording of the cache, sorted by τ², for a further look
  clips/*.wav      every window of each listed recording as MERT heard it: 24 kHz
                   mono, through the cache's codec chain if it has one.
                   --no-audio skips them; the YouTube links still point at each
                   window's start.

Audio paths come from the split's manifest under --data-root ($DATA on the
boxes). Tracks missing locally are fetched from --audio-s3, as listen_pairs.py
does. Nothing here runs MERT.

What to listen for, per recording: live (crowd, room sound), amateur (phone
microphone, noise floor), old (narrow bandwidth, hiss, mono), not music
(speech, silence, an interview), and whether the low group is polished studio
audio. Windows marked ▲ are above the human 95 % window quantile of τ², so the
Gaussian flags them at 5 % FPR at window level.

Examples:
    python scripts/listen_tails.py \\
        --checkpoint checkpoints/mert-flow-unfrozen/mert_flow_last.pt \\
        --data-root $DATA --audio-s3 $AUDIO_S3
    python scripts/listen_tails.py \\
        --checkpoint checkpoints/mert-flow-unfrozen/mert_flow_last.pt \\
        --data-root $DATA --audio-s3 $AUDIO_S3 --rank within-work --out listen_tails_within
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

import dilution_diagnostic as dd  # noqa: E402
import variance_levels as vl  # noqa: E402
from listen_pairs import AudioSource, _clock, write_wav  # noqa: E402
from shs100k_meta import DEFAULT_DATA_ROOT, SPLITS, Track, _parse_split, designated_splits  # noqa: E402
from src.data.codec import chain_name, decode_window_through, parse_chain  # noqa: E402
from src.data.manifest import manifest_dir, read_tracks  # noqa: E402
from src.data.pooled_cache import PooledCache  # noqa: E402
from src.data.windows import WindowConfig  # noqa: E402
from src.training import pick_device  # noqa: E402

GROUPS = ("high", "middle", "low")
QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="train_mert_flow.py checkpoint (mert_flow_last.pt)")
    parser.add_argument("--cache", type=Path, default=None,
                        help="pooled-MERT cache of one human split (default: the checkpoint's --val-cache)")
    parser.add_argument("--n", type=int, default=20, help="recordings in the high and in the low group")
    parser.add_argument("--n-middle", type=int, default=5, help="recordings around the median")
    parser.add_argument("--rank", choices=("absolute", "within-work"), default="absolute",
                        help="rank by τ², or by τ² relative to the recording's work")
    parser.add_argument("--min-windows", type=int, default=2,
                        help="leave out recordings with fewer cached windows")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("DATA", DEFAULT_DATA_ROOT)),
                        help="default: $DATA if set, else the repo's data/")
    parser.add_argument("--audio-s3", default=None,
                        help="s3://BUCKET/PREFIX holding {split}/ audio, for tracks missing locally")
    parser.add_argument("--no-audio", action="store_true", help="write no clips, only the lists")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("listen_tails"))
    return parser.parse_args()


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


def source_splits(split: str) -> dict[int, str]:
    """Work id -> the official split its audio and metadata live under."""
    return designated_splits().get(split, {})


def track_meta(split: str, sources: dict[int, str]) -> dict[str, Track]:
    """Title, artist and YouTube id per track key, from the official split CSVs."""
    names = set(sources.values()) or ({split} if split in SPLITS else set())
    meta: dict[str, Track] = {}
    for name in sorted(names):
        try:
            meta |= {t.key: t for t in _parse_split(name)}
        except Exception as exc:  # titles are a convenience, never fatal
            print(f"note: no SHS metadata for {name} ({exc})")
    return meta


def audio_paths(data_root: Path, split: str, keys: list[str], sources: dict[int, str]) -> dict[str, str]:
    """Track key -> audio path relative to the data root, from the manifest when it exists."""
    paths: dict[str, str] = {}
    if (manifest_dir(data_root, split) / "tracks.jsonl").exists():
        paths = {t.key: t.audio for t in read_tracks(data_root, split)}
    else:
        print(f"note: no manifest for {split} under {data_root}; guessing audio/<split>/<key>.webm")
    for key in keys:
        # not in the manifest: assume the usual layout and let the fetch say
        paths.setdefault(key, f"audio/{sources.get(int(key.split('_')[0]), split)}/{key}.webm")
    return paths


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def per_recording(keys: np.ndarray, log_tau2: np.ndarray, flow_nll: np.ndarray, min_windows: int) -> dict:
    """Window statistics -> one entry per recording, plus each window's recording index."""
    rec_keys, rec = np.unique(keys, return_inverse=True)
    n_win = np.bincount(rec)
    mean_log = np.bincount(rec, weights=log_tau2) / n_win
    mean_flow = np.bincount(rec, weights=flow_nll) / n_win
    eligible = n_win >= min_windows

    # The recording relative to its work: every eligible recording weighs equally.
    _, work = np.unique(np.array([k.split("_")[0] for k in rec_keys]), return_inverse=True)
    w_count = np.bincount(work, weights=eligible.astype(np.float64))
    w_sum = np.bincount(work, weights=np.where(eligible, mean_log, 0.0))
    work_mean = np.divide(w_sum, w_count, out=np.full_like(w_sum, np.nan), where=w_count > 0)
    vs_work = mean_log - work_mean[work]
    has_peers = eligible & (w_count[work] >= 2)
    return {"keys": rec_keys, "index": rec, "n_windows": n_win, "log_tau2": mean_log,
            "vs_work": np.where(has_peers, vs_work, np.nan), "flow_nll": mean_flow,
            "eligible": eligible, "rankable_within": has_peers}


def choose_groups(score: np.ndarray, rankable: np.ndarray, n: int, n_middle: int) -> dict[str, list[int]]:
    """Recording indices per group: high (largest first), middle, low (smallest first)."""
    idx = np.flatnonzero(rankable)
    order = idx[np.argsort(-score[idx], kind="stable")]
    n = min(n, len(order) // 2)
    median = np.median(score[idx])
    middle = idx[np.argsort(np.abs(score[idx] - median), kind="stable")[:n_middle]]
    middle = middle[np.argsort(-score[middle], kind="stable")]
    return {"high": order[:n].tolist(), "middle": middle.tolist(), "low": order[::-1][:n].tolist()}


def percentiles(score: np.ndarray, rankable: np.ndarray) -> np.ndarray:
    """Share of rankable recordings scoring below each one, in %."""
    out = np.full(len(score), np.nan)
    idx = np.flatnonzero(rankable)
    ranks = np.argsort(np.argsort(score[idx], kind="stable"), kind="stable")
    out[idx] = 100.0 * ranks / max(len(idx) - 1, 1)
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def youtube(track: Track | None, start: float) -> str:
    return f"{track.url}&t={int(start)}" if track else ""


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_index(out: Path, title: str, intro: str, groups: dict[str, list[dict]]) -> None:
    captions = {
        "high": "The most atypical recordings under this ranking.",
        "middle": "Around the median: a reference for what an ordinary recording sounds like.",
        "low": "The least atypical recordings: the dense core of the human distribution.",
    }
    sections = []
    for group in GROUPS:
        items = []
        for r in groups[group]:
            windows = []
            for w in r["windows"]:
                link = f" &middot; <a href='{html.escape(w['url'])}'>YouTube</a>" if w["url"] else ""
                player = (f"<br><audio controls preload='none' src='{html.escape(w['clip'])}'></audio>"
                          if w["clip"] else "")
                mark = " &#9650;" if w["above_q95"] else ""
                windows.append(f"<div class='win'>w{w['slot']} @ {_clock(w['start'])} &middot; "
                               f"τ² {w['tau2']:.2f}{mark}{link}{player}</div>")
            relative = (f" &middot; vs. work {r['tau2_vs_work']:.2f}"
                        if r["tau2_vs_work"] != "" else "")
            items.append(
                f"<div class='rec'><b>{r['rank']}. {html.escape(r['artist'] or '?')}</b> &mdash; "
                f"<i>{html.escape(r['title'] or '?')}</i><br><small>work {r['work']} &middot; "
                f"{r['key']} &middot; τ² {r['tau2']:.2f}{relative} &middot; "
                f"{r['percentile']:.1f}th percentile &middot; {r['n_windows']} windows</small>"
                f"{''.join(windows)}</div>"
            )
        sections.append(f"<section><h2>{group}</h2><p>{captions[group]}</p>{''.join(items)}</section>")
    page = (
        f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title>"
        "<style>body{font:15px system-ui;max-width:900px;margin:2em auto;padding:0 16px}"
        "section{border-top:1px solid #ccc;padding:.5em 0}.rec{margin:1.2em 0}"
        ".win{margin:.4em 0 .4em 1em}audio{width:100%;max-width:560px}</style>"
        f"<h1>{html.escape(title)}</h1><p>{intro}</p>{''.join(sections)}"
    )
    (out / "index.html").write_text(page, encoding="utf-8")


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
    chain = parse_chain(meta.get("codec"))
    split = meta["split"]
    print(f"flow {args.checkpoint} (arm {checkpoint.get('arm')}, epoch {checkpoint.get('epoch')})")
    print(f"cache {root}: {len(cache)} windows of {split}, codec {chain_name(chain) or 'none (original audio)'}")
    if root.resolve() == Path(checkpoint["train_cache"]).resolve():
        print("note: this is the flow's training cache, so τ² is in-sample (its tails are lighter)")

    rows = cache.rows()
    keys = np.array([row["key"] for row in rows], dtype=str)
    starts = np.array([float(row["start"]) for row in rows])
    axes, lam = dd.principal_axes(model)
    stats = vl.window_statistics(cache, model, axes / np.sqrt(lam), args.batch_size, device)
    q = stats["‖w‖²"]
    if len(q) != len(rows):
        raise SystemExit(f"{len(q)} scored windows against {len(rows)} index rows")
    tau2 = q / q.mean()
    q95 = float(np.quantile(tau2, 0.95))

    recs = per_recording(keys, np.log(tau2), stats["flow -log p"], args.min_windows)
    within = args.rank == "within-work"
    score = recs["vs_work"] if within else recs["log_tau2"]
    rankable = recs["rankable_within"] if within else recs["eligible"]
    if not rankable.any():
        raise SystemExit("no recording can be ranked; lower --min-windows")
    pct = percentiles(score, rankable)
    groups = choose_groups(score, rankable, args.n, args.n_middle)

    sources = source_splits(split)
    tracks = track_meta(split, sources)
    print(f"\n{len(rows)} windows, {len(recs['keys'])} recordings "
          f"({int(rankable.sum())} ranked, rank = {args.rank}); mean ‖w‖² = {q.mean():.1f}")
    print("window τ² quantiles: " + ", ".join(f"{p:.0%} {np.quantile(tau2, p):.2f}" for p in QUANTILES))

    def describe(r: int) -> dict:
        key = str(recs["keys"][r])
        track = tracks.get(key)
        vs = recs["vs_work"][r]
        return {"key": key, "work": key.split("_")[0], "title": track.title if track else "",
                "artist": track.artist if track else "", "url": track.url if track else "",
                "n_windows": int(recs["n_windows"][r]), "tau2": float(np.exp(recs["log_tau2"][r])),
                "tau2_vs_work": "" if np.isnan(vs) else float(np.exp(vs)),
                "percentile": float(pct[r]), "flow_nll": float(recs["flow_nll"][r])}

    # Every recording, for a further look.
    args.out.mkdir(parents=True, exist_ok=True)
    everything = [describe(r) for r in np.argsort(-recs["log_tau2"], kind="stable")]
    write_csv(args.out / "recordings.csv", everything,
              ["key", "work", "title", "artist", "n_windows", "tau2", "tau2_vs_work",
               "percentile", "flow_nll", "url"])

    listed = [r for g in GROUPS for r in groups[g]]
    audio = None
    if not args.no_audio:
        paths = audio_paths(args.data_root, split, [str(recs["keys"][r]) for r in listed], sources)
        audio = AudioSource(args.data_root, paths, args.audio_s3, args.out / "audio")
        (args.out / "clips").mkdir(exist_ok=True)
    window_config = WindowConfig(window_seconds=float(meta["window_seconds"]))

    report: dict[str, list[dict]] = {g: [] for g in GROUPS}
    missing = 0
    for group in GROUPS:
        print(f"\n=== {group} ===")
        print(f"{'#':>3} {'τ²':>6} {'vs work':>8} {'pct':>6}  {'work':>7}  {'key':<16} artist - title")
        for rank, r in enumerate(groups[group], start=1):
            entry = describe(r) | {"group": group, "rank": rank, "kind": "", "notes": ""}
            track = tracks.get(entry["key"])
            members = np.flatnonzero(recs["index"] == r)
            members = members[np.argsort(starts[members], kind="stable")]
            source = None
            if audio is not None:
                try:
                    source = audio.path(entry["key"])
                except SystemExit as exc:  # AudioSource exits on a missing file or a failed fetch
                    print(f"  no audio for {entry['key']}: {exc}")
                    missing += 1
            windows = []
            for slot, i in enumerate(members, start=1):
                clip = ""
                if source is not None:
                    name = f"{group}_{rank:02d}_{entry['key']}_w{slot}.wav"
                    try:
                        wave = decode_window_through(source, float(starts[i]), window_config, chain).numpy()
                        write_wav(args.out / "clips" / name, wave, window_config.sample_rate)
                        clip = f"clips/{name}"
                    except (RuntimeError, subprocess.CalledProcessError) as exc:
                        print(f"  could not decode {entry['key']} @ {starts[i]:.1f} s: {exc}")
                windows.append({"slot": slot, "start": float(starts[i]), "tau2": float(tau2[i]),
                                "above_q95": bool(tau2[i] >= q95), "url": youtube(track, starts[i]),
                                "clip": clip})
            entry["windows"] = windows
            entry["window_tau2"] = "; ".join(f"{_clock(w['start'])}={w['tau2']:.2f}" for w in windows)
            report[group].append(entry)
            vs = f"{entry['tau2_vs_work']:.2f}" if entry["tau2_vs_work"] != "" else "-"
            print(f"{rank:>3} {entry['tau2']:>6.2f} {vs:>8} {entry['percentile']:>6.1f}  {entry['work']:>7}  "
                  f"{entry['key']:<16} {entry['artist'][:28]} - {entry['title'][:40]}")
        works = [e["work"] for e in report[group]]
        if works:
            print(f"    {len(set(works))} distinct works among {len(works)} recordings")

    write_csv(args.out / "tails.csv", [e for g in GROUPS for e in report[g]],
              ["group", "rank", "key", "work", "title", "artist", "n_windows", "tau2", "tau2_vs_work",
               "percentile", "flow_nll", "window_tau2", "url", "kind", "notes"])
    ranking = "τ² relative to the recording's work" if within else "τ²"
    intro = (f"{split}, {len(recs['keys'])} recordings, ranked by {ranking}. τ² = ‖w‖² / mean ‖w‖², "
             f"so 1 is an average window; a recording's τ² is the geometric mean over its windows. "
             f"&#9650; marks a window above the human 95 % quantile (τ² ≥ {q95:.2f}). Record what each "
             f"recording is (live, amateur, old, not music, studio) in <code>tails.csv</code> "
             f"(<code>kind</code>, <code>notes</code>).")
    write_index(args.out, f"Human tails of τ ({split})", intro, report)
    if missing:
        print(f"\n{missing} recordings have no clips; pass --audio-s3 or check --data-root")
    print(f"\n-> {args.out}/index.html, tails.csv, recordings.csv")


if __name__ == "__main__":
    main()
