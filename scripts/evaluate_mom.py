#!/usr/bin/env python3
"""Does the flow put synthetic music out of distribution? Human vs. MoM, per source.

Scores songs from five sources with a trained detection flow
(scripts/train_flow.py) and draws one horizontal 1-D axis per source:

    human        SHS100K held-out test works (test50 by default)
    udio         MoM, Udio v1.5
    riffusion    MoM, Riffusion FUZZ-1.0
    diffrhythm   MoM, DiffRhythm
    yue          MoM, YuE

A song's position on its axis is its detection score

    log p(style | content) + alpha * log p(content)

aggregated over its windows (the minimum by default, CLAUDE.md "Windowing").
Higher means the flow finds the song more typical of the human music it was
trained on. If the detector works, the four generators clump on the left of
their axes and human music sits to the right. Each generator's row reports the
AUROC of human against that generator (1 = perfectly separated, 0.5 = chance,
below 0.5 = the generator looks *more* human than human music, the Nalisnick
et al. pathology CLAUDE.md warns about) and the share of its songs below the
dashed line, the human 5th percentile.

How a song is scored, and why it matches training:

  * windows are S seconds, with S read from the flow checkpoint (the length the
    flow was trained on), cut by the same `decode_window` (ffmpeg -> 24 kHz mono)
  * K windows per song (--windows-per-song), centred in K equal slices of the
    song, so never at second 0 or flush with the end; deterministic, so a rerun
    scores the same audio
  * content and style tokens come from train_flow's own `encode_windows`, on the
    frozen encoder the flow was trained on top of

No loudness or codec normalization is applied, because none was applied in
training. CLAUDE.md flags the YouTube-vs-mp3 codec confound; this script
measures the detector as trained, confound included.

The per-window log-probs are stored rather than the final score, so --alpha and
--aggregate can be changed afterwards with --plot-only, without a GPU:

    {out_dir}/scores.jsonl            one line per song, every window's three terms
    {out_dir}/run.json                what was scored: checkpoints, S, K, seed
    {out_dir}/ood_axes.png            the figure (--figure renames it)
    {out_dir}/ood_axes_summary.json   per-source quartiles, AUROCs, detection rates

Scoring is resumable: songs already in scores.jsonl are skipped, and sources
are interleaved, so an interrupted run still covers all five evenly. Raising
--per-source later extends the same deterministic sample.

MoM audio is read from s3://BUCKET/{mom-prefix}/{generator}/ or, if that is
empty, from the fetch_mom.py layout {mom-prefix}/audio/{generator}/. Pass
--mom-root to read a local mirror instead. Human audio comes from --data-root,
as in training; --human-audio-key streams the tracks that are not on disk from
the same bucket.

Examples:
    python scripts/evaluate_mom.py --flow-checkpoint checkpoints/flow/flow_last.pt \\
        --data-root $DATA --bucket songs-and-mert-features

    # re-plot the pure conditional score (alpha = 0) from the stored windows
    python scripts/evaluate_mom.py --flow-checkpoint checkpoints/flow/flow_last.pt \\
        --plot-only --alpha 0
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from shs100k_meta import DEFAULT_DATA_ROOT  # noqa: E402

HUMAN = "human"
GENERATORS = ("udio", "riffusion", "diffrhythm", "yue")
# The HF repo spells DiffRhythm "diffrythm" and fetch_mom.py kept that spelling
# in the S3 key, so both are looked for.
GENERATOR_DIRS: dict[str, tuple[str, ...]] = {"diffrhythm": ("diffrhythm", "diffrythm")}
AUDIO_SUFFIXES = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".webm")

# The three per-window terms stored for every window.
TERMS = ("style_given_content", "content", "gaussian")

# What changing the run would invalidate: scores.jsonl is only reused when
# these match.
RUN_IDENTITY = (
    "flow_checkpoint",
    "encoder_checkpoint",
    "window_seconds",
    "windows_per_song",
    "seed",
    "human_split",
)

# Chart surface and ink, painted explicitly rather than inherited from a
# matplotlib style, so the figure is identical on any machine (same convention
# as scripts/plot_mert_features.py). Human and synthetic take categorical slots
# 1 and 2 in their fixed order.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
HUMAN_COLOR = "#2a78d6"  # blue
SYNTH_COLOR = "#eb6834"  # orange


# --------------------------------------------------------------------------
# Songs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Song:
    source: str  # "human" or a generator
    song_id: str  # track key, or the MoM file's path below its generator prefix
    location: str  # human: path relative to --data-root; MoM: S3 key or local path
    duration: float | None = None  # seconds, when a manifest already knows it


def deterministic_sample(songs: list[Song], n: int, seed: int) -> list[Song]:
    """Sort, shuffle by (seed, source), keep the first n (0 = all).

    A larger n extends the same sample rather than drawing a new one, which is
    what lets a finished run be topped up without rescoring.
    """
    if not songs:
        return []
    ordered = sorted(songs, key=lambda s: s.song_id)
    random.Random(f"{seed}:{ordered[0].source}").shuffle(ordered)
    return ordered[:n] if n else ordered


def interleave(groups: list[list[Song]]) -> list[Song]:
    """Round-robin across sources, so a partial run is balanced across them."""
    out: list[Song] = []
    for i in range(max((len(g) for g in groups), default=0)):
        out.extend(g[i] for g in groups if i < len(g))
    return out


def human_songs(data_root: Path, split: str, window_seconds: float) -> list[Song]:
    from src.data.manifest import read_tracks

    try:
        tracks = read_tracks(data_root, split)
    except FileNotFoundError:
        raise SystemExit(
            f"no manifest at {data_root}/manifests/{split}/tracks.jsonl. Build it "
            f"with build_manifest.py, or pull manifests.tgz from "
            f"$RUN_S3/preprocessing/ (docs/learning_curve_runbook.md)."
        ) from None
    return [
        Song(HUMAN, t.key, t.audio, t.duration)
        for t in tracks
        if t.duration >= window_seconds
    ]


def generator_dirs(generator: str) -> tuple[str, ...]:
    return GENERATOR_DIRS.get(generator, (generator,))


def mom_songs_s3(s3, prefix: str, generator: str) -> tuple[str, list[Song]]:
    """The generator's audio under the first candidate prefix that has any."""
    base = f"{prefix.strip('/')}/" if prefix.strip("/") else ""
    candidates = [f"{base}{d}/" for d in generator_dirs(generator)] + [
        f"{base}audio/{d}/" for d in generator_dirs(generator)
    ]
    for candidate in candidates:
        if not s3.sample(candidate, 1):
            continue
        keys = [k for k in s3.list_keys(candidate) if k.lower().endswith(AUDIO_SUFFIXES)]
        if keys:
            return candidate, [
                Song(generator, os.path.splitext(k[len(candidate):])[0], k)
                for k in keys
            ]
    lines = [f"no {generator} audio under any of:"]
    lines += [f"    s3://{s3.bucket}/{c}" for c in candidates]
    for folder in (base, f"{base}audio/"):
        found = s3.folders(folder)
        if found:
            lines += [f"folders under s3://{s3.bucket}/{folder}:"]
            lines += [f"    {f}" for f in found]
    lines += ["Set --mom-prefix (and --bucket) to match the bucket."]
    raise SystemExit("\n".join(lines))


def mom_songs_local(root: Path, generator: str) -> tuple[str, list[Song]]:
    for d in generator_dirs(generator):
        for base in (root / d, root / "audio" / d):
            if not base.is_dir():
                continue
            files = sorted(
                p for p in base.rglob("*")
                if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
            )
            if files:
                return str(base), [
                    Song(generator, str(p.relative_to(base).with_suffix("")), str(p))
                    for p in files
                ]
    raise SystemExit(
        f"no {generator} audio under {root}/{{{','.join(generator_dirs(generator))}}}/ "
        f"or {root}/audio/<generator>/"
    )


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------


class AudioOpener:
    """Resolve a song to a seekable local file for ffmpeg.

    ffmpeg seeks to cut a window (-ss before -i), so an S3 object is pulled to
    a temp file for the duration of one song's windows and deleted after: disk
    use is a few songs in flight, and each song is transferred once however many
    windows come from it.
    """

    def __init__(
        self,
        data_root: Path,
        s3,
        human_key: str | None,
        mom_local: bool,
        tmp_dir: Path | None,
    ) -> None:
        self.data_root = data_root
        self.s3 = s3
        self.human_key = human_key
        self.mom_local = mom_local
        self.tmp_dir = tmp_dir

    def local_path(self, song: Song) -> Path | None:
        if song.source == HUMAN:
            return self.data_root / song.location
        return Path(song.location) if self.mom_local else None

    def s3_key(self, song: Song) -> str | None:
        if song.source != HUMAN:
            return None if self.mom_local else song.location
        if not self.human_key:
            return None
        path = Path(song.location)
        return self.human_key.format(
            path=song.location, split=path.parent.name, track=path.stem, ext=path.suffix
        )

    @contextlib.contextmanager
    def open(self, song: Song) -> Iterator[Path]:
        local = self.local_path(song)
        if local is not None and local.exists():
            yield local
            return
        key = self.s3_key(song)
        if key is None or self.s3 is None:
            hint = " (pass --human-audio-key to stream it from S3)" if song.source == HUMAN else ""
            raise FileNotFoundError(f"{local}{hint}")
        handle, name = tempfile.mkstemp(suffix=Path(key).suffix, dir=self.tmp_dir)
        os.close(handle)
        path = Path(name)
        try:
            try:
                self.s3.get_file(key, path)
            except Exception as exc:  # name the key: a bare 404 says nothing
                raise RuntimeError(f"s3://{self.s3.bucket}/{key}: {exc}") from exc
            yield path
        finally:
            path.unlink(missing_ok=True)


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def window_starts(duration: float, window: float, count: int) -> list[float]:
    """K window starts, each centred in one of K equal slices of the song.

    Never at second 0 or flush with the end, where count-ins, fade-outs and a
    generator's padding sit. Rounded to the millisecond decode_window seeks to.
    """
    if duration < window or count < 1:
        return []
    span = duration - window
    return [round(span * (i + 0.5) / count, 3) for i in range(count)]


@dataclass
class Loaded:
    song: Song
    duration: float | None
    starts: list[float]
    waves: list  # torch tensors, (window_samples,)
    error: str | None = None


def prefetch(fn: Callable, items: Iterable, workers: int) -> Iterator:
    """Map fn over items on a thread pool, in order, with bounded lookahead.

    Bounded because each loaded song pins K decoded windows in host memory.
    """
    workers = max(workers, 1)
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        queue: deque = deque()
        for item in iterator:
            queue.append(pool.submit(fn, item))
            if len(queue) >= 2 * workers:
                break
        while queue:
            future = queue.popleft()
            item = next(iterator, None)
            if item is not None:
                queue.append(pool.submit(fn, item))
            yield future.result()


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class Scorer:
    """Frozen MERT + encoder + flows, loaded exactly as train_flow.py builds them."""

    def __init__(
        self, flow_path: Path, encoder_path: Path | None, device_name: str
    ) -> None:
        import torch

        from src.config import config_from_dict
        from src.models import ConditionalGaussian, FlowDetector, MertExtractor
        from src.training import pick_device
        from train_flow import encode_windows, load_frozen_encoder

        self.torch = torch
        self.encode_windows = encode_windows
        self.device = pick_device(device_name)

        checkpoint = torch.load(flow_path, map_location=self.device, weights_only=False)
        self.flow_config = config_from_dict(checkpoint["config"])
        stored = checkpoint.get("encoder_checkpoint")
        self.encoder_path = encoder_path or (Path(stored) if stored else None)
        if self.encoder_path is None or not self.encoder_path.exists():
            raise SystemExit(
                f"encoder checkpoint {self.encoder_path} (recorded in {flow_path}) "
                "not found; pass --encoder-checkpoint"
            )
        self.encoder, self.encoder_config = load_frozen_encoder(
            self.encoder_path, self.device
        )
        self.mert = MertExtractor(self.encoder_config.mert).to(self.device)

        bottleneck = self.encoder_config.bottleneck
        self.detector = FlowDetector(bottleneck, self.flow_config.flow).to(self.device)
        self.detector.load_state_dict(checkpoint["detector"])
        self.baseline = ConditionalGaussian(bottleneck, self.flow_config.flow).to(self.device)
        self.baseline.load_state_dict(checkpoint["baseline"])
        self.detector.eval()
        self.baseline.eval()
        if not bool(self.detector.style_flow.whitener.initialized):
            raise SystemExit(f"{flow_path}: whitening statistics were never fitted")

        self.window_seconds = float(self.flow_config.data.window_seconds)
        self.sample_rate = int(self.encoder_config.mert.sample_rate)
        self.micro_batch = int(self.encoder_config.mert.micro_batch)
        self.autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        self.epoch = checkpoint.get("epoch")
        self.step = checkpoint.get("step")

    def score(self, waves: list) -> dict[str, list[float]]:
        """Windows -> the three per-window log-probs, as plain floats."""
        torch = self.torch
        with torch.no_grad():
            batch = torch.stack(waves).to(self.device)
            content, style = self.encode_windows(
                self.mert, self.encoder, batch, self.micro_batch, self.autocast
            )
            terms = self.detector.log_probs(content, style)
            gaussian = self.baseline.log_prob(content, style)
        return {
            "style_given_content": terms["style_given_content"].float().cpu().tolist(),
            "content": terms["content"].float().cpu().tolist(),
            "gaussian": gaussian.float().cpu().tolist(),
        }


# --------------------------------------------------------------------------
# Scoring run
# --------------------------------------------------------------------------


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_run(out_dir: Path, identity: dict, extra: dict, overwrite: bool) -> None:
    """Refuse to append to scores computed under a different setup.

    `identity` must match the previous run; `extra` is recorded but free to
    change (the trained alpha, which --plot-only uses as its default).
    """
    run_path = out_dir / "run.json"
    scores_path = out_dir / "scores.jsonl"
    if run_path.exists() and scores_path.exists() and not overwrite:
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        changed = [
            f"  {k}: {previous.get(k)!r} -> {identity[k]!r}"
            for k in RUN_IDENTITY
            if previous.get(k) != identity[k]
        ]
        if changed:
            raise SystemExit(
                f"{scores_path} was scored under a different setup:\n"
                + "\n".join(changed)
                + "\nUse another --out-dir, or --overwrite to start over."
            )
    if overwrite:
        scores_path.unlink(missing_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps({**identity, **extra}, indent=1) + "\n", encoding="utf-8")


def score_run(args: argparse.Namespace) -> None:
    from src.data.windows import WindowConfig, decode_window
    from src.s3 import S3

    scorer = Scorer(args.flow_checkpoint, args.encoder_checkpoint, args.device)
    window_seconds = scorer.window_seconds
    window_config = WindowConfig(window_seconds=window_seconds, sample_rate=scorer.sample_rate)
    print(
        f"device={scorer.device.type}  flow={args.flow_checkpoint} "
        f"(epoch {scorer.epoch}, step {scorer.step})  encoder={scorer.encoder_path}"
    )
    print(
        f"windows: {args.windows_per_song} x {window_seconds:g} s per song "
        f"(the flow's training length)"
    )
    for label, config in (("flow", scorer.flow_config), ("encoder", scorer.encoder_config)):
        seen = {config.data.train_split, config.data.val_split}
        if args.human_split in seen:
            print(
                f"WARNING: the {label} was trained or validated on {args.human_split!r}; "
                "the human row is then not held out."
            )

    identity = {
        "flow_checkpoint": str(args.flow_checkpoint),
        "encoder_checkpoint": str(scorer.encoder_path),
        "window_seconds": window_seconds,
        "windows_per_song": args.windows_per_song,
        "seed": args.seed,
        "human_split": args.human_split,
    }
    check_run(args.out_dir, identity, {"alpha": float(scorer.flow_config.flow.alpha)}, args.overwrite)

    needs_s3 = args.mom_root is None or args.human_audio_key
    s3 = S3(args.bucket, args.region) if needs_s3 else None

    groups = [
        deterministic_sample(
            human_songs(args.data_root, args.human_split, window_seconds),
            args.per_source,
            args.seed,
        )
    ]
    print(f"{HUMAN:<11} {len(groups[0]):>5} songs  {args.data_root}/manifests/{args.human_split}")
    for generator in args.generators:
        if args.mom_root is not None:
            where, songs = mom_songs_local(args.mom_root, generator)
        else:
            where, songs = mom_songs_s3(s3, args.mom_prefix, generator)
            where = f"s3://{args.bucket}/{where}"
        picked = deterministic_sample(songs, args.per_source, args.seed)
        groups.append(picked)
        print(f"{generator:<11} {len(picked):>5} songs  of {len(songs):>6} under {where}")

    opener = AudioOpener(args.data_root, s3, args.human_audio_key, args.mom_root is not None, args.tmp_dir)
    if not args.human_audio_key:
        missing = [s for s in groups[0] if not opener.local_path(s).exists()]
        if missing:
            raise SystemExit(
                f"{len(missing)} of {len(groups[0])} human tracks are not under "
                f"--data-root, e.g. {opener.local_path(missing[0])}. Sync the audio, "
                "or pass --human-audio-key to stream them from the bucket, e.g. "
                "--human-audio-key '{path}' when the bucket mirrors the data root."
            )

    scores_path = args.out_dir / "scores.jsonl"
    done = {(r["source"], r["song"]) for r in read_records(scores_path)}
    todo = [s for s in interleave(groups) if (s.source, s.song_id) not in done]
    print(f"{len(done)} songs already scored, {len(todo)} to go -> {scores_path}")
    if not todo:
        return

    def load(song: Song) -> Loaded:
        try:
            with opener.open(song) as path:
                duration = song.duration if song.duration is not None else probe_duration(path)
                starts = window_starts(duration, window_seconds, args.windows_per_song)
                waves = [decode_window(path, start, window_config) for start in starts]
            return Loaded(song, duration, starts, waves)
        except Exception as exc:  # recorded and reported; the song is retried next run
            return Loaded(song, None, [], [], f"{type(exc).__name__}: {exc}")

    batch_size = args.batch or scorer.flow_config.data.batch_tracks
    pending: list[tuple[int, float, object]] = []  # (song slot, start, wave)
    open_songs: dict[int, dict] = {}  # slot -> record being filled
    failures: list[tuple[Song, str]] = []
    n_songs = n_windows = 0
    started = time.time()

    with open(scores_path, "a", encoding="utf-8") as out:

        def write(record: dict) -> None:
            nonlocal n_songs
            out.write(json.dumps(record) + "\n")
            out.flush()  # a killed run keeps every finished song
            n_songs += 1
            if n_songs % 25 == 0:
                elapsed = time.time() - started
                rate = n_windows / max(elapsed, 1e-9)
                left = (len(todo) - n_songs - len(failures)) * args.windows_per_song
                print(
                    f"[{n_songs}/{len(todo)}] songs  {rate:.1f} windows/s  "
                    f"eta {left / max(rate, 1e-9) / 60:.0f} min  failed {len(failures)}",
                    flush=True,
                )

        def flush(batch: list[tuple[int, float, object]]) -> None:
            """Score one batch; write every song whose windows are now all in."""
            nonlocal n_windows
            terms = scorer.score([wave for _, _, wave in batch])
            for i, (slot, start, _) in enumerate(batch):
                open_songs[slot]["windows"].append(
                    {"start": start, **{t: terms[t][i] for t in TERMS}}
                )
            n_windows += len(batch)
            for slot in [s for s, r in open_songs.items() if len(r["windows"]) == r["_n"]]:
                record = open_songs.pop(slot)
                del record["_n"]
                write(record)

        for slot, loaded in enumerate(prefetch(load, todo, args.fetch_workers)):
            song = loaded.song
            if loaded.error is not None:
                failures.append((song, loaded.error))
                continue
            record = {
                "source": song.source,
                "song": song.song_id,
                "location": song.location,
                "duration": round(float(loaded.duration), 3),
                "windows": [],
            }
            if not loaded.waves:  # shorter than one window: final, not retried
                record["skipped"] = f"shorter than the {window_seconds:g} s window"
                write(record)
                continue
            record["_n"] = len(loaded.waves)
            open_songs[slot] = record
            pending.extend((slot, s, w) for s, w in zip(loaded.starts, loaded.waves))
            while len(pending) >= batch_size:
                flush(pending[:batch_size])
                pending = pending[batch_size:]
        if pending:
            flush(pending)

    elapsed = time.time() - started
    print(f"scored {n_songs} songs, {n_windows} windows in {elapsed / 60:.1f} min")
    if failures:
        print(f"{len(failures)} songs failed (re-run to retry):")
        for song, error in failures[:20]:
            print(f"  {song.source}/{song.song_id}: {error}")


# --------------------------------------------------------------------------
# Song scores and metrics
# --------------------------------------------------------------------------


def aggregate(values: np.ndarray, method: str, percentile: float) -> float:
    """Song-level score from per-window scores (CLAUDE.md "Windowing")."""
    if method == "min":
        return float(values.min())
    if method == "mean":
        return float(values.mean())
    return float(np.percentile(values, percentile))


def views(alpha: float) -> dict[str, tuple[str, Callable[[dict], float]]]:
    """Name -> (label, per-window score). 'score' is the one that is plotted."""
    label = "log p(style | content)"
    if alpha:
        label += f" + {alpha:g} · log p(content)"
    return {
        "score": (label, lambda w: w["style_given_content"] + alpha * w["content"]),
        "style_given_content": ("log p(style | content)", lambda w: w["style_given_content"]),
        "content": ("log p(content)", lambda w: w["content"]),
        "gaussian": ("Gaussian baseline log p(style | content)", lambda w: w["gaussian"]),
    }


def song_scores(
    records: list[dict], view: Callable[[dict], float], method: str, percentile: float
) -> dict[str, np.ndarray]:
    by_source: dict[str, list[float]] = {}
    for record in records:
        if not record["windows"]:
            continue
        values = np.array([view(w) for w in record["windows"]], dtype=np.float64)
        by_source.setdefault(record["source"], []).append(
            aggregate(values, method, percentile)
        )
    return {source: np.array(v) for source, v in by_source.items()}


def auroc(human: np.ndarray, synthetic: np.ndarray) -> float:
    """P(human song scores higher than synthetic song), ties counted half.

    Mann-Whitney U with average ranks; -inf ranks as the lowest value.
    """
    human = human[~np.isnan(human)]
    synthetic = synthetic[~np.isnan(synthetic)]
    if len(human) == 0 or len(synthetic) == 0:
        return float("nan")
    pooled = np.concatenate([human, synthetic])
    _, inverse, counts = np.unique(pooled, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = (ends - (counts - 1) / 2.0)[inverse]
    u = ranks[: len(human)].sum() - len(human) * (len(human) + 1) / 2.0
    return float(u / (len(human) * len(synthetic)))


def finite_quantiles(values: np.ndarray, qs: list[float]) -> list[float]:
    """Quantiles that tolerate -inf (a window the flow puts at zero density)."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return [float("nan")] * len(qs)
    safe = np.clip(values, -1e300, 1e300)
    return [float(q) for q in np.quantile(safe, qs)]


def summarize(
    records: list[dict],
    order: list[str],
    alpha: float,
    method: str,
    percentile: float,
    fpr: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    per_view = {
        name: song_scores(records, fn, method, percentile)
        for name, (_, fn) in views(alpha).items()
    }
    scores = per_view["score"]
    human = scores.get(HUMAN, np.array([]))
    threshold = finite_quantiles(human, [fpr])[0] if len(human) else float("nan")
    skipped: dict[str, int] = {}
    for record in records:
        if not record["windows"]:
            skipped[record["source"]] = skipped.get(record["source"], 0) + 1

    sources: dict[str, dict] = {}
    for source in order:
        values = scores.get(source, np.array([]))
        q1, median, q3 = finite_quantiles(values, [0.25, 0.5, 0.75])
        entry = {
            "songs": int(len(values)),
            "skipped": skipped.get(source, 0),
            "nan": int(np.isnan(values).sum()),
            "q1": q1,
            "median": median,
            "q3": q3,
        }
        if source != HUMAN:
            entry["auroc"] = {
                name: auroc(per_view[name].get(HUMAN, np.array([])), per_view[name].get(source, np.array([])))
                for name in per_view
            }
            entry["below_threshold"] = (
                float(np.mean(values[~np.isnan(values)] < threshold)) if len(values) else float("nan")
            )
        sources[source] = entry

    synthetic = [s for s in order if s != HUMAN]
    pooled = {
        name: np.concatenate([per_view[name].get(s, np.array([])) for s in synthetic] or [np.array([])])
        for name in per_view
    }
    summary = {
        "alpha": alpha,
        "aggregate": method,
        "percentile": percentile if method == "percentile" else None,
        "views": {name: label for name, (label, _) in views(alpha).items()},
        "threshold": {"human_quantile": fpr, "value": threshold},
        "sources": sources,
        "all_synthetic": {
            "songs": int(len(pooled["score"])),
            "auroc": {
                name: auroc(per_view[name].get(HUMAN, np.array([])), pooled[name])
                for name in per_view
            },
            "below_threshold": (
                float(np.mean(pooled["score"][~np.isnan(pooled["score"])] < threshold))
                if len(pooled["score"])
                else float("nan")
            ),
        },
    }
    return summary, scores


def print_summary(summary: dict, order: list[str]) -> None:
    fpr = summary["threshold"]["human_quantile"]
    print(f"\nsong score = {summary['views']['score']}, {summary['aggregate']} over windows")
    print(
        f"threshold = human {fpr:.0%} quantile = {summary['threshold']['value']:.2f} "
        f"(estimated on the same human songs, so 'below' is in-sample)\n"
    )
    header = (
        f"{'source':<12}{'songs':>6}{'median':>11}"
        f"{'AUROC':>8}{'s|c':>7}{'c':>7}{'gauss':>7}{'below':>8}"
    )
    print(header)
    print("-" * len(header))
    rows = [(s, summary["sources"][s]) for s in order] + [("all AI", summary["all_synthetic"])]
    for name, entry in rows:
        line = f"{name:<12}{entry['songs']:>6}"
        line += f"{entry['median']:>11.1f}" if "median" in entry else f"{'':>11}"
        if "auroc" in entry:
            a = entry["auroc"]
            line += (
                f"{a['score']:>8.3f}{a['style_given_content']:>7.3f}"
                f"{a['content']:>7.3f}{a['gaussian']:>7.3f}{entry['below_threshold']:>8.1%}"
            )
        print(line)
    print(
        "\nAUROC = P(human scores higher). s|c = log p(style | content) alone, "
        "c = log p(content) alone,\ngauss = the conditional-Gaussian baseline; "
        f"below = share of songs under the threshold (detected at {fpr:.0%} FPR)."
    )


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------


def plot(
    scores: dict[str, np.ndarray],
    order: list[str],
    summary: dict,
    run: dict,
    out: Path,
    xlim: tuple[float, float] | None,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: this runs on an EC2 box with no display
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    finite = np.concatenate(
        [v[np.isfinite(v)] for v in scores.values() if len(v)] or [np.array([0.0])]
    )
    if xlim is not None:
        lo, hi = xlim
    else:
        lo, hi = (float(q) for q in np.quantile(finite, [0.005, 0.995]))
        pad = 0.05 * ((hi - lo) or 1.0)
        lo, hi = lo - pad, hi + pad
    threshold = summary["threshold"]["value"]

    n = len(order)
    row, gap, top, bottom = 0.6, 0.42, 1.35, 0.8  # inches
    width_in = 11.0
    height_in = top + n * row + (n - 1) * gap + bottom
    left, plot_w = 0.12, 0.71
    fig = plt.figure(figsize=(width_in, height_in), facecolor=SURFACE)
    rng = np.random.default_rng(0)  # fixed jitter: the figure is reproducible

    axes = []
    for i, source in enumerate(order):
        y0 = (bottom + (n - 1 - i) * (row + gap)) / height_in
        ax = fig.add_axes(
            (left, y0, plot_w, row / height_in), sharex=axes[0] if axes else None
        )
        axes.append(ax)
        values = scores.get(source, np.array([]))
        values = values[~np.isnan(values)]
        color = HUMAN_COLOR if source == HUMAN else SYNTH_COLOR

        ax.set_facecolor(SURFACE)
        for side in ("top", "left", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK_2)
        ax.set_yticks([])
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(lo, hi)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=INK_2, labelsize=9, length=3)
        if i < n - 1:
            ax.tick_params(labelbottom=False)

        if np.isfinite(threshold) and lo <= threshold <= hi:
            ax.axvline(threshold, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)

        jitter = rng.uniform(-0.3, 0.3, len(values))
        below, above = values < lo, values > hi
        inside = ~below & ~above
        ax.scatter(values[inside], jitter[inside], s=13, color=color, alpha=0.5, linewidths=0, zorder=3)
        for mask, edge, marker, ha in ((below, lo, "<", "left"), (above, hi, ">", "right")):
            if mask.any():
                ax.scatter(
                    np.full(mask.sum(), edge), jitter[mask], s=22, marker=marker,
                    color=color, alpha=0.75, linewidths=0, zorder=3, clip_on=False,
                )
                ax.text(
                    edge, 0.47, f"{mask.sum()} beyond", ha=ha, va="bottom",
                    fontsize=8, color=INK_2,
                )

        entry = summary["sources"][source]
        if len(values):
            q1, median, q3 = entry["q1"], entry["median"], entry["q3"]
            a, b = max(q1, lo), min(q3, hi)
            if b > a:
                ax.add_patch(
                    Rectangle((a, -0.42), b - a, 0.84, facecolor=color, alpha=0.14, linewidth=0, zorder=2)
                )
            if lo <= median <= hi:
                ax.plot([median, median], [-0.42, 0.42], color=INK, linewidth=1.8, zorder=4, solid_capstyle="butt")

        ax.text(
            -0.015, 0.5, source, transform=ax.transAxes, ha="right", va="center",
            fontsize=12, color=INK, fontweight="bold" if source == HUMAN else "normal",
        )
        if source == HUMAN:
            note = f"n = {entry['songs']}\nreference"
        else:
            note = (
                f"n = {entry['songs']}\nAUROC {entry['auroc']['score']:.3f}\n"
                f"{entry['below_threshold']:.0%} below line"
            )
        ax.text(
            1.035, 0.5, note, transform=ax.transAxes, ha="left", va="center",
            fontsize=9, color=INK_2, linespacing=1.35,
        )

    axes[-1].set_xlabel(
        f"song score = {summary['views']['score']}   (nats)",
        color=INK, fontsize=10, labelpad=8,
    )

    def at(inches_from_top: float) -> float:
        return 1.0 - inches_from_top / height_in

    flow_name = Path(run.get("flow_checkpoint", "flow")).name
    fig.text(left, at(0.36), "How in-distribution does the flow find each song?",
             fontsize=14, color=INK, fontweight="bold", ha="left")
    method = summary["aggregate"]
    if method == "percentile":
        method = f"{summary['percentile']:g}th percentile"
    fig.text(
        left, at(0.66),
        f"{flow_name} · {method} over {run.get('windows_per_song', '?')} × "
        f"{run.get('window_seconds', '?'):g} s windows per song · one dot per song · "
        f"band = IQR, bar = median · dashed = human {summary['threshold']['human_quantile']:.0%} quantile",
        fontsize=9, color=INK_2, ha="left",
    )
    fig.text(left, at(top - 0.12), "← less like the human training music (OOD)",
             fontsize=9.5, color=INK_2, ha="left", va="bottom")
    fig.text(left + plot_w, at(top - 0.12), "more like it →",
             fontsize=9.5, color=INK_2, ha="right", va="bottom")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def report(args: argparse.Namespace) -> None:
    records = read_records(args.out_dir / "scores.jsonl")
    if not records:
        raise SystemExit(f"nothing scored yet in {args.out_dir / 'scores.jsonl'}")
    run_path = args.out_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}

    present = {r["source"] for r in records}
    order = [s for s in (HUMAN, *args.generators) if s in present]
    if HUMAN not in present:
        raise SystemExit("no human songs scored: there is nothing to compare against")
    alpha = args.alpha if args.alpha is not None else run.get("alpha", 1.0)

    summary, scores = summarize(records, order, alpha, args.aggregate, args.percentile, args.fpr)
    summary["run"] = run
    print_summary(summary, order)

    figure = args.out_dir / args.figure
    plot(scores, order, summary, run, figure, tuple(args.xlim) if args.xlim else None, args.dpi)
    # named after the figure, so a re-plot with another alpha keeps both
    summary_path = figure.with_name(f"{figure.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"\nfigure:  {figure}\nsummary: {summary_path}")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--flow-checkpoint", type=Path, required=True,
                        help="flow_last.pt written by train_flow.py")
    parser.add_argument("--encoder-checkpoint", type=Path,
                        help="default: the one recorded in the flow checkpoint")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="holds manifests/{human-split}/ and the human audio")
    parser.add_argument("--human-split", default="test50",
                        help="SHS100K split of human songs (default: test50, which "
                        "neither the encoder nor the flow selects on)")
    parser.add_argument("--human-audio-key", default=None,
                        help="S3 key template for human tracks missing from --data-root. "
                        "Fields: {path} (manifest path, e.g. audio/test/123_456.webm), "
                        "{split} (its folder, e.g. test), {track}, {ext}")
    parser.add_argument("--bucket", default="songs-and-mert-features")
    parser.add_argument("--region", default=None)
    parser.add_argument("--mom-prefix", default="mom",
                        help="S3 prefix of the MoM mirror (default: mom)")
    parser.add_argument("--mom-root", type=Path, default=None,
                        help="read MoM from this local directory instead of S3")
    parser.add_argument("--generators", nargs="+", default=list(GENERATORS),
                        help="MoM generators, in plotting order")
    parser.add_argument("--per-source", type=int, default=300,
                        help="songs per source, sampled deterministically (0 = all)")
    parser.add_argument("--windows-per-song", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="song sampling seed")
    parser.add_argument("--batch", type=int, default=0,
                        help="windows per encoder call (0 = the flow's batch_tracks)")
    parser.add_argument("--fetch-workers", type=int, default=8,
                        help="threads downloading and decoding songs ahead of the GPU")
    parser.add_argument("--tmp-dir", type=Path, default=None,
                        help="where S3 audio is staged (default: system temp)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: mom_eval/ next to the flow checkpoint")
    parser.add_argument("--overwrite", action="store_true",
                        help="discard scores.jsonl and score from scratch")
    parser.add_argument("--plot-only", action="store_true",
                        help="skip scoring; re-plot what scores.jsonl already holds")
    parser.add_argument("--alpha", type=float, default=None,
                        help="weight of log p(content) (default: the flow config's)")
    parser.add_argument("--aggregate", choices=("min", "percentile", "mean"), default="min",
                        help="per-window -> song score (default: min, CLAUDE.md)")
    parser.add_argument("--percentile", type=float, default=5.0,
                        help="with --aggregate percentile")
    parser.add_argument("--fpr", type=float, default=0.05,
                        help="human quantile drawn as the dashed threshold")
    parser.add_argument("--xlim", type=float, nargs=2, metavar=("LO", "HI"),
                        help="x range; songs beyond it are drawn as arrows at the edge")
    parser.add_argument("--figure", default="ood_axes.png")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = args.flow_checkpoint.resolve().parent / "mom_eval"
    if args.windows_per_song < 1:
        parser.error("--windows-per-song must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    if not args.plot_only:
        score_run(args)
    report(args)


if __name__ == "__main__":
    main()
