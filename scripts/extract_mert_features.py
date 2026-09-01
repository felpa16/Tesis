#!/usr/bin/env python3
"""Materialize the phase-2 training set: MERT layer mixes for aligned window pairs.

The training sample is not a track, it is an S-second window. This script walks
`manifests/{split}/pairs.jsonl`, samples aligned window pairs through each
pair's stored warping path exactly the way `AlignedPairDataset` does, runs
frozen MERT on each window, applies the two softmax layer-weight vectors frozen
at the end of phase 1, and uploads the results to

    s3://BUCKET/mert-features/content/{split}/{window_id}.npy    (N,1024) fp16
    s3://BUCKET/mert-features/style/{split}/{window_id}.npy

A window id is "{track_key}_{start_ms:08d}", so an object names the track and
the exact offset it came from. Which windows form a pair is recorded in
`manifests/{split}/window_pairs.jsonl` (`WindowPairEntry`), written locally and
uploaded alongside the features; that file is what phase-2 training reads.

Why the window is the right unit
--------------------------------
MERT features are not a local function of the audio -- the segment you feed the
model is part of the feature definition. Two mechanisms, both measured on this
model (see timeline.md):

  * The conv frontend's first layer is GroupNorm with one group per channel, so
    each channel is normalised over the *whole* input. The same 10 s of audio,
    alone versus inside a 30 s clip, already differs by a per-frame cosine of
    0.93 -- before a single attention layer runs.
  * The encoder is global. Chunking it and discarding 1 s / 2.5 s / 5 s of
    context per side gives last-layer cosines of 0.85 / 0.88 / 0.92 against an
    unchunked forward; it converges only as the margin approaches the input.

So there is no way to cache a whole track and slice windows out of it later
without changing the features. Caching the window itself sidesteps this
completely: each cached window is one `MERTModel.forward` on exactly the
waveform `decode_window` produces, which is bit-for-bit what online extraction
in `scripts/train.py` computes for that window.

Anchors are chosen deterministically -- evenly spaced over the alignment
points whose windows fit inside both covers -- so a rerun reproduces the same
training set rather than resampling it.

Storage: one 20 s window is 1,499 x 1024 fp16 = 3.07 MB per stream, so a pair
sample (two windows, two streams) is ~12.3 MB. The 8,570 surviving train pairs
at --windows-per-pair 4 come to ~420 GB.

Examples:
    # measure first: 50 pairs, one window each, reports stream similarity
    python scripts/extract_mert_features.py --bucket BUCKET --data-root $DATA \\
        --limit 50 --windows-per-pair 1

    # the real run
    python scripts/extract_mert_features.py --bucket BUCKET --data-root $DATA \\
        --windows-per-pair 4

    # two GPUs, one shard each
    python scripts/extract_mert_features.py --bucket BUCKET --data-root $DATA \\
        --shards 2 --shard 0
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from src.config import DataConfig, MertConfig, config_from_dict  # noqa: E402
from src.data.manifest import (  # noqa: E402
    PairEntry,
    TrackEntry,
    WindowPairEntry,
    manifest_dir,
    read_pairs,
    read_tracks,
    write_jsonl,
)
from src.data.windows import WindowConfig, _clamped_start, decode_window  # noqa: E402
from src.models.mert import LayerMix, MertExtractor  # noqa: E402
from src.training import pick_device  # noqa: E402

BRANCHES = ("content", "style")


# --------------------------------------------------------------------------
# Window identity
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """One window to run through MERT, identified by track key and offset."""

    key: str
    audio: str  # path relative to the data root
    start: float  # seconds, already rounded to ms

    @property
    def window_id(self) -> str:
        return f"{self.key}_{round(self.start * 1000):08d}"


def quantize(start: float) -> float:
    """Round a window start to the millisecond `decode_window` actually seeks to.

    decode_window formats -ss with three decimals, so quantizing here makes the
    window id and the decoded audio a one-to-one pair.
    """
    return round(start, 3)


def feature_key(prefix: str, branch: str, split: str, window_id: str) -> str:
    return f"{prefix.strip('/')}/{branch}/{split}/{window_id}.npy"


def frames_per_window(config: WindowConfig, geometry: "FrameGeometry") -> int:
    return geometry.n_frames(config.window_samples)


# --------------------------------------------------------------------------
# MERT frame geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameGeometry:
    """The conv frontend's sample->frame map, read off the model config.

    hop is the product of the conv strides (320 for MERT-v1-330M) and
    receptive_field the number of samples the first output frame consumes
    (400), so a window of L samples yields (L - rf) // hop + 1 frames. Read off
    the model config rather than hardcoded, since the 95M variant differs.
    """

    hop: int
    receptive_field: int

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> "FrameGeometry":
        config = model.config
        hop = 1
        for stride in config.conv_stride:
            hop *= stride
        field = 1
        for kernel, stride in zip(
            reversed(config.conv_kernel), reversed(config.conv_stride)
        ):
            field = (field - 1) * stride + kernel
        return cls(hop=hop, receptive_field=field)

    def n_frames(self, n_samples: int) -> int:
        if n_samples < self.receptive_field:
            return 0
        return (n_samples - self.receptive_field) // self.hop + 1


# --------------------------------------------------------------------------
# Sampling aligned window pairs
# --------------------------------------------------------------------------


def pair_anchors(
    data_root: Path,
    pair: PairEntry,
    durations: dict[str, float],
    config: WindowConfig,
    count: int,
) -> list[tuple[float, float]]:
    """Evenly spaced aligned (t_a, t_b) anchors from a pair's warping path.

    Same validity test as AlignedPairDataset._anchor -- an anchor is usable only
    if a full window centered on it fits inside *both* covers -- but spread
    deterministically over the valid points instead of drawn at random, so the
    materialized training set is reproducible and covers the aligned span
    rather than clustering wherever the RNG landed.
    """
    data = np.load(data_root / pair.alignment)
    t_a, t_b = data["t_a"], data["t_b"]
    half = config.window_seconds / 2.0
    dur_a = durations[pair.key_a]
    dur_b = durations[pair.key_b]
    valid = (
        (t_a >= half)
        & (t_a <= dur_a - half)
        & (t_b >= half)
        & (t_b <= dur_b - half)
    )
    indices = np.flatnonzero(valid)
    if len(indices) == 0:  # short tracks: fall back to the path midpoint
        indices = np.array([len(t_a) // 2])
    picks = indices[
        np.unique(np.linspace(0, len(indices) - 1, count).round().astype(int))
    ]
    return [(float(t_a[i]), float(t_b[i])) for i in picks]


def plan_samples(
    data_root: Path,
    pairs: list[PairEntry],
    tracks: list[TrackEntry],
    config: WindowConfig,
    windows_per_pair: int,
    n_frames: int,
) -> tuple[list[WindowPairEntry], dict[str, Window]]:
    """Pairs -> the window pairs to materialize and the unique windows to compute.

    A window shared by several pairs (the same cover aligned against two others,
    landing on the same offset) is computed and stored once; the manifest just
    references its id twice.
    """
    durations = {t.key: t.duration for t in tracks}
    audio = {t.key: t.audio for t in tracks}
    samples: list[WindowPairEntry] = []
    windows: dict[str, Window] = {}

    def add(key: str, anchor: float) -> tuple[Window, float]:
        start = quantize(
            _clamped_start(anchor, durations[key], config.window_seconds)
        )
        window = Window(key=key, audio=audio[key], start=start)
        windows.setdefault(window.window_id, window)
        return window, start

    for pair in pairs:
        if (
            durations.get(pair.key_a, 0.0) < config.window_seconds
            or durations.get(pair.key_b, 0.0) < config.window_seconds
        ):
            continue
        for anchor_a, anchor_b in pair_anchors(
            data_root, pair, durations, config, windows_per_pair
        ):
            window_a, start_a = add(pair.key_a, anchor_a)
            window_b, start_b = add(pair.key_b, anchor_b)
            samples.append(
                WindowPairEntry(
                    song_id=pair.song_id,
                    key_a=pair.key_a,
                    key_b=pair.key_b,
                    anchor_a=anchor_a,
                    anchor_b=anchor_b,
                    start_a=start_a,
                    start_b=start_b,
                    score=pair.score,
                    oti=pair.oti,
                    window_a=window_a.window_id,
                    window_b=window_b.window_id,
                    n_frames=n_frames,
                )
            )
    return samples, windows


# --------------------------------------------------------------------------
# S3
# --------------------------------------------------------------------------


class S3:
    """Thin boto3 wrapper handing each thread its own client.

    boto3 clients are not documented as thread-safe, and this script uploads
    from a worker pool while the main thread runs MERT.
    """

    def __init__(self, bucket: str, region: str | None = None) -> None:
        import boto3

        self.bucket = bucket
        self._session = boto3.session.Session(region_name=region)
        self._local = threading.local()

    @property
    def client(self):
        client = getattr(self._local, "client", None)
        if client is None:
            client = self._session.client("s3")
            self._local.client = client
        return client

    def list_keys(self, prefix: str) -> list[str]:
        keys = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", ()):
                if not obj["Key"].endswith("/") and obj["Size"] > 0:
                    keys.append(obj["Key"])
        return keys

    def put_array(self, key: str, array: np.ndarray) -> int:
        buffer = io.BytesIO()
        np.save(buffer, array)
        size = buffer.tell()
        buffer.seek(0)
        self.client.upload_fileobj(buffer, self.bucket, key)
        return size

    def put_file(self, key: str, path: Path) -> None:
        self.client.upload_file(str(path), self.bucket, key)

    def get_file(self, key: str, path: Path) -> None:
        self.client.download_file(self.bucket, key, str(path))


# --------------------------------------------------------------------------
# Audio source
# --------------------------------------------------------------------------


class AudioSource:
    """Resolve a track's audio to a seekable local file.

    ffmpeg needs to seek to cut a window (-ss before -i), which rules out
    piping an S3 object through stdin. With --s3-audio the object is pulled to
    a temp file for the duration of one track's windows and deleted after, so
    the instance never holds more than the tracks currently in flight -- a few
    MB each -- and each track is transferred exactly once no matter how many
    windows come from it.
    """

    def __init__(
        self,
        data_root: Path,
        s3: "S3 | None",
        prefix: str,
        tmp_dir: Path | None,
    ) -> None:
        self.data_root = data_root
        self.s3 = s3
        self.prefix = prefix.strip("/")
        self.tmp_dir = tmp_dir
        self.fetched = 0
        self.bytes_in = 0
        self._lock = threading.Lock()

    def object_key(self, relative: str) -> str:
        """Manifest paths are relative to the data root; S3 mirrors that layout."""
        return f"{self.prefix}/{relative}" if self.prefix else relative

    @contextlib.contextmanager
    def open(self, relative: str):
        local = self.data_root / relative
        if local.exists() or self.s3 is None:
            yield local
            return
        suffix = Path(relative).suffix
        handle, name = tempfile.mkstemp(suffix=suffix, dir=self.tmp_dir)
        os.close(handle)
        path = Path(name)
        try:
            self.s3.get_file(self.object_key(relative), path)
            with self._lock:
                self.fetched += 1
                self.bytes_in += path.stat().st_size
            yield path
        finally:
            path.unlink(missing_ok=True)


def group_windows(windows: list[Window], cap: int) -> list[list[Window]]:
    """Group windows by source track so each track is opened (or fetched) once.

    Groups are capped so one popular cover appearing in dozens of pairs cannot
    pin an unbounded number of decoded waveforms in host memory; with local
    audio a split group costs nothing, and over S3 it costs one extra GET of a
    ~4 MB object.
    """
    by_track: dict[str, list[Window]] = {}
    for window in windows:
        by_track.setdefault(window.key, []).append(window)
    groups: list[list[Window]] = []
    for key in sorted(by_track):
        ordered = sorted(by_track[key], key=lambda w: w.start)
        for i in range(0, len(ordered), cap):
            groups.append(ordered[i : i + cap])
    return groups


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def load_layer_mixes(
    checkpoint: dict, n_hidden_states: int, device: torch.device, path: Path
) -> dict[str, LayerMix]:
    """Rebuild both LayerMix modules from a phase-1 layer-weights file.

    The logits are the source of truth; the stored softmax is used only to
    verify that this file's weights survive the round trip, which catches a
    checkpoint written by a different LayerMix parameterization.
    """
    mixes: dict[str, LayerMix] = {}
    for branch in BRANCHES:
        logits = checkpoint[f"{branch}_logits"]
        if logits.numel() != n_hidden_states:
            raise SystemExit(
                f"{path}: {branch} weights cover {logits.numel()} hidden states, "
                f"but the config expects {n_hidden_states}"
            )
        mix = LayerMix(n_hidden_states)
        with torch.no_grad():
            mix.weights.copy_(logits)
        stored = checkpoint.get(f"{branch}_softmax")
        if stored is not None and not torch.allclose(
            mix.softmax_weights, stored, atol=1e-5
        ):
            raise SystemExit(
                f"{path}: {branch} softmax does not reproduce from its logits"
            )
        mixes[branch] = mix.to(device).eval()
    return mixes


@torch.no_grad()
def extract_batch(
    waves: torch.Tensor,
    mert: MertExtractor,
    mixes: dict[str, LayerMix],
    autocast,
) -> dict[str, np.ndarray]:
    """(B, window_samples) waveforms -> {'content': (B,N,1024), 'style': ...} fp16.

    One MERTModel.forward per batch entry, which is the same call the online
    training loop makes for a window, so the cached features are interchangeable
    with running MERT live.
    """
    with autocast:
        hidden = mert(waves)  # (B, 25, N, 1024)
        return {
            branch: mix(hidden).float() for branch, mix in mixes.items()
        }


def similarity(content: torch.Tensor, style: torch.Tensor) -> dict[str, float]:
    """Frame-wise cosine between the two mixes, raw and mean-centered.

    Raw cosine is inflated by the large DC offset the two mixes share (both are
    convex combinations of the same 25 hidden states). The centered figure --
    each window minus its own mean frame -- is the one that says whether the two
    streams carry different information, i.e. whether caching both earns its
    storage (timeline.md, "cache one stream or two?").
    """
    raw = torch.cosine_similarity(content, style, dim=-1).mean().item()
    centered = torch.cosine_similarity(
        content - content.mean(-2, keepdim=True),
        style - style.mean(-2, keepdim=True),
        dim=-1,
    ).mean().item()
    return {"cos": raw, "cos_centered": centered}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket for the features")
    parser.add_argument("--region", default=None, help="AWS region of the bucket")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="directory holding audio/, alignments/ and manifests/",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--out-prefix",
        default="mert-features",
        help="S3 prefix for the cached mixes (default: mert-features)",
    )
    parser.add_argument(
        "--layer-weights",
        type=Path,
        default=Path("checkpoints/run2-pool16/phase1_layer_weights.pt"),
        help="phase-1 layer-weights file (content/style logits + softmax)",
    )
    parser.add_argument(
        "--windows-per-pair",
        type=int,
        default=4,
        help="aligned windows sampled per cover pair, spread evenly over the "
        "usable part of the warping path",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DataConfig().window_seconds,
        help="window length S; the whole point of caching windows rather than "
        "tracks is that this is fixed at extraction time",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="extra alignment-quality floor on top of the manifest's own",
    )
    parser.add_argument(
        "--s3-audio",
        action="store_true",
        help="read source audio from S3 instead of requiring it under "
        "--data-root: each track is pulled to a temp file, all of its windows "
        "are cut from it, and it is deleted. Manifests and alignments are "
        "still read locally (88 MB for train, so sync those).",
    )
    parser.add_argument(
        "--audio-prefix",
        default="",
        help="S3 prefix in front of the manifest's audio paths; empty means "
        "the bucket mirrors the data root (audio/{split}/KEY.ext)",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="where --s3-audio stages tracks (default: system temp). Point it "
        "at instance NVMe rather than a small root volume.",
    )
    parser.add_argument(
        "--group-max",
        type=int,
        default=32,
        help="max windows decoded from one track at a time",
    )
    parser.add_argument("--batch", type=int, default=0, help="0 = MertConfig.micro_batch")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp32", action="store_true", help="no bf16 autocast")
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--upload-workers", type=int, default=8)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="use N pairs spread evenly across the manifest; 0 = all of them",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="re-extract windows already in S3"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan the work and estimate storage, then exit",
    )
    args = parser.parse_args()

    if not 0 <= args.shard < args.shards:
        raise SystemExit(f"--shard must be in [0, {args.shards})")
    if not args.data_root or not args.data_root.is_dir():
        raise SystemExit(f"--data-root is not a directory: {args.data_root!r}")
    if args.windows_per_pair < 1:
        raise SystemExit("--windows-per-pair must be at least 1")

    device = pick_device(args.device)
    weights_path = args.layer_weights
    if not weights_path.exists():
        raise SystemExit(f"layer weights not found: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    # keep the MERT variant, sample rate and micro-batch identical to phase 1
    mert_config = (
        config_from_dict(checkpoint["config"]).mert
        if "config" in checkpoint
        else MertConfig()
    )
    mixes = load_layer_mixes(
        checkpoint, mert_config.n_hidden_states, device, weights_path
    )
    mert = MertExtractor(mert_config).to(device)
    geometry = FrameGeometry.from_model(mert.model)
    window_config = WindowConfig(
        window_seconds=args.window_seconds, sample_rate=mert_config.sample_rate
    )
    n_frames = frames_per_window(window_config, geometry)
    batch_size = args.batch or mert_config.micro_batch
    print(
        f"device={device}  mert={mert_config.model_name}  "
        f"layer weights={weights_path} (epoch {checkpoint.get('epoch')}, "
        f"step {checkpoint.get('step')})"
    )
    print(
        f"window: {window_config.window_seconds:g} s = "
        f"{window_config.window_samples} samples = {n_frames} frames, "
        f"{batch_size} per forward"
    )

    tracks = read_tracks(args.data_root, args.split)
    pairs = [p for p in read_pairs(args.data_root, args.split) if p.score >= args.min_score]
    if args.limit and args.limit < len(pairs):
        # Spread the sample across the manifest instead of taking the head:
        # pairs.jsonl is ordered by alignment filename, so the first N all come
        # from the lowest song ids, which is a poor basis for the one-stream-
        # or-two decision this flag exists to inform.
        picks = np.unique(
            np.linspace(0, len(pairs) - 1, args.limit).round().astype(int)
        )
        pairs = [pairs[i] for i in picks]
    samples, windows = plan_samples(
        args.data_root, pairs, tracks, window_config, args.windows_per_pair, n_frames
    )
    per_window = n_frames * mert_config.dim * 2  # fp16, one stream
    print(
        f"[{args.split}] {len(pairs)} pairs -> {len(samples)} window pairs, "
        f"{len(windows)} unique windows "
        f"({human_bytes(len(windows) * per_window * len(BRANCHES))} of features)"
    )

    s3 = S3(args.bucket, args.region)
    todo = sorted(windows, key=lambda w: (windows[w].key, windows[w].start))
    if not args.overwrite:
        cached = {
            branch: {
                Path(k).stem
                for k in s3.list_keys(
                    f"{args.out_prefix.strip('/')}/{branch}/{args.split}/"
                )
            }
            for branch in BRANCHES
        }
        done = cached["content"] & cached["style"]  # both streams, or redo it
        todo = [w for w in todo if w not in done]
        print(f"{len(done)} windows already cached, {len(todo)} to compute")

    # Shard by track group, not by window: two shards must never fetch and
    # decode the same source file.
    groups = group_windows([windows[w] for w in todo], args.group_max)
    if args.shards > 1:
        groups = [g for i, g in enumerate(groups) if i % args.shards == args.shard]
        print(
            f"shard {args.shard}/{args.shards}: {len(groups)} track groups, "
            f"{sum(len(g) for g in groups)} windows"
        )

    # The manifest describes the whole planned set, not just this shard's share,
    # so every shard writes the same file and training sees one coherent index.
    manifest_path = manifest_dir(args.data_root, args.split) / "window_pairs.jsonl"
    write_jsonl(manifest_path, samples)
    print(f"window manifest: {manifest_path}")

    if args.dry_run or not groups:
        return

    source = AudioSource(
        args.data_root,
        s3 if args.s3_audio else None,
        args.audio_prefix,
        args.tmp_dir,
    )
    if not args.s3_audio:
        missing = sorted(
            {g[0].audio for g in groups if not (args.data_root / g[0].audio).exists()}
        )
        if missing:
            raise SystemExit(
                f"{len(missing)} source files are not under --data-root, e.g. "
                f"{missing[0]}. Pass --s3-audio to stream them from the bucket "
                f"instead of downloading the corpus."
            )
    else:
        print(
            f"streaming audio from s3://{args.bucket}/"
            f"{source.object_key('audio/' + args.split)}/ "
            f"({len(groups)} track fetches, staged in "
            f"{args.tmp_dir or tempfile.gettempdir()})"
        )

    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and not args.fp32
        else torch.autocast(device.type, enabled=False)
    )
    decode_pool = ThreadPoolExecutor(max_workers=max(args.decode_workers, 1))
    upload_pool = ThreadPoolExecutor(max_workers=max(args.upload_workers, 1))

    def fetch(group: list[Window]) -> list[tuple[str, torch.Tensor]]:
        """One track: open it once, cut every window it owes, release it."""
        with source.open(group[0].audio) as path:
            return [
                (w.window_id, decode_window(path, w.start, window_config))
                for w in group
            ]

    uploads: list[tuple[str, Future]] = []
    failures: list[tuple[str, str]] = []
    stat_sums = {"cos": 0.0, "cos_centered": 0.0}
    total_windows = sum(len(g) for g in groups)
    written = 0
    processed = 0
    batches = 0
    started = time.time()

    def drain_uploads(keep: int) -> None:
        """Bound in-flight uploads; each one pins a 3 MB array in host memory.

        A failed upload is recorded rather than raised: it must not end a long
        run, and because a window counts as cached only when *both* streams are
        in S3, the next run picks it up again.
        """
        while len(uploads) > keep:
            label, future = uploads.pop(0)
            try:
                future.result()
            except Exception as exc:
                failures.append((label, f"upload failed: {type(exc).__name__}: {exc}"))

    queue: deque[tuple[list[Window], Future]] = deque()
    pending: list[tuple[str, torch.Tensor]] = []
    ahead = max(args.decode_workers, 1) + 1
    try:
        for group in groups[:ahead]:
            queue.append((group, decode_pool.submit(fetch, group)))
        next_index = len(queue)

        while queue or pending:
            while queue and len(pending) < batch_size:
                group, future = queue.popleft()
                if next_index < len(groups):
                    queue.append(
                        (groups[next_index], decode_pool.submit(fetch, groups[next_index]))
                    )
                    next_index += 1
                try:
                    pending.extend(future.result())
                except Exception as exc:
                    failures.append(
                        (group[0].key, f"decode: {type(exc).__name__}: {exc}")
                    )
            if not pending:
                continue

            batch, pending = pending[:batch_size], pending[batch_size:]
            ids = [window_id for window_id, _ in batch]
            waves = torch.stack([wave for _, wave in batch]).to(device)
            try:
                streams = extract_batch(waves, mert, mixes, autocast)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                failures.append((",".join(ids), "OOM: lower --batch"))
                continue

            stats = similarity(streams["content"], streams["style"])
            for name in stat_sums:
                stat_sums[name] += stats[name] * len(ids)
            arrays = {
                branch: stream.to(torch.float16).cpu().numpy()
                for branch, stream in streams.items()
            }
            for i, window_id in enumerate(ids):
                for branch in BRANCHES:
                    array = arrays[branch][i]
                    written += array.nbytes
                    out_key = feature_key(
                        args.out_prefix, branch, args.split, window_id
                    )
                    uploads.append(
                        (out_key, upload_pool.submit(s3.put_array, out_key, array))
                    )
            drain_uploads(4 * max(args.upload_workers, 1))

            processed += len(ids)
            batches += 1
            if batches % 10 == 0 or processed >= total_windows:
                elapsed = time.time() - started
                rate = processed / max(elapsed, 1e-9)
                remaining = total_windows - processed
                print(
                    f"[{processed}/{total_windows}] windows  "
                    f"cos={stat_sums['cos'] / processed:.4f} "
                    f"centered={stat_sums['cos_centered'] / processed:.4f}  "
                    f"{rate * 3600:.0f} win/hr  "
                    f"eta {remaining / max(rate * 3600, 1e-9):.1f} h  "
                    f"{human_bytes(written)} out"
                    + (
                        f"  {human_bytes(source.bytes_in)} in "
                        f"({source.fetched} tracks)"
                        if args.s3_audio
                        else ""
                    ),
                    flush=True,
                )
    finally:
        drain_uploads(0)
        decode_pool.shutdown(wait=False, cancel_futures=True)
        upload_pool.shutdown(wait=True)

    manifest_key = f"{args.out_prefix.strip('/')}/manifests/{args.split}/window_pairs.jsonl"
    try:
        s3.put_file(manifest_key, manifest_path)
        print(f"uploaded manifest to s3://{args.bucket}/{manifest_key}")
    except Exception as exc:
        failures.append((manifest_key, f"manifest upload: {type(exc).__name__}: {exc}"))

    print(f"\n{'=' * 72}")
    print(
        f"{processed} windows, {human_bytes(written)} to "
        f"s3://{args.bucket}/{args.out_prefix.strip('/')}/"
        f"{{content,style}}/{args.split}/"
    )
    if source.fetched:
        print(
            f"fetched {source.fetched} source tracks from S3 "
            f"({human_bytes(source.bytes_in)})"
        )
    if processed:
        print(
            f"\ncontent vs. style mix similarity (mean over {processed} windows)\n"
            f"  frame cosine           {stat_sums['cos'] / processed:.4f}\n"
            f"  centered frame cosine  {stat_sums['cos_centered'] / processed:.4f}\n"
            "  -> centered cosine above ~0.99 means the two mixes are the same\n"
            "     stream; cache one and feed both encoders from it (timeline.md,\n"
            "     'cache one stream or two?')."
        )
    if failures:
        print(f"\n{len(failures)} failures:")
        for label, message in failures[:20]:
            print(f"  {label}: {message}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        sys.exit(1)


if __name__ == "__main__":
    main()
