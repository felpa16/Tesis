#!/usr/bin/env python3
"""Map the detection flow's fitted distribution in 2-D, with the windows it scores.

One figure, one pair of axes, four things on it:

    the flow's distribution   shaded regions holding 50 / 90 / 99 % of its mass
    train windows             human music the flow was fitted to (train_w100)
    test50 windows            human music from works it never saw
    MoM windows               n windows of each generator

so one picture answers three questions: does the flow cover the data it was
trained on, does it cover held-out human music as well, and where do the
generators land relative to it.

Works for both checkpoint kinds evaluate_mom.py scores:

  * train_mert_flow.py (no disentanglement): the space is the flow's input,
    the standardized 1024-d layer mix of time-pooled MERT.
  * train_flow.py (disentanglement): the space is the whitened, flattened
    content and style tokens side by side (4096-d), the inputs of p(content)
    and p(style | content). --space content / style keeps one half.

How the 2-D picture is made, and what it can and cannot show:

  * Projection. PCA, fitted by default on the train windows (--fit-on), so the
    axes are the two directions in which the human training music varies most.
    Every source and the flow's samples go through the same linear map.
  * The flow's distribution. A flow has no closed-form 2-D marginal, but it can
    be sampled. --samples draws from it (for the disentangled checkpoint
    ancestrally: content from p(content), then style from p(style | content),
    i.e. the alpha = 1 joint), the draws are projected with the same PCA, and
    a Gaussian KDE of the projections is the flow's marginal density along the
    two axes. The shaded regions are its highest-density regions: the smallest
    areas holding 50, 90 and 99 % of the flow's mass in this view.
  * What it cannot show. A window can sit inside the 50 % region and still be
    improbable under the flow, because it is unusual along a direction the
    projection drops. So the legend also gives each source's median per-window
    log-density in the full space (evaluate_mom.py's score) and, for each
    generator, the window-level AUROC against test50. The terminal table adds
    the share of each source's windows inside each region: train windows at
    ~50 / 90 / 99 % mean the flow's marginal in this view matches its data.

Windows are cut as evaluate_mom.py cuts them: the same deterministic song order,
window_starts and decode_window, with S taken from the checkpoint, so with the
same --seed the first test50 and MoM songs are the ones evaluate_mom.py scores.
Each source contributes its first --n windows, --windows-per-song from each
song, skipping songs that fail to load. For the MERT-only flow the train
windows are random rows of the pooled cache it was trained on (recorded in the
checkpoint; --train-cache overrides), so they are its literal training
windows. When that cache is absent, or for the disentangled flow, they are
decoded from the train split's audio like the other sources.

Outputs, in --out-dir (default: flow_map/ next to the flow checkpoint):

    embeddings.npz          every window's point in the flow's space, plus the flow's samples
    windows.jsonl           one line per window: source, song, start, log-density terms
    run.json                what was computed
    flow_map.png            the figure (--figure renames it)
    flow_map_summary.json   per-source medians, AUROCs and region coverage

Only computing needs the GPU. A rerun with the same sampling arguments reuses
embeddings.npz, so --space, --fit-on, --levels and --alpha can change without
recomputing; --plot-only never loads a model at all.

Examples:
    python scripts/plot_flow_map.py \\
        --flow-checkpoint checkpoints/mert-flow-unfrozen/mert_flow_last.pt \\
        --data-root $DATA --bucket $BUCKET
    python scripts/plot_flow_map.py \\
        --flow-checkpoint checkpoints/flow-mqa-w100/flow_last.pt \\
        --data-root $DATA --bucket $BUCKET --human-audio-key '{path}'
    # the style half only, axes fitted on the flow's own samples
    python scripts/plot_flow_map.py \\
        --flow-checkpoint checkpoints/flow-mqa-w100/flow_last.pt \\
        --plot-only --space style --fit-on samples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

import evaluate_mom as em  # noqa: E402
from shs100k_meta import DEFAULT_DATA_ROOT  # noqa: E402

# What changing the run would invalidate: embeddings.npz is reused only when
# these match.
RUN_IDENTITY = (
    "flow_checkpoint",
    "encoder_checkpoint",
    "train_split",
    "train_cache",
    "human_split",
    "generators",
    "n",
    "windows_per_song",
    "samples",
    "seed",
)

# Chart surface and ink, as in evaluate_mom.py and plot_mert_features.py.
SURFACE = em.SURFACE
INK = em.INK
INK_2 = em.INK_2
GRID = em.GRID

# A scatter puts every pair of sources side by side, so the palette has to
# separate all pairs, not just neighbours. Five hues of the project palette
# pass that test (worst normal-vision dE 15.6; worst CVD dE 6.9, which is
# legal only with a second cue) and six do not. So both human sources share
# blue and differ by fill (train filled, held-out hollow), and every source
# also has its own marker shape.
HUMAN_COLOR = em.HUMAN_COLOR
GENERATOR_STYLE = {
    "udio": ("#1baf7a", "^"),  # aqua
    "riffusion": ("#eda100", "s"),  # yellow
    "diffrhythm": ("#008300", "D"),  # green
    "yue": ("#e34948", "v"),  # red
}
EXTRA_STYLE = [("#eb6834", "P"), ("#e87ba4", "X"), ("#6250d6", "p")]
# Neutral ramp for the flow's regions, lightest = the largest mass: the regions
# are a backdrop, so they take no categorical hue.
REGION_FILLS = ("#f1f0ec", "#e3e2dc", "#d3d2ca", "#c3c2b7")


# --------------------------------------------------------------------------
# Embedding (GPU)
# --------------------------------------------------------------------------


class Embedder(em.Scorer):
    """evaluate_mom.py's Scorer that also returns each window's point in the
    flow's input space, and can draw samples from the flow."""

    def spaces(self) -> dict[str, list[int]]:
        """Column ranges of the named halves of a point (none for MERT-only)."""
        if self.kind == "mert_flow":
            return {}
        from src.models.flow import latent_features

        size = latent_features(self.encoder_config.bottleneck)
        return {"content": [0, size], "style": [size, 2 * size]}

    def embed(self, waves: list) -> tuple[np.ndarray, dict[str, list[float]]]:
        """Windows -> points (B, F) and the per-window log-probs (self.terms)."""
        torch = self.torch
        with torch.no_grad():
            batch = torch.stack(waves).to(self.device)
            if self.kind == "mert_flow":
                from src.models.mert_flow import pool_layers

                pooled = pool_layers(self.mert, batch, self.micro_batch, self.autocast)
                return self.embed_pooled(pooled)
            content, style = self.encode_windows(
                self.mert, self.encoder, batch, self.micro_batch, self.autocast
            )
            terms = self.detector.log_probs(content, style)
            terms["gaussian"] = self.baseline.log_prob(content, style)
            point = torch.cat(
                [
                    self.detector.content_flow.whitener(content.flatten(1)),
                    self.detector.style_flow.whitener(style.flatten(1)),
                ],
                dim=1,
            )
        return self._numpy(point, terms)

    def embed_pooled(self, pooled) -> tuple[np.ndarray, dict[str, list[float]]]:
        """MERT-only: time-pooled layers (B, 25, 1024) -> points and log-probs."""
        with self.torch.no_grad():
            pooled = pooled.float()
            point = self.mert_flow.features(pooled)
            terms = self.mert_flow.log_probs(pooled)
        return self._numpy(point, terms)

    def _numpy(self, point, terms) -> tuple[np.ndarray, dict[str, list[float]]]:
        return (
            point.float().cpu().numpy(),
            {t: terms[t].float().cpu().tolist() for t in self.terms},
        )

    def sample(self, count: int, batch: int, seed: int) -> np.ndarray:
        """`count` draws from the flow, as points in the same space as embed()'s.

        Disentangled: ancestral, content from p(content), then style from
        p(style | content) given that content, so the draws follow the joint
        the alpha = 1 score evaluates.
        """
        torch = self.torch
        torch.manual_seed(seed)
        draws = []
        with torch.no_grad():
            for start in range(0, count, batch):
                m = min(batch, count - start)
                if self.kind == "mert_flow":
                    point = self.mert_flow.flow().sample((m,))
                else:
                    content_flow = self.detector.content_flow
                    style_flow = self.detector.style_flow
                    bottleneck = self.encoder_config.bottleneck
                    z_content = content_flow.flow().sample((m,))
                    whitener = content_flow.whitener
                    content = (z_content * whitener.std + whitener.mean).reshape(
                        m, bottleneck.n_tokens, bottleneck.token_dim
                    )
                    z_style = style_flow.flow(style_flow.pooler(content)).sample()
                    point = torch.cat([z_content, z_style], dim=1)
                draws.append(point.float().cpu().numpy())
        return np.concatenate(draws)


def human_candidates(
    data_root: Path, split: str, window_seconds: float, seed: int,
    opener: em.AudioOpener, streamed: bool,
) -> list[em.Song]:
    """Every usable track of a split, in evaluate_mom.py's deterministic order.

    Without --human-audio-key only tracks on disk can be read, so the order is
    kept but the rest are dropped up front instead of failing one by one.
    """
    songs = em.deterministic_sample(em.human_songs(data_root, split, window_seconds), 0, seed)
    if streamed:
        return songs
    local = [s for s in songs if opener.local_path(s).exists()]
    if len(local) < len(songs):
        print(
            f"{split}: {len(local)} of {len(songs)} tracks are on disk under {data_root}; "
            "sampling from those (--human-audio-key streams the rest)"
        )
    return local


def embed_source(
    embedder: Embedder, label: str, songs: list[em.Song], load, n: int,
    batch: int, workers: int,
) -> tuple[list[np.ndarray], list[dict], list[str]]:
    """The first n windows of `songs`, in order, skipping songs that fail."""
    points: list[np.ndarray] = []
    rows: list[dict] = []
    failures: list[str] = []
    pending: list[tuple[dict, object]] = []

    def flush(items: list[tuple[dict, object]]) -> None:
        point, terms = embedder.embed([wave for _, wave in items])
        points.append(point)
        for i, (row, _) in enumerate(items):
            rows.append({**row, **{t: terms[t][i] for t in embedder.terms}})

    collected = 0
    for loaded in em.prefetch(load, songs, workers):
        if loaded.error is not None:
            failures.append(f"{label}/{loaded.song.song_id}: {loaded.error}")
            continue
        for start, wave in list(zip(loaded.starts, loaded.waves))[: n - collected]:
            pending.append(({"source": label, "song": loaded.song.song_id, "start": start}, wave))
            collected += 1
        while len(pending) >= batch:
            flush(pending[:batch])
            pending = pending[batch:]
        if collected >= n:
            break
    if pending:
        flush(pending)
    print(f"{label:<12} {len(rows):>5} windows  ({len(failures)} songs failed)", flush=True)
    return points, rows, failures


def train_from_cache(
    embedder: Embedder, label: str, root: Path, n: int, seed: int, batch: int
) -> tuple[list[np.ndarray], list[dict]]:
    """MERT-only: n random rows of the pooled cache the flow was trained on."""
    import torch

    from src.data.pooled_cache import PooledCache

    cache = PooledCache(root)
    index = np.sort(np.random.default_rng(seed).choice(len(cache), min(n, len(cache)), replace=False))
    meta = cache.rows()
    points: list[np.ndarray] = []
    rows: list[dict] = []
    for start in range(0, len(index), batch):
        chunk = index[start : start + batch]
        pooled = torch.from_numpy(cache.gather(chunk)).to(embedder.device)
        point, terms = embedder.embed_pooled(pooled)
        points.append(point)
        for i, row in enumerate(chunk):
            rows.append({
                "source": label,
                "song": meta[row]["key"],
                "start": meta[row]["start"],
                **{t: terms[t][i] for t in embedder.terms},
            })
    print(f"{label:<12} {len(rows):>5} windows  (rows of the training cache {root})")
    return points, rows


def recorded_train_cache(flow_path: Path) -> str | None:
    import torch

    # mmap: only the one string is needed, not the weights
    checkpoint = torch.load(flow_path, map_location="cpu", weights_only=False, mmap=True)
    return checkpoint.get("train_cache")


def identity(args: argparse.Namespace) -> dict:
    return {
        "flow_checkpoint": str(args.flow_checkpoint),
        "encoder_checkpoint": str(args.encoder_checkpoint) if args.encoder_checkpoint else None,
        "train_split": args.train_split,
        "train_cache": str(args.train_cache) if args.train_cache else None,
        "human_split": args.human_split,
        "generators": list(args.generators),
        "n": args.n,
        "windows_per_song": args.windows_per_song,
        "samples": args.samples,
        "seed": args.seed,
    }


def computed(args: argparse.Namespace) -> bool:
    """True when embeddings.npz exists and was computed with these arguments."""
    run_path = args.out_dir / "run.json"
    if not (run_path.exists() and (args.out_dir / "embeddings.npz").exists()):
        return False
    previous = json.loads(run_path.read_text(encoding="utf-8"))
    return all(previous.get(k) == v for k, v in identity(args).items())


def compute(args: argparse.Namespace) -> None:
    from src.data.windows import WindowConfig
    from src.s3 import S3
    from track_mom import song_loader

    embedder = Embedder(args.flow_checkpoint, args.encoder_checkpoint, args.device)
    window_seconds = embedder.window_seconds
    train_split = args.train_split or embedder.flow_config.data.train_split
    print(
        f"device={embedder.device.type}  kind={embedder.kind}  flow={args.flow_checkpoint} "
        f"(epoch {embedder.epoch}, step {embedder.step})  encoder={embedder.encoder_path}"
    )
    print(f"{args.n} windows of {window_seconds:g} s per source, "
          f"{args.windows_per_song} per song; {args.samples} flow samples")
    if train_split == args.human_split:
        raise SystemExit(f"--human-split {args.human_split} is the flow's training split")
    seen = {embedder.flow_config.data.train_split, embedder.flow_config.data.val_split}
    if embedder.kind != "mert_flow":
        seen |= {embedder.encoder_config.data.train_split, embedder.encoder_config.data.val_split}
    if args.human_split in seen:
        print(f"WARNING: {args.human_split!r} was used to train or validate this model; "
              "the held-out row is then not held out.")

    train_cache = None
    if embedder.kind == "mert_flow":
        recorded = args.train_cache or recorded_train_cache(args.flow_checkpoint)
        if recorded and (Path(recorded) / "meta.json").exists():
            train_cache = Path(recorded)
        else:
            print(f"training cache {recorded} not found; decoding {train_split} audio instead")
    elif args.train_cache:
        print("--train-cache ignored: the disentangled flow's encoder needs audio, not pooled MERT")

    needs_s3 = args.mom_root is None or args.human_audio_key
    s3 = S3(args.bucket, args.region) if needs_s3 else None
    opener = em.AudioOpener(
        args.data_root, s3, args.human_audio_key, args.mom_root is not None, args.tmp_dir
    )
    window = WindowConfig(window_seconds=window_seconds, sample_rate=embedder.sample_rate)
    load = song_loader(opener, window, args.windows_per_song)
    batch = args.batch or embedder.flow_config.data.batch_tracks

    def human(split: str) -> list[em.Song]:
        return human_candidates(
            args.data_root, split, window_seconds, args.seed, opener, bool(args.human_audio_key)
        )

    sources: list[tuple[str, list[em.Song]]] = []
    if train_cache is None:
        sources.append((train_split, human(train_split)))
    sources.append((args.human_split, human(args.human_split)))
    for generator in args.generators:
        if args.mom_root is not None:
            _, songs = em.mom_songs_local(args.mom_root, generator)
        else:
            _, songs = em.mom_songs_s3(s3, args.mom_prefix, generator)
        sources.append((generator, em.deterministic_sample(songs, 0, args.seed)))

    points: list[np.ndarray] = []
    rows: list[dict] = []
    failures: list[str] = []
    if train_cache is not None:
        points, rows = train_from_cache(
            embedder, train_split, train_cache, args.n, args.seed, args.sample_batch
        )
    for label, songs in sources:
        p, r, f = embed_source(embedder, label, songs, load, args.n, batch, args.fetch_workers)
        points, rows, failures = points + p, rows + r, failures + f
    for line in failures[:10]:
        print(f"  failed: {line}")

    samples = embedder.sample(args.samples, args.sample_batch, args.seed)
    print(f"flow samples: {samples.shape[0]} x {samples.shape[1]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "embeddings.npz",
        points=np.concatenate(points).astype(np.float32),
        samples=samples.astype(np.float32),
    )
    with open(args.out_dir / "windows.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    run = {
        **identity(args),
        "kind": embedder.kind,
        "encoder_used": str(embedder.encoder_path),
        "train_label": train_split,
        "train_from": f"cache {train_cache}" if train_cache else "audio",
        "order": [train_split, args.human_split, *args.generators],
        "window_seconds": window_seconds,
        "spaces": embedder.spaces(),
        "alpha": float(embedder.flow_config.flow.alpha),
        "failed_songs": len(failures),
    }
    (args.out_dir / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Projection and the flow's 2-D density (CPU)
# --------------------------------------------------------------------------


def pca_fit(x: np.ndarray, max_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top two principal axes -> mean (F,), components (2, F), variance ratios (2,).

    Exact SVD of the centred rows (at most max_rows of them); signs fixed so
    each axis's largest loading is positive, so reruns draw the same way up.
    """
    if len(x) > max_rows:
        x = x[np.random.default_rng(seed).choice(len(x), max_rows, replace=False)]
    mean = x.mean(axis=0)
    centered = x - mean
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    components = right[:2]
    signs = np.sign(components[np.arange(2), np.abs(components).argmax(axis=1)])
    ratios = singular[:2] ** 2 / (centered**2).sum()
    return mean, components * signs[:, None], ratios


def bandwidth(points: np.ndarray) -> np.ndarray:
    """Silverman's rule per axis (in 2-D the factor is n^(-1/6)).

    The IQR term keeps a handful of extreme flow draws from widening the kernel.
    """
    std = points.std(axis=0)
    iqr = np.subtract(*np.percentile(points, [75, 25], axis=0)) / 1.349
    sigma = np.where(iqr > 0, np.minimum(std, iqr), std)
    return sigma * len(points) ** (-1 / 6)


def kde(points: np.ndarray, query: np.ndarray, h: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Gaussian KDE of 2-D `points` (diagonal bandwidth h), evaluated at `query`."""
    norm = 1.0 / (2 * np.pi * h.prod() * len(points))
    out = np.empty(len(query))
    for start in range(0, len(query), chunk):
        d = (query[start : start + chunk, None, :] - points[None, :, :]) / h
        out[start : start + chunk] = np.exp(-0.5 * (d**2).sum(axis=-1)).sum(axis=1) * norm
    return out


def hdr_thresholds(density_at_samples: np.ndarray, masses: list[float]) -> list[float]:
    """Density level whose superlevel set holds each mass (Hyndman 1996): the
    (1 - mass) quantile of the density at the flow's own draws."""
    return [float(np.quantile(density_at_samples, 1.0 - m)) for m in masses]


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def source_style(label: str, run: dict) -> tuple[str, str, bool]:
    """(color, marker, filled) of a source; fixed per entity, never by rank."""
    if label == run["train_label"]:
        return HUMAN_COLOR, "o", True
    if label == run["human_split"]:
        return HUMAN_COLOR, "o", False
    if label in GENERATOR_STYLE:
        return (*GENERATOR_STYLE[label], True)
    extra = [g for g in run["generators"] if g not in GENERATOR_STYLE]
    return (*EXTRA_STYLE[extra.index(label) % len(EXTRA_STYLE)], True)


def load_outputs(out_dir: Path) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    run_path, npz_path = out_dir / "run.json", out_dir / "embeddings.npz"
    if not (run_path.exists() and npz_path.exists()):
        raise SystemExit(f"nothing computed yet in {out_dir}; run without --plot-only first")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    rows = em.read_records(out_dir / "windows.jsonl")
    with np.load(npz_path) as data:
        points, samples = data["points"], data["samples"]
    if len(rows) != len(points):
        raise SystemExit(f"{out_dir}: {len(points)} points but {len(rows)} window lines")
    return run, rows, points, samples


def select_space(x: np.ndarray, run: dict, space: str) -> np.ndarray:
    if space == "all":
        return x
    if space not in run["spaces"]:
        raise SystemExit(f"--space {space} needs a disentangled checkpoint; this one is {run['kind']}")
    start, stop = run["spaces"][space]
    return x[:, start:stop]


def report(args: argparse.Namespace) -> None:
    run, rows, points, samples = load_outputs(args.out_dir)
    kind = run["kind"]
    order = [s for s in run["order"] if any(r["source"] == s for r in rows)]
    index = {s: np.array([i for i, r in enumerate(rows) if r["source"] == s]) for s in order}
    reference = run["human_split"]

    points = select_space(points, run, args.space).astype(np.float64)
    samples = select_space(samples, run, args.space).astype(np.float64)
    finite = np.isfinite(samples).all(axis=1)
    if not finite.all():
        print(f"dropping {int((~finite).sum())} non-finite flow samples of {len(samples)}")
    samples = samples[finite]

    fit_rows = {
        "train": points[index[run["train_label"]]] if run["train_label"] in index else points,
        "samples": samples,
        "all": points,
    }[args.fit_on]
    mean, components, ratios = pca_fit(fit_rows, args.pca_rows, args.seed)
    coords = (points - mean) @ components.T
    sample_coords = (samples - mean) @ components.T

    masses = sorted(args.levels, reverse=True)  # largest mass = lowest density level
    h = bandwidth(sample_coords)
    thresholds = hdr_thresholds(kde(sample_coords, sample_coords, h), masses)
    density_at_points = kde(sample_coords, coords, h)

    tail = (1.0 - args.frame_quantile) / 2
    pooled = np.vstack([coords, sample_coords])
    lo, hi = np.quantile(pooled, [tail, 1.0 - tail], axis=0)
    pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], args.grid), np.linspace(lo[1], hi[1], args.grid))
    grid = kde(sample_coords, np.column_stack([gx.ravel(), gy.ravel()]), h).reshape(gx.shape)

    alpha = args.alpha if args.alpha is not None else run.get("alpha", 1.0)
    score_label, score = em.views(alpha, kind)["score"]
    scores = {s: np.array([score(rows[i]) for i in index[s]]) for s in order}
    stats: dict[str, dict] = {}
    for s in order:
        inside = density_at_points[index[s]][:, None] >= np.array(thresholds)[None, :]
        pinned = ((coords[index[s]] < lo) | (coords[index[s]] > hi)).any(axis=1)
        entry = {
            "windows": int(len(index[s])),
            "median_score": em.finite_quantiles(scores[s], [0.5])[0],
            "inside": {f"{m:g}": float(inside[:, k].mean()) for k, m in enumerate(masses)},
            "beyond_frame": int(pinned.sum()),
        }
        if s not in (reference, run["train_label"]) and reference in scores:
            entry["auroc_vs_" + reference] = em.auroc(scores[reference], scores[s])
        stats[s] = entry

    space_name = {"all": "content + style" if run["spaces"] else "MERT mix"}.get(args.space, args.space)
    print_table(stats, order, masses, reference, score_label)
    print(f"\nspace: {space_name} ({points.shape[1]}-d)   PCA fitted on {args.fit_on}: "
          f"PC1 {ratios[0]:.1%}, PC2 {ratios[1]:.1%} of its variance")

    figure = args.out_dir / args.figure
    plot(
        coords, grid, (gx, gy), thresholds, masses, (lo, hi), order, index, stats, run,
        ratios, space_name, score_label, len(samples), args, figure,
    )
    summary = {
        "run": run,
        "space": args.space,
        "fit_on": args.fit_on,
        "pca_variance_ratio": [float(r) for r in ratios],
        "score": score_label,
        "alpha": alpha,
        "regions": {f"{m:g}": t for m, t in zip(masses, thresholds)},
        "flow_samples": int(len(samples)),
        "sources": stats,
    }
    summary_path = figure.with_name(f"{figure.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"\nfigure:  {figure}\nsummary: {summary_path}")


def print_table(
    stats: dict, order: list[str], masses: list[float], reference: str, score_label: str
) -> None:
    print(f"\nper-window score = {score_label} (nats), full space")
    header = (
        f"{'source':<14}{'windows':>8}{'median':>11}{'AUROC':>8}"
        + "".join(f"{'in ' + format(m, '.0%'):>9}" for m in masses)
        + f"{'beyond':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in order:
        e = stats[s]
        auroc = e.get(f"auroc_vs_{reference}")
        line = f"{s:<14}{e['windows']:>8}{e['median_score']:>11.1f}"
        line += f"{auroc:>8.3f}" if auroc is not None else f"{'':>8}"
        line += "".join(f"{e['inside'][f'{m:g}']:>9.0%}" for m in masses)
        print(line + f"{e['beyond_frame']:>8}")
    print(
        f"\nAUROC = P(a {reference} window scores higher than this source's window).\n"
        "in X% = share of windows inside the flow's X% region in this 2-D view; a flow "
        "that fits its training data in this view puts its train row near X%.\n"
        "beyond = windows outside the frame, drawn pinned to its edge."
    )


def plot(
    coords: np.ndarray,
    grid: np.ndarray,
    mesh: tuple[np.ndarray, np.ndarray],
    thresholds: list[float],
    masses: list[float],
    frame: tuple[np.ndarray, np.ndarray],
    order: list[str],
    index: dict[str, np.ndarray],
    stats: dict[str, dict],
    run: dict,
    ratios: np.ndarray,
    space_name: str,
    score_label: str,
    n_samples: int,
    args: argparse.Namespace,
    out: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: this runs on an EC2 box with no display
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    lo, hi = frame
    fig = plt.figure(figsize=(13.0, 8.6), facecolor=SURFACE)
    ax = fig.add_axes((0.065, 0.085, 0.6, 0.76))
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])

    # the flow's regions, largest mass (lowest density level) lightest
    levels = list(thresholds)  # ascending, since masses are descending
    top = max(float(grid.max()), levels[-1]) * (1 + 1e-6) + 1e-300
    fills = REGION_FILLS[: len(levels)]
    ax.contourf(*mesh, grid, levels=[*levels, top], colors=fills, zorder=1)
    lines = ax.contour(*mesh, grid, levels=levels, colors=INK_2, linewidths=0.7, zorder=2)
    ax.clabel(lines, fmt={t: f"{m:.0%}" for t, m in zip(levels, masses)}, fontsize=7.5, colors=INK_2)

    fit_name = {"train": "train-window", "samples": "flow-sample", "all": "all-window"}[args.fit_on]
    handles = []
    for z, s in enumerate(order):
        color, marker, filled = source_style(s, run)
        shown = np.clip(coords[index[s]], lo, hi)  # beyond the frame: pinned to its edge
        common = dict(marker=marker, zorder=3 + z, clip_on=False)
        if filled:
            ax.scatter(shown[:, 0], shown[:, 1], s=18, c=color, alpha=0.8,
                       edgecolors=SURFACE, linewidths=0.4, **common)
        else:
            ax.scatter(shown[:, 0], shown[:, 1], s=22, facecolors="none",
                       edgecolors=color, linewidths=1.0, alpha=0.9, **common)
        e = stats[s]
        role = {run["train_label"]: " (trained on)", run["human_split"]: " (held out)"}.get(s, "")
        detail = f"{e['windows']} windows · median log p {e['median_score']:.1f}"
        auroc = e.get(f"auroc_vs_{run['human_split']}")
        if auroc is not None:
            detail += f"\nAUROC vs {run['human_split']} {auroc:.3f}"
        if e["beyond_frame"]:
            detail += f" · {e['beyond_frame']} at edge"
        handles.append(Line2D(
            [], [], linestyle="none", marker=marker, markersize=6.5,
            markerfacecolor=color if filled else "none",
            markeredgecolor=SURFACE if filled else color,
            markeredgewidth=0.5 if filled else 1.1,
            label=f"{s}{role}\n{detail}",
        ))
    for fill, m in zip(fills, masses):
        handles.append(Patch(facecolor=fill, edgecolor=INK_2, linewidth=0.6,
                             label=f"flow: smallest region holding {m:.0%} of its mass"))
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.69, 0.85), frameon=False,
        fontsize=8.5, labelcolor=INK_2, labelspacing=1.1, handletextpad=0.8, borderaxespad=0,
    )

    ax.set_xlabel(f"PC1 · {ratios[0]:.1%} of {fit_name} variance", color=INK_2, fontsize=9)
    ax.set_ylabel(f"PC2 · {ratios[1]:.1%} of {fit_name} variance", color=INK_2, fontsize=9)
    flow_name = Path(run["flow_checkpoint"]).name
    fig.text(0.065, 0.955, "Where the flow puts its mass, and where each source's windows land",
             fontsize=14, color=INK, fontweight="bold", ha="left")
    fig.text(
        0.065, 0.91,
        f"{flow_name} · {space_name} · PCA fitted on {args.fit_on} · one mark per "
        f"{run['window_seconds']:g} s window, {run['windows_per_song']} per song\n"
        f"regions: KDE of {n_samples} flow samples in this projection · legend scores: "
        f"{score_label}, full space, per window",
        fontsize=9, color=INK_2, ha="left", va="top", linespacing=1.4,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--flow-checkpoint", type=Path, required=True,
                        help="flow_last.pt (train_flow.py) or mert_flow_last.pt (train_mert_flow.py)")
    parser.add_argument("--encoder-checkpoint", type=Path,
                        help="disentangled only; default: the one recorded in the flow checkpoint")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="holds manifests/{split}/ and the human audio")
    parser.add_argument("--train-split", default=None,
                        help="split the train windows come from (default: the flow's training split)")
    parser.add_argument("--train-cache", type=Path, default=None,
                        help="MERT-only: pooled cache of the train windows (default: the one "
                        "recorded in the checkpoint)")
    parser.add_argument("--human-split", default="test50", help="held-out human split")
    parser.add_argument("--human-audio-key", default=None,
                        help="S3 key template for human tracks missing from --data-root "
                        "(see evaluate_mom.py), e.g. '{path}'")
    parser.add_argument("--bucket", default="songs-and-mert-features")
    parser.add_argument("--region", default=None)
    parser.add_argument("--mom-prefix", default="mom")
    parser.add_argument("--mom-root", type=Path, default=None,
                        help="read MoM from this local directory instead of S3")
    parser.add_argument("--generators", nargs="+", default=list(em.GENERATORS))
    parser.add_argument("--n", type=int, default=300, help="windows per source")
    parser.add_argument("--windows-per-song", type=int, default=1,
                        help="windows cut from each song (1 = n distinct songs per source)")
    parser.add_argument("--samples", type=int, default=5000, help="draws from the flow")
    parser.add_argument("--seed", type=int, default=0,
                        help="song order (evaluate_mom.py's), cache rows and flow samples")
    parser.add_argument("--batch", type=int, default=0,
                        help="windows per encoder call (0 = the flow's batch_tracks)")
    parser.add_argument("--sample-batch", type=int, default=500,
                        help="flow draws, or cached train rows, per call")
    parser.add_argument("--fetch-workers", type=int, default=8)
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: flow_map/ next to the flow checkpoint")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute even when embeddings.npz matches these arguments")
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw from embeddings.npz without loading any model")
    view = parser.add_argument_group("view (free to change with --plot-only)")
    view.add_argument("--space", choices=("all", "content", "style"), default="all",
                      help="disentangled only: both token sets, or one of them")
    view.add_argument("--fit-on", choices=("train", "samples", "all"), default="train",
                      help="what the PCA axes are fitted on: the train windows, the "
                      "flow's samples, or every plotted window")
    view.add_argument("--pca-rows", type=int, default=2000, help="rows the PCA is fitted on, at most")
    view.add_argument("--levels", type=float, nargs="+", default=[0.5, 0.9, 0.99],
                      help="masses of the flow's shaded regions (at most 4)")
    view.add_argument("--alpha", type=float, default=None,
                      help="disentangled: weight of log p(content) in the legend's score "
                      "(default: the flow config's; the regions are always the joint)")
    view.add_argument("--frame-quantile", type=float, default=0.995,
                      help="per-axis share of windows and samples inside the frame")
    view.add_argument("--grid", type=int, default=180, help="density grid points per axis")
    view.add_argument("--figure", default="flow_map.png")
    view.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = args.flow_checkpoint.resolve().parent / "flow_map"
    if args.n < 1 or args.windows_per_song < 1 or args.samples < 2:
        parser.error("--n, --windows-per-song must be at least 1 and --samples at least 2")
    if not 1 <= len(args.levels) <= len(REGION_FILLS) or not all(0 < m < 1 for m in args.levels):
        parser.error(f"--levels takes 1 to {len(REGION_FILLS)} masses strictly between 0 and 1")
    if len(set(args.levels)) != len(args.levels):
        parser.error("--levels must be distinct")
    if not 0 < args.frame_quantile <= 1:
        parser.error("--frame-quantile must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    if not args.plot_only:
        if computed(args) and not args.overwrite:
            print(f"reusing {args.out_dir / 'embeddings.npz'} (same arguments; --overwrite recomputes)")
        else:
            compute(args)
    report(args)


if __name__ == "__main__":
    main()
