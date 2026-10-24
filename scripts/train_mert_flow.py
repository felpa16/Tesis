#!/usr/bin/env python3
"""MERT-only baseline: does content/style disentanglement add anything?

Removes the whole representation-learning stage. Time-pooled MERT hidden
states, mixed by ONE softmax layer-weight vector (25 weights), are
standardized and modelled by one unconditional neural spline flow, log p(x) of
human music, with a closed-form full-covariance Gaussian on the same features
as baseline. See src/models/mert_flow.py.

One mix, not two: the content/style pair of mixes exists only because the
main pipeline has two branches with different objectives. Here the flow
models p(human music) from a single embedding of the audio, so a second mix
would be part of the disentanglement under test. --mix picks which of the
checkpoint's two vectors is used (frozen) or trained from (unfrozen).

Exactly one arm must be chosen; there is no default:

  --frozen-25     the 25 weights stay at the --mix vector of --mix-checkpoint
                  (the mqa-w100 run), applied to the raw hidden states exactly as
                  the encoders consumed them. Only the encoders differ from the
                  main pipeline.
  --unfrozen-25   the 25 weights train jointly with the flow, starting from
                  the --mix vector of --mix-checkpoint (or --init uniform). Because maximum
                  likelihood rewards low-entropy representations, the arm runs
                  constrained (all on by default, each one ablatable):
                    * fixed training-set per-layer normalization (--layer-norm)
                    * per-dimension standardization by the mix's exact std
                    * log-det redundancy penalty (--redundancy-weight, 1.0
                      cancels the Gaussian-level reward for collinear or
                      duplicated features)
                    * KL(w || uniform) per stream (--entropy-weight)
                  and is watched: weights, per-dimension means and variances,
                  norms, covariance eigenvalues and effective rank are logged
                  every --diag-every steps (TensorBoard + diagnostics.jsonl),
                  with a warning (or a stop, --stop-on-collapse) when the joint
                  effective rank falls below --collapse-ratio of its start.

Input is the pooled-MERT cache from scripts/cache_pooled_mert.py. The first run
fits the training-set moments of all 25 layers jointly and stores them as
{train-cache}/layer_stats.pt (2.6 GB, reused by both arms).

The loss is the flow's NLL in nats per dimension (plus penalties), so penalty
weights read on the same scale. Score MoM afterwards with
scripts/evaluate_mom.py --flow-checkpoint {checkpoint-dir}/mert_flow_last.pt,
or with scripts/evaluate_mert_flow.py for the codec controls.

Codec. A train cache built with cache_pooled_mert.py --codec CHAIN trains the
flow on human audio re-encoded through CHAIN; the val cache must use the same
chain. The chain is recorded in the checkpoint, the default checkpoint dir gets
it as a suffix, --track-metrics scores its evaluation songs through it, and
scripts/evaluate_mert_flow.py scores test50 through it. Human training audio
and both classes at evaluation then share the last codec (experiments.md
section 15).

Overfitting watch: before training and after every epoch, the flow's NLL is
measured in eval mode on the val cache and on a fixed random subset of the
train cache of the same size (--train-eval-windows), so the two numbers are
directly comparable. Each epoch appends one line to
{checkpoint-dir}/history.jsonl, and the run ends by drawing
{checkpoint-dir}/loss_curves.png (scripts/plot_training_curves.py).

--track-metrics also scores human val50 songs against the MoM generators each
epoch, sampled and windowed as evaluate_mom.py does, and draws
metric_curves.png. The MERT pass over those songs runs once, before training,
into --metrics-cache (shared by every run with the same settings); scoring an
epoch then takes seconds (scripts/track_mom.py). These AUROCs are for watching
how detection evolves; choosing an epoch by them would select on the
generators the detector is meant never to see.

Examples:
    python scripts/train_mert_flow.py --frozen-25 \\
        --mix-checkpoint checkpoints/mqa-w100/best.pt \\
        --train-cache cache/pooled/train --val-cache cache/pooled/val50
    python scripts/train_mert_flow.py --unfrozen-25 \\
        --mix-checkpoint checkpoints/mqa-w100/best.pt \\
        --train-cache cache/pooled/train --val-cache cache/pooled/val50
    # with per-epoch MoM AUROCs
    python scripts/train_mert_flow.py --frozen-25 \\
        --mix-checkpoint checkpoints/mqa-w100/best.pt \\
        --train-cache cache/pooled/train --val-cache cache/pooled/val50 \\
        --track-metrics --data-root $DATA --bucket $BUCKET
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from plot_training_curves import plot_run  # noqa: E402
from src.config import TrainConfig, config_from_dict, load_config  # noqa: E402
from src.data.codec import chain_slug, parse_chain  # noqa: E402
from src.data.pooled_cache import PooledCache  # noqa: E402
from src.models.mert_flow import (  # noqa: E402
    STREAMS,
    LayerCovariance,
    MertFlow,
    correlation,
    diagnostics,
    kl_to_uniform,
    load_mix_logits,
    redundancy_penalty,
)
from src.training import make_scheduler, pick_device  # noqa: E402


# --------------------------------------------------------------------------
# Training-set moments
# --------------------------------------------------------------------------


@torch.no_grad()
def fit_layer_stats(cache: PooledCache, device: torch.device, max_rows: int) -> dict:
    """Mean (L, D) and covariance (L, D, L, D) of the pooled layers, two passes.

    Per-shard products are fp32 (on centred data) and summed in fp64.
    """
    n_layers, dim = cache.n_layers, cache.dim
    total = torch.zeros(n_layers * dim, dtype=torch.float64, device=device)
    n = 0
    for array in cache.shards(max_rows):
        total += torch.from_numpy(array).to(device).reshape(len(array), -1).sum(0, dtype=torch.float64)
        n += len(array)
    mean = (total / n).float()
    acc = torch.zeros(n_layers * dim, n_layers * dim, dtype=torch.float64, device=device)
    for array in cache.shards(max_rows):
        x = torch.from_numpy(array).to(device).reshape(len(array), -1) - mean
        acc += (x.T @ x).double()
        del x
    cov = (acc / max(n - 1, 1)).float()
    del acc
    return {
        "mean": mean.reshape(n_layers, dim).cpu(),
        "cov": cov.reshape(n_layers, dim, n_layers, dim).cpu(),
        "n": n,
        "max_rows": max_rows,
    }


def layer_stats(cache: PooledCache, device: torch.device, max_rows: int, refit: bool) -> dict:
    path = cache.root / "layer_stats.pt"
    if path.exists() and not refit:
        stats = torch.load(path, map_location="cpu", weights_only=False)
        if stats.get("max_rows") == max_rows and stats.get("n_cache") == len(cache):
            print(f"layer moments: loaded {path} ({stats['n']} windows)")
            return stats
        print(f"layer moments: {path} was fitted on a different set of windows; refitting")
    started = time.time()
    stats = fit_layer_stats(cache, device, max_rows)
    stats["n_cache"] = len(cache)
    torch.save(stats, path)
    print(f"layer moments: fitted on {stats['n']} windows in {time.time() - started:.0f}s -> {path}")
    return stats


# --------------------------------------------------------------------------
# Loading (shared with evaluate_mom.py)
# --------------------------------------------------------------------------


def build_model(checkpoint: dict) -> tuple[MertFlow, TrainConfig]:
    config = config_from_dict(checkpoint["config"])
    model = MertFlow(
        checkpoint["n_layers"], checkpoint["dim"], tuple(checkpoint["streams"]),
        config.flow.style, checkpoint["layer_norm"], trainable=False,
    )
    model.load_state_dict(checkpoint["model"])
    return model, config


def load_mert_flow(path: Path, device: torch.device) -> tuple[MertFlow, TrainConfig, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("kind") != "mert_flow":
        raise SystemExit(f"{path} is not a train_mert_flow.py checkpoint")
    model, config = build_model(checkpoint)
    model.requires_grad_(False)
    model.eval()
    return model.to(device), config, checkpoint


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


class Monitor:
    """Diagnostics to stdout, TensorBoard and diagnostics.jsonl; collapse watch."""

    def __init__(self, writer: SummaryWriter, path: Path, collapse_ratio: float) -> None:
        self.writer = writer
        self.path = path
        self.collapse_ratio = collapse_ratio
        self.initial_rank: float | None = None

    def __call__(self, report: dict, step: int, streams: tuple[str, ...]) -> bool:
        """Log one diagnostics report; True when the representation collapsed."""
        scalars = report["scalars"]
        for name, value in scalars.items():
            self.writer.add_scalar(f"diag/{name}", value, step)
        for name, values in report["spectra"].items():
            self.writer.add_histogram(f"spectrum/{name}", np.log10(np.clip(values, 1e-12, None)), step)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, **report}) + "\n")

        rank = scalars["train/joint/effective_rank"]
        if self.initial_rank is None:
            self.initial_rank = rank
        parts = [f"eff_rank {rank:.0f}/{self.initial_rank:.0f}"]
        for s in streams:
            parts.append(
                f"{s}: argmax L{scalars[f'weights/{s}/argmax']} "
                f"max/min {scalars[f'weights/{s}/max_min_ratio']:.2f} "
                f"cos_init {scalars[f'weights/{s}/cos_init']:.3f}"
            )
        if "train/cross_stream_corr_mean" in scalars:
            parts.append(f"xcorr {scalars['train/cross_stream_corr_mean']:.3f}")
        if "probe/z_var_median" in scalars:
            parts.append(f"probe z var {scalars['probe/z_var_median']:.2f}")
        print(f"  diag step {step}: " + "  ".join(parts), flush=True)

        collapsed = rank < self.collapse_ratio * self.initial_rank
        if collapsed:
            print(
                f"  WARNING: joint effective rank {rank:.1f} fell below "
                f"{self.collapse_ratio:.0%} of its initial {self.initial_rank:.1f}"
            )
        return collapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    arm = parser.add_mutually_exclusive_group(required=True)
    arm.add_argument("--frozen-25", action="store_true", help="layer weights fixed")
    arm.add_argument("--unfrozen-25", action="store_true", help="layer weights trained, constrained")
    parser.add_argument("--mix-checkpoint", type=Path, required=True,
                        help="phase-1 checkpoint (e.g. checkpoints/mqa-w100/best.pt) or "
                        "layer-weights file holding the content and style logits")
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, help="held-out cache: val NLL and probe diagnostics")
    parser.add_argument("--mix", choices=STREAMS, default="content",
                        help="which of the checkpoint's layer-weight vectors to use "
                        "(frozen) or start from (unfrozen); default: content")
    parser.add_argument("--config", type=Path, help="JSON config; the flow is config.flow.style")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, help="flow learning rate (default: config optim.lr)")
    parser.add_argument("--mix-lr", type=float, default=1e-3, help="layer-logit learning rate")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--diag-every", type=int, default=250, help="steps between diagnostics")
    parser.add_argument("--probe-windows", type=int, default=4096,
                        help="held-out windows for empirical diagnostics (should exceed the feature count)")
    parser.add_argument("--val-max-windows", type=int, default=0, help="0 = whole val cache")
    parser.add_argument("--train-eval-windows", type=int, default=0,
                        help="fixed train windows scored each epoch for the train/val "
                        "comparison (0 = as many as val, or 20000 without val)")
    parser.add_argument("--stats-windows", type=int, default=0,
                        help="train windows used for the layer moments (0 = all)")
    parser.add_argument("--refit-stats", action="store_true")
    parser.add_argument("--gaussian-shrinkage", type=float, default=0.01)
    unfrozen = parser.add_argument_group("--unfrozen-25 only")
    unfrozen.add_argument("--init", choices=("checkpoint", "uniform"), default="checkpoint",
                          help="start the weights from --mix-checkpoint, or exactly uniform")
    unfrozen.add_argument("--layer-norm", action=argparse.BooleanOptionalAction, default=True,
                          help="fixed per-layer standardization before mixing")
    unfrozen.add_argument("--redundancy-weight", type=float, default=1.0)
    unfrozen.add_argument("--entropy-weight", type=float, default=0.01)
    unfrozen.add_argument("--collapse-ratio", type=float, default=0.5)
    unfrozen.add_argument("--stop-on-collapse", action="store_true")
    tracking = parser.add_argument_group("--track-metrics (per-epoch MoM scoring)")
    tracking.add_argument("--track-metrics", action="store_true",
                          help="score human vs. MoM songs after every epoch")
    tracking.add_argument("--metrics-cache", type=Path,
                          help="pooled MERT of the evaluation songs, built on first use "
                          "(default: {train-cache}/../eval-{metrics-human-split})")
    tracking.add_argument("--metrics-human-split", default="val50")
    tracking.add_argument("--data-root", type=Path,
                          help="holds the human manifests and audio (default: the one "
                          "the train cache was built from)")
    tracking.add_argument("--bucket", default="songs-and-mert-features")
    tracking.add_argument("--region", default=None)
    tracking.add_argument("--mom-prefix", default="mom")
    tracking.add_argument("--mom-root", type=Path, default=None,
                          help="read MoM from this local directory instead of S3")
    tracking.add_argument("--generators", nargs="+", default=None,
                          help="default: evaluate_mom.py's four")
    tracking.add_argument("--per-source", type=int, default=300)
    tracking.add_argument("--windows-per-song", type=int, default=4)
    tracking.add_argument("--metrics-seed", type=int, default=0, help="song sampling seed")
    return parser.parse_args()


def initial_logits(args: argparse.Namespace, streams: tuple[str, ...]) -> dict[str, torch.Tensor]:
    logits = load_mix_logits(args.mix_checkpoint)
    if args.unfrozen_25 and args.init == "uniform":
        return {s: torch.zeros_like(logits[s]) for s in streams}
    return {s: logits[s] for s in streams}


@torch.no_grad()
def evaluate(
    model: MertFlow, cache: PooledCache, indices: np.ndarray, batch_size: int, device, prefix: str
) -> dict[str, float]:
    """Flow and Gaussian NLL (nats per dim) over fixed cache rows, eval mode."""
    was_training = model.training
    model.eval()
    sums = {"nll": 0.0, "nll_gaussian": 0.0}
    for start in range(0, len(indices), batch_size):
        batch = cache.gather(indices[start : start + batch_size])
        terms = model.log_probs(torch.from_numpy(batch).to(device))
        sums["nll"] -= float(terms["flow"].sum())
        sums["nll_gaussian"] -= float(terms["gaussian"].sum())
    model.train(was_training)
    return {f"{prefix}/{k}": v / max(len(indices), 1) / model.n_features for k, v in sums.items()}


def metrics_tracker(args: argparse.Namespace, train: PooledCache, config: TrainConfig, device):
    """The cached evaluation songs for --track-metrics, built on first use."""
    import evaluate_mom
    from track_mom import EvalCache, build_eval_cache, cache_matches

    chain = parse_chain(train.meta.get("codec"))  # the evaluation songs go through the training chain
    root = args.metrics_cache or train.root.parent / (
        f"eval-{args.metrics_human_split}" + (f"-{chain_slug(chain)}" if chain else ""))
    settings = {
        "human_split": args.metrics_human_split,
        "per_source": args.per_source,
        "windows_per_song": args.windows_per_song,
        "window_seconds": float(train.meta["window_seconds"]),
        "seed": args.metrics_seed,
        "generators": list(args.generators or evaluate_mom.GENERATORS),
        "model_name": train.meta["model_name"],
        "codec": train.meta.get("codec"),
    }
    if not cache_matches(root, settings):
        build_eval_cache(
            root,
            data_root=args.data_root or Path(train.meta["data_root"]),
            human_split=settings["human_split"],
            generators=settings["generators"],
            per_source=settings["per_source"],
            windows_per_song=settings["windows_per_song"],
            window_seconds=settings["window_seconds"],
            seed=settings["seed"],
            bucket=args.bucket,
            region=args.region,
            mom_prefix=args.mom_prefix,
            mom_root=args.mom_root,
            mert_config=config.mert,
            device=device,
            codec=chain,
        )
    cache = EvalCache(root, device)
    print(f"metrics: {len(cache.rows)} cached windows ({args.metrics_human_split} vs. "
          f"{', '.join(settings['generators'])}) from {root}")
    return cache


def main() -> None:
    args = parse_args()
    arm = "frozen" if args.frozen_25 else "unfrozen"
    trainable = args.unfrozen_25
    layer_norm = trainable and args.layer_norm  # frozen: the raw mixes mqa-w100 consumed
    streams = (args.mix,)

    config = load_config(args.config)
    for value, setter in (
        (args.epochs, lambda v: setattr(config, "epochs", v)),
        (args.lr, lambda v: setattr(config.optim, "lr", v)),
        (args.seed, lambda v: setattr(config, "seed", v)),
        (args.device, lambda v: setattr(config, "device", v)),
        (args.log_every, lambda v: setattr(config, "log_every", v)),
    ):
        if value is not None:
            setter(value)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = pick_device(config.device)

    train = PooledCache(args.train_cache)
    val = PooledCache(args.val_cache) if args.val_cache else None
    for cache in [c for c in (val,) if c is not None]:
        for key in ("window_seconds", "model_name", "codec"):  # no "codec" = original audio
            if cache.meta.get(key) != train.meta.get(key):
                raise SystemExit(f"{cache.root} and {train.root} differ in {key}")
    codec = train.meta.get("codec")
    if train.meta["model_name"] != config.mert.model_name:
        raise SystemExit(f"cache was built with {train.meta['model_name']}, config says {config.mert.model_name}")
    config.data.window_seconds = float(train.meta["window_seconds"])  # what evaluate_mom will cut
    config.data.train_split = train.meta["split"]
    config.data.val_split = val.meta["split"] if val else "none"

    run = f"mert-flow-{arm}" + (f"-{chain_slug(parse_chain(codec))}" if codec else "")
    checkpoint_dir = args.checkpoint_dir or Path(config.checkpoint_dir) / run
    log_dir = args.log_dir or Path(config.log_dir) / time.strftime(f"{run}-%Y%m%d-%H%M%S")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"arm={arm}  mix={args.mix}  layer_norm={layer_norm}  device={device.type}\n"
        f"train[{train.meta['split']}]: {len(train)} windows   "
        f"val: {f'{len(val)} windows' if val else 'none'}   S={config.data.window_seconds:g}s   "
        f"codec {codec or 'none (original audio)'}"
    )

    tracker = metrics_tracker(args, train, config, device) if args.track_metrics else None
    if val is not None:
        val_rows = np.arange(min(args.val_max_windows or len(val), len(val)))
    n_train_eval = args.train_eval_windows or (len(val_rows) if val is not None else 20000)
    train_rows = np.sort(
        np.random.default_rng(config.seed + 1).choice(len(train), min(n_train_eval, len(train)), replace=False)
    )

    stats = layer_stats(train, device, args.stats_windows, args.refit_stats)
    moments = LayerCovariance(stats["mean"].to(device), stats["cov"].to(device), layer_norm)
    del stats

    model = MertFlow(train.n_layers, train.dim, streams, config.flow.style, layer_norm, trainable).to(device)
    init = initial_logits(args, streams)
    with torch.no_grad():
        for s in streams:
            model.logits[s].copy_(init[s].to(device))
        model.layer_mean.copy_(moments.layer_mean)
        model.layer_std.copy_(moments.layer_std)
    init_weights = {s: torch.softmax(init[s], 0) for s in streams}
    n_flow = sum(p.numel() for p in model.flow.parameters())
    print(f"flow over {model.n_features} features: {n_flow / 1e6:.1f}M parameters; "
          f"layer weights {'trainable' if trainable else 'frozen'} ({train.n_layers})")

    def current_moments() -> tuple[torch.Tensor, torch.Tensor]:
        return moments.moments([model.weights()[s] for s in streams])

    with torch.no_grad():
        mean0, cov0 = current_moments()
        model.set_moments(mean0, cov0, args.gaussian_shrinkage)
    if not trainable:
        del moments  # 2.6 GB; the frozen moments now live in the model buffers
    probe_source = val or train
    probe = torch.from_numpy(
        probe_source.gather(np.arange(min(args.probe_windows, len(probe_source))))
    ).to(device)
    if len(probe) <= model.n_features:
        print(f"note: {len(probe)} probe windows <= {model.n_features} features; "
              "the probe spectrum is rank-deficient by construction")

    flow_params = list(model.flow.parameters())
    groups = [{"params": flow_params, "lr": config.optim.lr}]
    if trainable:
        groups.append({"params": list(model.logits.parameters()), "lr": args.mix_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)  # decay pulls LU toward singular
    steps_per_epoch = len(train) // args.batch_size
    total_steps = config.epochs * steps_per_epoch
    scheduler = make_scheduler(optimizer, config, total_steps)
    print(f"{steps_per_epoch} steps/epoch x {config.epochs} epochs = {total_steps} steps, "
          f"batch {args.batch_size}")

    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("args", f"```json\n{json.dumps(vars(args), default=str, indent=1)}\n```")
    monitor = Monitor(writer, checkpoint_dir / "diagnostics.jsonl", args.collapse_ratio)
    (checkpoint_dir / "diagnostics.jsonl").unlink(missing_ok=True)

    def diagnose(step: int) -> bool:
        with torch.no_grad():
            mean, cov = current_moments() if trainable else (mean0, cov0)
            return monitor(diagnostics(model, mean, cov, init_weights, probe), step, streams)

    def save(epoch: int, step: int) -> Path:
        if trainable:
            with torch.no_grad():
                model.set_moments(*current_moments(), args.gaussian_shrinkage)
        path = checkpoint_dir / "mert_flow_last.pt"
        torch.save(
            {
                "kind": "mert_flow",
                "arm": arm,
                "streams": list(streams),
                "layer_norm": layer_norm,
                "n_layers": train.n_layers,
                "dim": train.dim,
                "model": model.state_dict(),
                "config": config.to_dict(),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "mix_checkpoint": str(args.mix_checkpoint),
                "train_cache": str(args.train_cache),
                "codec": codec,
                "init_weights": {s: w.tolist() for s, w in init_weights.items()},
                "epoch": epoch,
                "step": step,
            },
            path,
        )
        return path

    history_path = checkpoint_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)

    def record_epoch(epoch: int, step: int, running_nll: float | None) -> None:
        """Train/val NLL (and MoM metrics) of the current model -> history.jsonl."""
        record: dict = {"epoch": epoch, "step": step}
        if running_nll is not None:
            record["train_running/nll"] = running_nll
        record.update(evaluate(model, train, train_rows, args.batch_size, device, "train"))
        if val is not None:
            record.update(evaluate(model, val, val_rows, args.batch_size, device, "val"))
        if tracker is not None:
            was_training = model.training
            model.eval()
            record.update(tracker.score(model))
            model.train(was_training)
        for k, v in record.items():
            if isinstance(v, float):
                writer.add_scalar(k, v, step)
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        line = f"  epoch {epoch}: train nll {record['train/nll']:.4f}"
        if val is not None:
            line += (f"  val nll {record['val/nll']:.4f}  gap {record['val/nll'] - record['train/nll']:+.4f}"
                     f"  (gaussian val {record['val/nll_gaussian']:.4f})")
        line += "  nats/dim"
        if tracker is not None:
            line += (f"\n  epoch {epoch}: AUROC all {record['metrics/auroc/all/flow']:.3f} "
                     f"(gaussian {record['metrics/auroc/all/gaussian']:.3f})  " + "  ".join(
                         f"{g} {record[f'metrics/auroc/{g}/flow']:.3f}" for g in tracker.order[1:]))
        print(line, flush=True)
        if epoch:
            plot_run(checkpoint_dir)  # redrawn every epoch, so a running job can be checked

    collapsed = diagnose(0)
    record_epoch(0, 0, None)
    step = 0
    rng = np.random.default_rng(config.seed)
    model.train()
    for epoch in range(config.epochs):
        started = time.time()
        nll_sum, nll_steps = torch.zeros((), device=device), 0
        for batch in train.batches(args.batch_size, rng, drop_last=True):
            x = torch.from_numpy(batch).to(device)
            if trainable:
                mean, cov = current_moments()
                corr, std = correlation(cov)
                z = model.features(x, mean, std)
                redundancy = redundancy_penalty(corr)
                entropy = torch.stack([kl_to_uniform(w) for w in model.weights().values()]).mean()
            else:
                z = model.features(x)
                redundancy = entropy = torch.zeros((), device=device)
            nll = -model.flow_log_prob(z).mean() / model.n_features
            loss = nll + args.redundancy_weight * redundancy + args.entropy_weight * entropy
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step}: nll={float(nll)}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(flow_params, config.optim.grad_clip)
            if trainable:
                torch.nn.utils.clip_grad_norm_(model.logits.parameters(), config.optim.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            nll_sum += nll.detach()  # summed on the device: no per-step sync
            nll_steps += 1

            if step % config.log_every == 0 or step == 1:
                print(
                    f"epoch {epoch} step {step}/{total_steps}  nll/dim={float(nll.detach()):.4f}  "
                    f"redundancy={float(redundancy):.4f}  kl_uniform={float(entropy):.4f}",
                    flush=True,
                )
                writer.add_scalar("train/nll_per_dim", float(nll.detach()), step)
                writer.add_scalar("train/redundancy", float(redundancy), step)
                writer.add_scalar("train/kl_uniform", float(entropy), step)
                writer.add_scalar("train/loss", float(loss.detach()), step)
                writer.add_scalar("train/grad_norm", float(grad_norm), step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
            if args.diag_every and step % args.diag_every == 0:
                collapsed = diagnose(step)
                if collapsed and args.stop_on_collapse:
                    break
        print(f"epoch {epoch} finished in {time.time() - started:.0f}s")
        save(epoch + 1, step)  # also refreshes the unfrozen arm's standardization
        record_epoch(epoch + 1, step, float(nll_sum) / max(nll_steps, 1))
        if collapsed and args.stop_on_collapse:
            print("stopping: --stop-on-collapse")
            break
    diagnose(step)
    writer.close()
    for s, w in model.weights().items():
        print(f"final {s} weights: " + " ".join(f"{float(v):.3f}" for v in w))
    print(f"done at step {step}; checkpoint: {checkpoint_dir / 'mert_flow_last.pt'}")


if __name__ == "__main__":
    main()
