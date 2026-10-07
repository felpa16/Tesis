"""Reusable pieces of the phase-1 training loop (imported by scripts/train.py).

Kept in src/ so smoke tests and future phase-2 scripts can exercise the exact
loss orchestration used in training.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import LossConfig, TrainConfig
from src.losses import (
    cross_correlation_loss,
    cycle_loss,
    hsic_loss,
    mil_nce,
    pool_frames,
    pool_tokens,
    standardized_mse,
)
from src.models.mert import MertExtractor
from src.models.model import DisentanglementModel


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def worker_init(worker_id: int) -> None:
    """Give DataLoader workers distinct python/numpy RNG states."""
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_optimizer(
    model: torch.nn.Module, config: TrainConfig, weight_decay: float | None = None
) -> torch.optim.AdamW:
    """AdamW with weight decay on matrix-shaped parameters only.

    Flow training passes weight_decay=0.0: decaying LU matrices and ActNorm
    scales pulls the transforms toward singularity.
    """
    if weight_decay is None:
        weight_decay = config.optim.weight_decay
    decay, no_decay = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.optim.lr,
    )


def make_scheduler(optimizer, config: TrainConfig, total_steps: int):
    """Linear warmup then cosine decay to min_lr_ratio * lr."""
    warmup = config.optim.warmup_steps
    floor = config.optim.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = min((step - warmup) / max(total_steps - warmup, 1), 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_layer_weights(
    model: DisentanglementModel, path: Path, recon_pool: int
) -> None:
    """Start both LayerMix vectors from a frozen phase-1 layer-weights file.

    The file (phase1_layer_weights.pt) carries each branch's logits, their
    softmax as a round-trip check, and the Standardizer statistics phase 1
    ended with. The statistics are loaded only when they were measured at the
    same recon_pool. Otherwise they describe a different target, and the EMA
    starts fresh from the first batch instead.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for branch, mix in (("content", model.content_mix), ("style", model.style_mix)):
        logits = checkpoint[f"{branch}_logits"]
        if logits.shape != mix.weights.shape:
            raise SystemExit(
                f"{path}: {branch} logits have shape {tuple(logits.shape)}, "
                f"the model expects {tuple(mix.weights.shape)}"
            )
        with torch.no_grad():
            mix.weights.copy_(logits)
        stored = checkpoint.get(f"{branch}_softmax")
        if stored is not None and not torch.allclose(
            mix.softmax_weights.detach().cpu(), stored, atol=1e-5
        ):
            raise SystemExit(f"{path}: {branch} softmax does not reproduce from its logits")

    stored_pool = checkpoint.get("config", {}).get("loss", {}).get("recon_pool")
    if stored_pool != recon_pool:
        print(
            f"layer weights: standardizer stats skipped (measured at recon_pool "
            f"{stored_pool}, this run uses {recon_pool})"
        )
        return
    for branch, std in (("content", model.content_std), ("style", model.style_std)):
        mean = checkpoint.get(f"{branch}_std_mean")
        var = checkpoint.get(f"{branch}_std_var")
        if mean is None or var is None:
            continue
        std.mean.copy_(mean)
        std.var.copy_(var)
        std.initialized.fill_(True)


class LayerMixConvergence:
    """Stopping rule for a run whose only job is to converge the layer mixes.

    Scale is the whole difficulty here, so the test is *relative*. Both
    vectors are a softmax over 25 hidden states: they start at 0.04 each, and
    the entire journey from uniform to a converged mix is small — run #2's
    content vector ended at ||w - uniform|| = 0.033, i.e. an MSE of 4.4e-5
    against uniform. An absolute MSE threshold is therefore worse than useless:
    MSE < 1e-3 means ||delta|| < 0.158, almost five times the whole journey, so
    it is satisfied before the first step and would "converge" instantly.

    So: has each vector moved less than `tol` of how far it has come from
    uniform, measured over the last `window` samples, `patience` checks
    running? That is condition 1 of the CLAUDE.md freeze criterion with a
    meaningful scale attached.

    Conditions 2 and 3 are *reported*, never used to stop. A vector that never
    left uniform passes every stationarity test there is (CLAUDE.md), and run
    #1's style vector did exactly that because its only gradient source was
    degenerate — so `warn_cos_uniform` flags a mix that has not actually
    selected anything, and the caller is expected to check its objective is
    learning before trusting the file.
    """

    def __init__(
        self,
        tol: float = 0.02,
        window: int = 4,
        patience: int = 3,
        warn_cos_uniform: float = 0.9995,
    ) -> None:
        self.tol = float(tol)
        self.window = int(window)
        self.patience = int(patience)
        self.warn_cos_uniform = float(warn_cos_uniform)
        self.history: dict[str, list[torch.Tensor]] = {}
        self.passes = 0

    def update(self, weights: dict[str, torch.Tensor]) -> tuple[bool, str]:
        """Record one sample; return (converged, a line to print)."""
        parts, moved = [], []
        for branch, w in sorted(weights.items()):
            w = w.detach().float().cpu()
            n = w.numel()
            uniform = torch.full((n,), 1.0 / n)
            deviation = float((w - uniform).norm())
            past = self.history.setdefault(branch, [])
            past.append(w)
            if len(past) > self.window + 1:
                past.pop(0)

            if len(past) > self.window:
                delta = float((w - past[0]).norm())
                # relative to the journey so far, not to the weights themselves
                relative = delta / max(deviation, 1e-12)
                moved.append(relative)
                parts.append(
                    f"{branch} moved {relative:7.4f} of its {deviation:.4f} "
                    f"departure (cos to uniform {float(F.cosine_similarity(w, uniform, dim=0)):.4f}, "
                    f"max/min {float(w.max() / w.min().clamp_min(1e-12)):.2f}x)"
                )
            else:
                parts.append(
                    f"{branch} warming up ({len(past)}/{self.window + 1} samples), "
                    f"departure {deviation:.4f}"
                )

        if moved and max(moved) < self.tol:
            self.passes += 1
        else:
            self.passes = 0
        converged = self.passes >= self.patience
        suffix = f"  [{self.passes}/{self.patience} consecutive]" if moved else ""
        return converged, "; ".join(parts) + suffix

    def warnings(self, weights: dict[str, torch.Tensor]) -> list[str]:
        """Branches that converged without ever leaving uniform."""
        out = []
        for branch, w in sorted(weights.items()):
            w = w.detach().float().cpu()
            uniform = torch.full((w.numel(),), 1.0 / w.numel())
            cosine = float(F.cosine_similarity(w, uniform, dim=0))
            if cosine > self.warn_cos_uniform:
                out.append(
                    f"{branch}: cos to uniform {cosine:.6f} — this vector never "
                    f"left its initialization, so its stationarity means nothing. "
                    f"Check that the objective driving it is learning "
                    f"(scripts/inspect_phase1.py) before freezing it."
                )
        return out


def save_layer_weights(
    path: Path, model: DisentanglementModel, config: TrainConfig, step: int
) -> None:
    """Write the two mixes in the format load_layer_weights() reads.

    Carries the logits (what the model restores), their softmax as a
    round-trip check, the Standardizer statistics, and the config — the
    statistics are only reusable at the same recon_pool, which is why
    load_layer_weights checks it.
    """
    payload: dict = {"config": config.to_dict(), "step": step}
    for branch, mix, std in (
        ("content", model.content_mix, model.content_std),
        ("style", model.style_mix, model.style_std),
    ):
        payload[f"{branch}_logits"] = mix.weights.detach().cpu().clone()
        payload[f"{branch}_softmax"] = mix.softmax_weights.detach().cpu().clone()
        payload[f"{branch}_std_mean"] = std.mean.detach().cpu().clone()
        payload[f"{branch}_std_var"] = std.var.detach().cpu().clone()
    torch.save(payload, path)


def extract_mixes(
    mert: MertExtractor,
    model: DisentanglementModel,
    waves: torch.Tensor,
    micro_batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Waveforms (B, L) -> branch mixes (B, N, 1024) x2, micro-batched.

    The stacked 25-layer hidden states are collapsed to the two mixes chunk by
    chunk so the full (B, 25, N, 1024) tensor never materializes. Gradients
    flow only into the LayerMix weights (MERT itself runs under no_grad).
    """
    content_chunks, style_chunks = [], []
    for chunk in waves.split(micro_batch):
        hidden = mert(chunk)
        content_mix, style_mix = model.mix(hidden)
        content_chunks.append(content_mix)
        style_chunks.append(style_mix)
    return torch.cat(content_chunks), torch.cat(style_chunks)


def pooled_targets(
    loss_config: LossConfig, content_mix: torch.Tensor, style_mix: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Branch mixes -> reconstruction targets, average-pooled over time.

    The encoders still consume the mixes at full 75 Hz resolution; only the
    decoder's target is pooled. Feed these to the Standardizer as well, so its
    statistics describe the distribution the loss is actually scored against.
    """
    factor = loss_config.recon_pool
    return pool_frames(content_mix, factor), pool_frames(style_mix, factor)


def compute_losses(
    model: DisentanglementModel,
    loss_config: LossConfig,
    content: torch.Tensor,
    style: torch.Tensor,
    content_target: torch.Tensor,
    style_target: torch.Tensor,
    n_pairs: int,
    n_candidates: int,
    song_ids: torch.Tensor | None = None,
    queue: tuple[torch.Tensor, torch.Tensor] | None = None,
    recon_index: torch.Tensor | None = None,
    keys: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """All objectives for one batch.

    content_target/style_target are the *pooled* mixes from pooled_targets(),
    so n_frames below is the pooled length and the decoder emits that many
    frames rather than the full 75 Hz sequence.

    song_ids, (P,) work id per pair, keeps two pairs of the same work from
    being contrasted as negatives (see mil_nce).

    queue holds extra contrastive negatives from earlier steps (ContentQueue).

    keys, (P*K, D), are the B-side vectors from the momentum encoder. When a
    queue is in use they replace the online encoder's B-side output as the
    contrastive positives, so the positive and the queued negatives come from
    the same (slowly moving) encoder. Without that the model can separate them
    by age instead of by content and the representation collapses.

    recon_index selects which windows the plain reconstruction term 2a is scored
    on; None means every window. It exists so augmented windows can feed the
    contrastive loss without becoming reconstruction targets — the decoder and
    the style branch must only ever be fitted to real recordings.

    Batch layout along dim 0 (established by the train script):
        [0, P)              A-side windows of the P aligned pairs
        [P, P + P*K)        the K candidate B-side windows per pair, flattened
        [P + P*K, B)        plain track windows (reconstruction only)
    """
    p, k = n_pairs, n_candidates
    batch, n_frames = content_target.shape[0], content_target.shape[1]
    device = content.device
    losses: dict[str, torch.Tensor] = {}

    # 2a. plain reconstruction (main weight), on real windows only
    if recon_index is None:
        recon_content, recon_style = content, style
        target_content, target_style = content_target, style_target
    else:
        recon_content, recon_style = content[recon_index], style[recon_index]
        target_content = content_target[recon_index]
        target_style = style_target[recon_index]
    pred_content, pred_style = model.decode(recon_content, recon_style, n_frames)
    losses["recon"] = standardized_mse(
        pred_content, target_content, model.content_std, loss_config.cosine_weight
    ) + standardized_mse(
        pred_style, target_style, model.style_std, loss_config.cosine_weight
    )

    # 1. contrastive on content tokens (MIL-NCE over the K candidates)
    if p > 0 and loss_config.contrastive_weight > 0:
        anchors = pool_tokens(content[:p])
        candidates = (
            keys if keys is not None else pool_tokens(content[p : p + p * k])
        ).view(p, k, -1)
        losses["contrastive"] = mil_nce(
            anchors, candidates, loss_config.temperature, song_ids, queue
        )

    # 2b. cover-swap reconstruction: decode(c_a, s_b) vs. B's mixes (low weight)
    if p > 0 and loss_config.swap_weight > 0:
        b0 = p + torch.arange(p, device=device) * k  # first candidate per pair
        pred_content, pred_style = model.decode(content[:p], style[b0], n_frames)
        losses["swap"] = standardized_mse(
            pred_content,
            content_target[b0],
            model.content_std,
            loss_config.cosine_weight,
        ) + standardized_mse(
            pred_style, style_target[b0], model.style_std, loss_config.cosine_weight
        )

    # 2c. latent cycle-consistency on a random subset with a derangement
    if loss_config.cycle_weight > 0 and batch >= 2:
        m = min(batch, max(2, round(loss_config.cycle_fraction * batch)))
        idx = torch.randperm(batch, device=device)[:m]
        losses["cycle"] = cycle_loss(
            model, content[idx.roll(1)], style[idx], n_frames
        )

    # 3. content-style decorrelation
    if loss_config.decorrelation_weight > 0:
        decorrelate = (
            cross_correlation_loss
            if loss_config.decorrelation == "xcorr"
            else hsic_loss
        )
        losses["decorrelation"] = decorrelate(content.flatten(1), style.flatten(1))

    weights = {
        "recon": loss_config.recon_weight,
        "contrastive": loss_config.contrastive_weight,
        "swap": loss_config.swap_weight,
        "cycle": loss_config.cycle_weight,
        "decorrelation": loss_config.decorrelation_weight,
    }
    total = sum(weights[name] * value for name, value in losses.items())
    return total, losses
