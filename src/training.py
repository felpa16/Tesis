"""Reusable pieces of the phase-1 training loop (imported by scripts/train.py).

Kept in src/ so smoke tests and future phase-2 scripts can exercise the exact
loss orchestration used in training.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch

from src.config import LossConfig, TrainConfig
from src.losses import (
    cross_correlation_loss,
    cycle_loss,
    hsic_loss,
    mil_nce,
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


def compute_losses(
    model: DisentanglementModel,
    loss_config: LossConfig,
    content: torch.Tensor,
    style: torch.Tensor,
    content_mix: torch.Tensor,
    style_mix: torch.Tensor,
    n_pairs: int,
    n_candidates: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """All objectives for one batch.

    Batch layout along dim 0 (established by the train script):
        [0, P)              A-side windows of the P aligned pairs
        [P, P + P*K)        the K candidate B-side windows per pair, flattened
        [P + P*K, B)        plain track windows (reconstruction only)
    """
    p, k = n_pairs, n_candidates
    batch, n_frames = content_mix.shape[0], content_mix.shape[1]
    device = content.device
    losses: dict[str, torch.Tensor] = {}

    # 2a. plain reconstruction on every window (main weight)
    pred_content, pred_style = model.decode(content, style, n_frames)
    losses["recon"] = standardized_mse(
        pred_content, content_mix, model.content_std, loss_config.cosine_weight
    ) + standardized_mse(
        pred_style, style_mix, model.style_std, loss_config.cosine_weight
    )

    # 1. contrastive on content tokens (MIL-NCE over the K candidates)
    if p > 0 and loss_config.contrastive_weight > 0:
        anchors = pool_tokens(content[:p])
        candidates = pool_tokens(content[p : p + p * k]).view(p, k, -1)
        losses["contrastive"] = mil_nce(
            anchors, candidates, loss_config.temperature
        )

    # 2b. cover-swap reconstruction: decode(c_a, s_b) vs. B's mixes (low weight)
    if p > 0 and loss_config.swap_weight > 0:
        b0 = p + torch.arange(p, device=device) * k  # first candidate per pair
        pred_content, pred_style = model.decode(content[:p], style[b0], n_frames)
        losses["swap"] = standardized_mse(
            pred_content, content_mix[b0], model.content_std, loss_config.cosine_weight
        ) + standardized_mse(
            pred_style, style_mix[b0], model.style_std, loss_config.cosine_weight
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
