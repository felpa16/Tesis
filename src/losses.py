"""Training objectives for the representation-learning stage (see CLAUDE.md).

1. mil_nce            — contrastive loss on content tokens (K=1 -> InfoNCE)
2. standardized_mse   — reconstruction terms 2a/2b on standardized mixes
   cycle_loss         — term 2c, decode-swap-re-encode with detached targets
3. cross_correlation_loss / hsic_loss — content-style decorrelation
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.model import DisentanglementModel, Standardizer


def pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """(B, n_tokens, D) -> L2-normalized (B, D) for contrastive/retrieval use."""
    return F.normalize(tokens.mean(dim=1), dim=-1)


def mil_nce(
    anchors: torch.Tensor, candidates: torch.Tensor, temperature: float
) -> torch.Tensor:
    """MIL-NCE over pooled content vectors.

    anchors: (P, D) — one per pair (cover A side).
    candidates: (P, K, D) — K candidate windows of cover B per pair; all K are
    treated as soft positives, every other pair's candidates as negatives.
    With K=1 this reduces to standard InfoNCE with in-batch negatives.
    """
    p, k, d = candidates.shape
    if p < 2:
        return anchors.new_zeros(())
    anchors = F.normalize(anchors, dim=-1)
    flat = F.normalize(candidates.reshape(p * k, d), dim=-1)
    logits = anchors @ flat.T / temperature  # (P, P*K)
    positive = torch.zeros(p, p * k, dtype=torch.bool, device=logits.device)
    rows = torch.arange(p, device=logits.device).repeat_interleave(k)
    cols = torch.arange(p * k, device=logits.device)
    positive[rows, cols] = True
    pos_logsumexp = logits.masked_fill(~positive, float("-inf")).logsumexp(dim=1)
    all_logsumexp = logits.logsumexp(dim=1)
    return (all_logsumexp - pos_logsumexp).mean()


def standardized_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    standardizer: Standardizer,
    cosine_weight: float = 0.0,
) -> torch.Tensor:
    """MSE (optionally + cosine term) against the per-dim standardized target."""
    target = standardizer.normalize(target.detach())
    loss = F.mse_loss(pred, target)
    if cosine_weight > 0:
        loss = loss + cosine_weight * (
            1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
        )
    return loss


def cycle_loss(
    model: DisentanglementModel,
    content_swapped: torch.Tensor,
    style: torch.Tensor,
    n_frames: int,
) -> torch.Tensor:
    """Term 2c: decode(s_a, c_b) -> re-encode -> recover s_a and c_b.

    Targets are detached so the loss shapes the decode-re-encode path instead
    of dragging the original encodings around (CLAUDE.md).
    """
    pred_content_mix, pred_style_mix = model.decode(content_swapped, style, n_frames)
    content_mix = model.content_std.denormalize(pred_content_mix)
    style_mix = model.style_std.denormalize(pred_style_mix)
    content_rec, style_rec = model.encode_mixes(content_mix, style_mix)
    return F.mse_loss(content_rec, content_swapped.detach()) + F.mse_loss(
        style_rec, style.detach()
    )


def cross_correlation_loss(
    u: torch.Tensor, v: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Mean squared entry of the batch cross-correlation matrix between u and v."""
    if u.shape[0] < 2:
        return u.new_zeros(())
    u = (u - u.mean(dim=0)) / (u.std(dim=0) + eps)
    v = (v - v.mean(dim=0)) / (v.std(dim=0) + eps)
    corr = u.T @ v / u.shape[0]
    return corr.pow(2).mean()


def _rbf_kernel(x: torch.Tensor) -> torch.Tensor:
    d2 = torch.cdist(x, x).pow(2)
    off_diag = d2.detach()[~torch.eye(x.shape[0], dtype=torch.bool, device=x.device)]
    sigma2 = off_diag.median().clamp_min(1e-8)  # median heuristic, no grad
    return torch.exp(-d2 / (2.0 * sigma2))


def hsic_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Biased HSIC estimator with RBF kernels (median-heuristic bandwidth)."""
    b = u.shape[0]
    if b < 4:
        return u.new_zeros(())
    k = _rbf_kernel(u)
    l = _rbf_kernel(v)
    h = torch.eye(b, device=u.device, dtype=u.dtype) - 1.0 / b
    return torch.trace(k @ h @ l @ h) / (b - 1) ** 2
