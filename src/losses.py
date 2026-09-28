"""Training objectives for the representation-learning stage (see CLAUDE.md).

1. mil_nce            — contrastive loss on content tokens (K=1 -> InfoNCE)
   ContentQueue       — FIFO of recent content vectors, extra negatives for it
2. pool_frames        — temporal pooling of the reconstruction target
   standardized_mse   — reconstruction terms 2a/2b on standardized mixes
   cycle_loss         — term 2c, decode-swap-re-encode with detached targets
3. cross_correlation_loss / hsic_loss — content-style decorrelation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.model import DisentanglementModel, Standardizer


def pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """(B, n_tokens, D) -> L2-normalized (B, D) for contrastive/retrieval use."""
    return F.normalize(tokens.mean(dim=1), dim=-1)


def pool_frames(x: torch.Tensor, factor: int) -> torch.Tensor:
    """(B, N, D) -> (B, N // factor, D), average-pooled over time.

    The reconstruction target is pooled because at 75 fps most of the
    standardized variance in MERT features is frame-to-frame detail that a
    16x256 latent set cannot represent, which drowns the between-window signal
    the latents *can* carry (CLAUDE.md, Decoder). Trailing frames that do not
    fill a block are dropped. factor=1 is the ablation and is a no-op.

    Pool before standardizing, never after: averaging shrinks the variance, so
    pooling standardized values would put the target below unit variance and
    break the "predict the dataset mean scores 1.0 per branch" yardstick.
    """
    if factor <= 1:
        return x
    return F.avg_pool1d(
        x.transpose(1, 2), kernel_size=factor, stride=factor
    ).transpose(1, 2)


def mil_nce(
    anchors: torch.Tensor,
    candidates: torch.Tensor,
    temperature: float,
    groups: torch.Tensor | None = None,
    queue: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """MIL-NCE over pooled content vectors.

    anchors: (P, D) — one per pair (cover A side).
    candidates: (P, K, D) — K candidate windows of cover B per pair; all K are
    treated as soft positives, every other pair's candidates as negatives.
    With K=1 this reduces to standard InfoNCE with in-batch negatives.
    groups: optional (P,) work id per pair. Another pair from the same work is
    the same composition, so its candidates are dropped from the denominator
    instead of being pushed away as negatives. The fewer works a dataset has,
    the more often this happens, so without the mask the loss would penalize
    small subsets more.
    queue: optional (vectors (Q, D), works (Q,)) of extra negatives from earlier
    steps, see ContentQueue. At P=4, K=1 an anchor otherwise has 3 negatives and
    the loss saturates (timeline.md, 2026-09-27); the queue makes the
    denominator large without making the batch large. Queue entries of the
    anchor's own work are masked out exactly like in-batch ones.
    """
    p, k, d = candidates.shape
    if p < 1 or (p < 2 and queue is None):
        return anchors.new_zeros(())
    anchors = F.normalize(anchors, dim=-1)
    flat = F.normalize(candidates.reshape(p * k, d), dim=-1)
    candidate_groups = None if groups is None else groups.repeat_interleave(k)
    if queue is not None:
        queue_vectors, queue_groups = queue
        flat = torch.cat([flat, F.normalize(queue_vectors.to(flat.dtype), dim=-1)])
        if candidate_groups is not None:
            candidate_groups = torch.cat(
                [candidate_groups, queue_groups.to(candidate_groups)]
            )
    logits = anchors @ flat.T / temperature  # (P, P*K + Q)
    positive = torch.zeros(
        p, flat.shape[0], dtype=torch.bool, device=logits.device
    )
    rows = torch.arange(p, device=logits.device).repeat_interleave(k)
    cols = torch.arange(p * k, device=logits.device)
    positive[rows, cols] = True
    if candidate_groups is not None:
        same_work = groups[:, None] == candidate_groups[None, :]
        logits = logits.masked_fill(same_work & ~positive, float("-inf"))
    pos_logsumexp = logits.masked_fill(~positive, float("-inf")).logsumexp(dim=1)
    all_logsumexp = logits.logsumexp(dim=1)
    return (all_logsumexp - pos_logsumexp).mean()


class ContentQueue(nn.Module):
    """FIFO of recent pooled content vectors, used as extra contrastive negatives.

    Why: at --batch-pairs 4 --n-candidates 1 each anchor sees 3 negatives, so
    the task is a 4-way choice that saturates once works separate coarsely — the
    measured plateau in timeline.md (2026-09-27). A queue enlarges the
    denominator without enlarging the batch, which matters because the batch is
    capped by MERT's activation memory, not by the contrastive term.

    What it does NOT fix: a model that has memorized which work each training
    recording belongs to ranks same-work candidates first however many negatives
    there are. The queue attacks saturation; only more works (or augmentation)
    attacks memorization.

    Entries are detached and stored L2-normalized, and they are not re-encoded
    as the encoder trains, so old entries go stale. Keep `size` small enough
    that the encoder's drift over size/(P*K) steps stays modest; the alternative
    is a momentum encoder (MoCo), which costs a second copy of the weights.

    The buffers are non-persistent, so the queue never enters a checkpoint: an
    older checkpoint still loads, and a resumed run refills over its first
    size/(P*K) steps.
    """

    def __init__(self, size: int, dim: int) -> None:
        super().__init__()
        self.size = int(size)
        self.register_buffer("vectors", torch.zeros(self.size, dim), persistent=False)
        self.register_buffer(
            "works", torch.full((self.size,), -1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "cursor", torch.zeros((), dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "filled", torch.zeros((), dtype=torch.long), persistent=False
        )

    def negatives(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The valid entries, or None until something has been pushed."""
        n = int(self.filled)
        if n == 0:
            return None
        return self.vectors[:n], self.works[:n]

    @torch.no_grad()
    def push(self, vectors: torch.Tensor, works: torch.Tensor) -> None:
        """Add this batch's candidate vectors, overwriting the oldest entries."""
        vectors = F.normalize(vectors.detach().float(), dim=-1)
        works = works.detach().reshape(-1).to(torch.long)
        if vectors.shape[0] > self.size:  # a batch larger than the queue
            vectors, works = vectors[-self.size :], works[-self.size :]
        n = vectors.shape[0]
        index = (torch.arange(n, device=vectors.device) + int(self.cursor)) % self.size
        self.vectors[index] = vectors.to(self.vectors.dtype)
        self.works[index] = works.to(self.works.device)
        self.cursor.fill_((int(self.cursor) + n) % self.size)
        self.filled.fill_(min(int(self.filled) + n, self.size))


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
