"""Held-out evaluation metrics for the content branch.

Validation (scripts/train.py) and test evaluation (scripts/evaluate_encoder.py)
share these, so a number on val50 and a number on test50 mean the same thing.
"""

from __future__ import annotations

import numpy as np
import torch


def content_retrieval(
    a: torch.Tensor,
    b: torch.Tensor,
    works: torch.Tensor,
    tracks_a: torch.Tensor,
    tracks_b: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Per-query cover retrieval with pooled content vectors.

    a, b: (Q, D) L2-normalized pooled content of the A and B windows of Q
    aligned pairs. works: (Q,) work id per pair. tracks_a, tracks_b: (Q,)
    integer ids of the recording each window was cut from.

    Query i is A-window i, and every B window is a candidate:
      pair_hit  the nearest B is its own pair's B (exact aligned window)
      work_hit  the nearest B belongs to the same work (any cover of it)
      work_ap   average precision of ranking the same-work B windows first

    B windows cut from the query's own recording are removed from the ranking,
    because finding the same recording again is not cover retrieval. With
    several pairs per work, a track can be on the A side of one pair and the
    B side of another.
    """
    q = a.shape[0]
    idx = torch.arange(q)
    similarity = a @ b.T
    ignore = (tracks_b[None, :] == tracks_a[:, None]) & ~torch.eye(q, dtype=torch.bool)
    similarity = similarity.masked_fill(ignore, float("-inf"))
    relevant = (works[:, None] == works[None, :]) & ~ignore

    top = similarity.argmax(dim=1)
    order = similarity.argsort(dim=1, descending=True)
    ranked = relevant.gather(1, order).float()
    precision = ranked.cumsum(dim=1) / torch.arange(1, q + 1)
    return {
        "pair_hit": (top == idx).float(),
        "work_hit": relevant[idx, top].float(),
        "work_ap": (precision * ranked).sum(dim=1) / ranked.sum(dim=1).clamp_min(1.0),
    }


def bootstrap_over_works(
    values, works, n: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% interval of a per-query mean, resampling *works* with replacement.

    Queries from one work are correlated (same composition, often the same
    recordings), so the work is the independent unit. Resampling queries
    would understate the uncertainty. values and works are per-query arrays
    (tensors or lists). Pass a per-query *difference* between two runs scored
    on the same queries to get a paired interval for the gap.
    """
    values_np = np.asarray(values, dtype=np.float64)
    works_np = np.asarray(works)
    unique = np.unique(works_np)
    sums = np.array([values_np[works_np == w].sum() for w in unique])
    counts = np.array([(works_np == w).sum() for w in unique])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(n, len(unique)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)
