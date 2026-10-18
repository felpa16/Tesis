"""Supervised linear probe on time-pooled MERT: a diagnostic, never the detector.

The detector learns p(human music) and never sees synthetic audio. This probe
does: it is trained human-vs-synthetic on MoM generators and tested on a
generator it never saw (scripts/train_mert_probe.py, evaluate_mert_probe.py).
What it measures is how much human-vs-synthetic information the pooled MERT
features carry at all. If even a supervised probe cannot separate a held-out
generator, no density model on these features will; if it can, the
information is there and it is the likelihood score that fails to use it.

Same input as the MERT-only flow (src/models/mert_flow.py): the (25, 1024)
time-mean-pooled hidden states of one window. Each layer is standardized by
fixed training-set statistics, so no layer wins the mix by scale alone; one
softmax vector of 25 trainable weights mixes them; one linear layer turns the
mix into a logit.
"""

from __future__ import annotations

import torch
from torch import nn


class MertProbe(nn.Module):
    """(B, 25, 1024) pooled layers -> (B,) logits, positive = synthetic."""

    def __init__(self, n_layers: int, dim: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))  # uniform mix at init
        self.register_buffer("layer_mean", torch.zeros(n_layers, dim))
        self.register_buffer("layer_std", torch.ones(n_layers, dim))
        self.head = nn.Linear(dim, 1)

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        z = (pooled - self.layer_mean) / self.layer_std
        mix = torch.einsum("bld,l->bd", z, self.weights())
        return self.head(mix).squeeze(-1)
