"""Frozen MERT feature extraction and per-branch layer mixing.

MERT-v1-330M is used purely as a pretrained feature extractor: its weights
are frozen and its forward pass runs under no_grad. Each encoder branch owns
a LayerMix — a learnable softmax-weighted sum over the 25 hidden states —
whose weights are the only trainable parameters touching MERT's output
(phase-1 online strategy, see CLAUDE.md "Feature caching strategy").
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import MertConfig


class MertExtractor(nn.Module):
    """Frozen MERT: raw waveform (B, L) -> stacked hidden states (B, 25, N, 1024).

    The model's preprocessor has do_normalize=false, so waveforms are fed in
    directly with no feature-extractor roundtrip.
    """

    def __init__(self, config: MertConfig) -> None:
        super().__init__()
        from transformers import AutoModel

        self.config = config
        self.model = AutoModel.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        self.model.eval()
        self.model.requires_grad_(False)

    def train(self, mode: bool = True) -> "MertExtractor":
        super().train(mode)
        self.model.eval()  # stays frozen regardless of outer train/eval calls
        return self

    @torch.no_grad()
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_values=wave, output_hidden_states=True)
        return torch.stack(outputs.hidden_states, dim=1)


class LayerMix(nn.Module):
    """Softmax-weighted sum over MERT hidden states: (B, L, N, D) -> (B, N, D)."""

    def __init__(self, n_hidden_states: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_hidden_states))

    @property
    def softmax_weights(self) -> torch.Tensor:
        return torch.softmax(self.weights, dim=0)

    def freeze(self) -> None:
        self.weights.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weights = self.softmax_weights.to(hidden_states.dtype)
        return torch.einsum("blnd,l->bnd", hidden_states, weights)
