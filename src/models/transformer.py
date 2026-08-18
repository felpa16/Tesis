"""Standard pre-LN Transformer encoder operating on MERT-dim sequences.

(B, N, d_model) -> (B, N, d_model); sequence length is never reduced — the
cross-attention bottleneck does the summarization afterwards.
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import EncoderConfig
from src.models.attention import MultiHeadAttention


def feed_forward(d_model: int, ffn_ratio: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, ffn_ratio * d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(ffn_ratio * d_model, d_model),
    )


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, ffn_ratio: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, use_rope=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = feed_forward(d_model, ffn_ratio, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            TransformerBlock(
                config.d_model, config.n_heads, config.ffn_ratio, config.dropout
            )
            for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
