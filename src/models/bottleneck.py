"""Cross-attention bottleneck: variable-length sequence -> fixed latent tokens.

Learnable latent queries cross-attend over the Transformer output (no RoPE —
the queries are an unordered set), then a Linear + LayerNorm maps each token
to the latent dimension: (B, N, d_model) -> (B, n_tokens, token_dim).
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import BottleneckConfig
from src.models.attention import MultiHeadAttention


class CrossAttentionBottleneck(nn.Module):
    def __init__(self, d_model: int, config: BottleneckConfig) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(config.n_tokens, d_model) * 0.02)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, config.n_heads, use_rope=False)
        self.proj = nn.Linear(d_model, config.token_dim)
        self.norm = nn.LayerNorm(config.token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        tokens = self.attn(queries, kv=self.kv_norm(x))
        return self.norm(self.proj(tokens))
