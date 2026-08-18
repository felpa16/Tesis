"""Perceiver-IO-style reconstruction decoder (training scaffolding only).

Latent memory = content + style tokens projected to d_model with a learned
type embedding per branch (no positional encoding — the latents are an
unordered set). Output length N is set by repeating one shared learned query
N times, with positions injected via RoPE inside the query self-attention, so
variable-length and cross-length reconstruction come for free.

The decoder predicts BOTH branch mixes (content-mix and style-mix sequences),
matching the phase-2 caching decision in CLAUDE.md: the reconstruction target
is the pair of softmax-weighted MERT mixes, not the raw hidden states.
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import BottleneckConfig, DecoderConfig
from src.models.attention import MultiHeadAttention
from src.models.transformer import feed_forward


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_ratio: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, use_rope=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, use_rope=False)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = feed_forward(d_model, ffn_ratio, dropout=0.0)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.self_norm(x))
        x = x + self.cross_attn(self.cross_norm(x), kv=memory)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class PerceiverDecoder(nn.Module):
    """(content (B,8,256), style (B,8,256), n_frames) -> two (B, N, out_dim) mixes."""

    def __init__(
        self, config: DecoderConfig, bottleneck: BottleneckConfig, out_dim: int
    ) -> None:
        super().__init__()
        d_model = config.d_model
        self.n_tokens = bottleneck.n_tokens
        self.latent_proj = nn.Linear(bottleneck.token_dim, d_model)
        self.type_embed = nn.Parameter(torch.zeros(2, d_model))  # content, style
        self.memory_norm = nn.LayerNorm(d_model)
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.blocks = nn.ModuleList(
            DecoderBlock(d_model, config.n_heads, config.ffn_ratio)
            for _ in range(config.n_layers)
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 2 * out_dim)

    def forward(
        self, content: torch.Tensor, style: torch.Tensor, n_frames: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = torch.cat([content, style], dim=1)  # (B, 16, token_dim)
        memory = self.latent_proj(latents)
        types = torch.repeat_interleave(
            self.type_embed, self.n_tokens, dim=0
        )  # (16, d_model)
        memory = self.memory_norm(memory + types)

        x = self.query.expand(content.shape[0], n_frames, -1)
        for block in self.blocks:
            x = block(x, memory)
        out = self.out_proj(self.out_norm(x))
        content_mix_hat, style_mix_hat = out.chunk(2, dim=-1)
        return content_mix_hat, style_mix_hat
