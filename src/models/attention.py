"""Multi-head attention with optional Rotary Positional Embeddings.

One module serves self-attention (kv=None, RoPE allowed) and cross-attention
(kv given, no RoPE — used where keys are unordered sets: bottleneck latent
queries over encoder output, decoder queries over latent memory).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE cos/sin tables, grown lazily to the longest sequence."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = base ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    def forward(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._cos
        if (
            cached is None
            or cached.shape[0] < n
            or cached.device != device
            or cached.dtype != dtype
        ):
            positions = torch.arange(n, device=device, dtype=torch.float32)
            freqs = torch.outer(positions, self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos = emb.cos().to(dtype)
            self._sin = emb.sin().to(dtype)
        return self._cos[:n], self._sin[:n]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q and k (B, H, N, D) by position tables (N, D)."""
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_rope: bool = False) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"{d_model=} not divisible by {n_heads=}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim) if use_rope else None

    def forward(
        self, query: torch.Tensor, kv: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.rope is not None and kv is not None:
            raise ValueError("RoPE is only valid for self-attention")
        kv = query if kv is None else kv
        b, n_q, _ = query.shape
        n_kv = kv.shape[1]

        q = self.q_proj(query).view(b, n_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(b, n_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(b, n_kv, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            cos, sin = self.rope(n_q, q.device, q.dtype)
            q, k = apply_rope(q, k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n_q, self.n_heads * self.head_dim)
        return self.out_proj(out)
