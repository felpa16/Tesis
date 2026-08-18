"""Full representation-learning model: layer mixes, encoders, bottlenecks, decoder.

`encode_mixes` is the canonical entry point downstream of the branch mixes —
the cycle-consistency loss re-encodes decoded mixes through it, and phase-2
training on cached mixes will call it directly (skipping MERT + LayerMix).
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import TrainConfig
from src.models.bottleneck import CrossAttentionBottleneck
from src.models.decoder import PerceiverDecoder
from src.models.mert import LayerMix
from src.models.transformer import TransformerEncoder


class Standardizer(nn.Module):
    """Running per-dimension mean/std (EMA) for reconstruction targets.

    Needed because the branch mixes drift while the layer weights train
    (phase 1); once the weights freeze the statistics settle and behave like
    fixed dataset statistics. Buffers travel with the model checkpoint.
    """

    def __init__(self, dim: int, momentum: float = 0.99, eps: float = 1e-5) -> None:
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float().reshape(-1, x.shape[-1])
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)
        if not bool(self.initialized):
            self.mean.copy_(mean)
            self.var.copy_(var)
            self.initialized.fill_(True)
        else:
            self.mean.mul_(self.momentum).add_(mean, alpha=1.0 - self.momentum)
            self.var.mul_(self.momentum).add_(var, alpha=1.0 - self.momentum)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sqrt(self.var + self.eps) + self.mean


class DisentanglementModel(nn.Module):
    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        if config.encoder.d_model != config.mert.dim:
            raise ValueError("encoder d_model must equal the MERT dimension")
        self.config = config

        self.content_mix = LayerMix(config.mert.n_hidden_states)
        self.style_mix = LayerMix(config.mert.n_hidden_states)

        self.content_encoder = TransformerEncoder(config.encoder)
        self.style_encoder = (
            self.content_encoder
            if config.encoder.share_encoder
            else TransformerEncoder(config.encoder)
        )
        self.content_bottleneck = CrossAttentionBottleneck(
            config.encoder.d_model, config.bottleneck
        )
        self.style_bottleneck = CrossAttentionBottleneck(
            config.encoder.d_model, config.bottleneck
        )
        self.decoder = PerceiverDecoder(
            config.decoder, config.bottleneck, out_dim=config.mert.dim
        )
        self.content_std = Standardizer(config.mert.dim, config.standardizer_momentum)
        self.style_std = Standardizer(config.mert.dim, config.standardizer_momentum)

    def mix(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 25, N, 1024) -> content mix, style mix, each (B, N, 1024)."""
        return self.content_mix(hidden_states), self.style_mix(hidden_states)

    def encode_mixes(
        self, content_mix: torch.Tensor, style_mix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Branch mixes -> content tokens (B,8,256), style tokens (B,8,256)."""
        content = self.content_bottleneck(self.content_encoder(content_mix))
        style = self.style_bottleneck(self.style_encoder(style_mix))
        return content, style

    def encode(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        content_mix, style_mix = self.mix(hidden_states)
        content, style = self.encode_mixes(content_mix, style_mix)
        return content, style, content_mix, style_mix

    def decode(
        self, content: torch.Tensor, style: torch.Tensor, n_frames: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Latents -> predicted (content mix, style mix) in STANDARDIZED space."""
        return self.decoder(content, style, n_frames)

    def freeze_layer_weights(self) -> None:
        self.content_mix.freeze()
        self.style_mix.freeze()

    def layer_weight_summary(self) -> dict[str, torch.Tensor]:
        return {
            "content": self.content_mix.softmax_weights.detach().cpu(),
            "style": self.style_mix.softmax_weights.detach().cpu(),
        }
