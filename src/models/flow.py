"""Downstream detection: conditional normalizing flows on the learned latents.

Implements the CLAUDE.md "Downstream Detection" stage on zuko building blocks
(hand-rolled flows are prone to subtle log-determinant bugs, and the flow is
not the research contribution). Each flow step, applied in the normalizing
data -> noise direction, is:

    ActNorm-style affine -> invertible LU-parameterized linear -> RQS coupling

where the coupling conditioner MLP sees [z_half, context]. Latent tokens are
flattened and whitened with fixed training-set statistics (fitted once via
`Whitener.fit`, stored in buffers); because inputs are whitened, the
identity-initialized affine starts exactly where ActNorm's data-dependent
initialization would. All log_prob values include the whitening
log-determinant, so they are true densities over the unwhitened latents.

Detection score (factorized joint, low score -> synthetic / OOD):

    score = log p(style | content) + alpha * log p(content)
"""

from __future__ import annotations

import torch
from torch import nn
from zuko.flows import (
    Flow,
    GeneralCouplingTransform,
    UnconditionalDistribution,
    UnconditionalTransform,
)
from zuko.distributions import DiagNormal
from zuko.transforms import (
    LULinearTransform,
    MonotonicAffineTransform,
    MonotonicRQSTransform,
)

from src.config import BottleneckConfig, FlowConfig, NsfConfig
from src.models.attention import MultiHeadAttention


def latent_features(bottleneck: BottleneckConfig) -> int:
    """Dimensionality of one flattened token set: n_tokens * token_dim."""
    return bottleneck.n_tokens * bottleneck.token_dim


class Whitener(nn.Module):
    """Fixed mean/std whitening from training-set statistics.

    Unlike the EMA `Standardizer`, statistics are fitted once (the encoders
    are frozen at this stage) and stored in buffers, so they travel with the
    flow checkpoint. `log_det` is the whitening log-Jacobian to add to the
    flow's log_prob so scores are densities in the unwhitened space.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> None:
        """Fit from (M, dim) latents (typically a few thousand train windows)."""
        x = x.detach().float()
        self.mean.copy_(x.mean(dim=0))
        self.std.copy_(x.std(dim=0).clamp_min(self.eps))
        self.initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.initialized):
            raise RuntimeError("Whitener.fit() must run before computing log_probs")
        return (x - self.mean) / self.std

    def log_det(self) -> torch.Tensor:
        return -self.std.log().sum()


class ContentPooler(nn.Module):
    """Content tokens (B, n_tokens, token_dim) -> context vector (B, context_dim).

    One learned query cross-attends over the tokens (no RoPE — the tokens are
    an unordered set, as in the bottleneck), followed by an MLP.
    """

    def __init__(self, token_dim: int, config: FlowConfig) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, token_dim) * 0.02)
        self.kv_norm = nn.LayerNorm(token_dim)
        self.attn = MultiHeadAttention(token_dim, config.pool_heads, use_rope=False)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, config.context_dim),
            nn.GELU(),
            nn.Linear(config.context_dim, config.context_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self.query.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        pooled = self.attn(query, kv=self.kv_norm(tokens)).squeeze(1)
        return self.mlp(pooled)


def build_nsf(features: int, context: int, config: NsfConfig) -> Flow:
    """Neural spline flow over `features` dims, optionally conditional.

    Transforms are listed (and applied) in the data -> noise direction, so
    each step reads ActNorm -> LU linear -> RQS coupling as in CLAUDE.md.
    Identity initialization: zero affine, LU = I, and zuko's coupling MLPs
    start near-identity, so training begins from a well-conditioned map.
    """
    half_mask = torch.arange(features) < features // 2
    transforms = []
    for step in range(config.n_transforms):
        transforms += [
            # learnable per-dim affine; with whitened inputs, identity init
            # equals ActNorm's data-dependent init
            UnconditionalTransform(
                MonotonicAffineTransform,
                torch.zeros(features),
                torch.zeros(features),
            ),
            UnconditionalTransform(LULinearTransform, torch.eye(features)),
            GeneralCouplingTransform(
                features=features,
                context=context,
                mask=half_mask if step % 2 == 0 else ~half_mask,
                univariate=MonotonicRQSTransform,
                shapes=[(config.n_bins,), (config.n_bins,), (config.n_bins - 1,)],
                hidden_features=tuple(config.hidden_features),
            ),
        ]
    base = UnconditionalDistribution(
        DiagNormal, torch.zeros(features), torch.ones(features), buffer=True
    )
    return Flow(transform=transforms, base=base)


class ConditionalStyleFlow(nn.Module):
    """log p(style | content): the core detection density.

    Style tokens are flattened and whitened; content tokens are pooled into
    the context vector that conditions every coupling layer.
    """

    def __init__(self, bottleneck: BottleneckConfig, config: FlowConfig) -> None:
        super().__init__()
        features = latent_features(bottleneck)
        self.pooler = ContentPooler(bottleneck.token_dim, config)
        self.whitener = Whitener(features)
        self.flow = build_nsf(features, config.context_dim, config.style)

    def log_prob(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """(B, n_tokens, token_dim) x2 -> (B,) log p(style | content)."""
        z = self.whitener(style.flatten(1))
        context = self.pooler(content)
        return self.flow(context).log_prob(z) + self.whitener.log_det()


class MarginalFlow(nn.Module):
    """Unconditional flow over one flattened token set.

    Used for the p(content) term of the factorized score; also reusable as
    the cheap unconditional style flow in the likelihood-ratio sanity check
    log p(style | content) - log p(style).
    """

    def __init__(self, bottleneck: BottleneckConfig, config: NsfConfig) -> None:
        super().__init__()
        features = latent_features(bottleneck)
        self.whitener = Whitener(features)
        self.flow = build_nsf(features, 0, config)

    def log_prob(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, n_tokens, token_dim) -> (B,) log p(tokens)."""
        z = self.whitener(tokens.flatten(1))
        return self.flow().log_prob(z) + self.whitener.log_det()


class ConditionalGaussian(nn.Module):
    """Sanity baseline: diagonal Gaussian p(style | content) from an MLP on c."""

    def __init__(self, bottleneck: BottleneckConfig, config: FlowConfig) -> None:
        super().__init__()
        features = latent_features(bottleneck)
        self.pooler = ContentPooler(bottleneck.token_dim, config)
        self.whitener = Whitener(features)
        self.head = nn.Sequential(
            nn.Linear(config.context_dim, config.context_dim),
            nn.GELU(),
            nn.Linear(config.context_dim, 2 * features),
        )

    def log_prob(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """(B, n_tokens, token_dim) x2 -> (B,) log p(style | content)."""
        z = self.whitener(style.flatten(1))
        mean, log_std = self.head(self.pooler(content)).chunk(2, dim=-1)
        log_std = log_std.clamp(-7.0, 7.0)
        gaussian = torch.distributions.Normal(mean, log_std.exp())
        return gaussian.log_prob(z).sum(dim=-1) + self.whitener.log_det()


class FlowDetector(nn.Module):
    """Factorized density model: conditional style flow + content marginal.

    Keeping the two terms separate (rather than one monolithic joint flow)
    enables the alpha sweep that tests the core research hypothesis and shows
    per-song which term flagged it (CLAUDE.md "Downstream Detection").
    """

    def __init__(self, bottleneck: BottleneckConfig, config: FlowConfig) -> None:
        super().__init__()
        self.config = config
        self.style_flow = ConditionalStyleFlow(bottleneck, config)
        self.content_flow = MarginalFlow(bottleneck, config.content)

    def fit_whitening(
        self, content_tokens: torch.Tensor, style_tokens: torch.Tensor
    ) -> None:
        """Fit both whiteners from (M, n_tokens, token_dim) training latents."""
        self.style_flow.whitener.fit(style_tokens.flatten(1))
        self.content_flow.whitener.fit(content_tokens.flatten(1))

    def log_probs(
        self, content: torch.Tensor, style: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Both factorized terms, each (B,) — kept separate for diagnostics."""
        return {
            "style_given_content": self.style_flow.log_prob(content, style),
            "content": self.content_flow.log_prob(content),
        }

    def score(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        alpha: float | None = None,
    ) -> torch.Tensor:
        """Per-window score = log p(style | content) + alpha * log p(content)."""
        alpha = self.config.alpha if alpha is None else alpha
        terms = self.log_probs(content, style)
        return terms["style_given_content"] + alpha * terms["content"]


def aggregate_scores(
    scores: torch.Tensor, method: str = "min", percentile: float = 5.0
) -> torch.Tensor:
    """Song-level score from per-window scores (1-D tensor) -> scalar.

    Which of minimum / low percentile works better is a hyperparameter
    (CLAUDE.md "Windowing").
    """
    if method == "min":
        return scores.min()
    if method == "percentile":
        return torch.quantile(scores, percentile / 100.0)
    raise ValueError(f"unknown aggregation method: {method!r}")
