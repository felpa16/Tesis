#!/usr/bin/env python3
"""Smoke test of the downstream flow-detection stack (no MERT, no data).

Random-tensor checks with a small flow config at the real latent shape
(8x256 -> 2048): whitening fit + fail-loud guard, conditional/marginal flow
log_probs (shape, finiteness, real conditioning), backward through pooler and
flows, factorized score decomposition, sampling/invertibility, song-level
aggregation, the conditional-Gaussian baseline, and an end-to-end pass
through a frozen DisentanglementModel.

Example:
    python scripts/smoke_test_flow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from src.config import NsfConfig, TrainConfig  # noqa: E402
from src.models import (  # noqa: E402
    ConditionalGaussian,
    DisentanglementModel,
    FlowDetector,
    aggregate_scores,
)


def small_flow_config(config: TrainConfig) -> None:
    """Shrink the flows so the test runs fast on CPU; latent dims stay real."""
    config.flow.style = NsfConfig(n_transforms=2, n_bins=4, hidden_features=(32, 32))
    config.flow.content = NsfConfig(n_transforms=2, n_bins=4, hidden_features=(32, 32))
    config.flow.context_dim = 64


def check_detector(config: TrainConfig) -> FlowDetector:
    torch.manual_seed(0)
    bn = config.bottleneck
    detector = FlowDetector(bn, config.flow)
    baseline = ConditionalGaussian(bn, config.flow)
    n_params = sum(p.numel() for p in detector.parameters())
    print(f"detector params: {n_params / 1e6:.1f}M")

    batch = 8
    content = torch.randn(batch, bn.n_tokens, bn.token_dim)
    style = torch.randn(batch, bn.n_tokens, bn.token_dim)

    # fail-loud guard: log_prob before whitening fit must raise
    try:
        detector.style_flow.log_prob(content, style)
        raise AssertionError("log_prob before Whitener.fit() should raise")
    except RuntimeError:
        pass
    fit_content = torch.randn(32, bn.n_tokens, bn.token_dim)
    fit_style = torch.randn(32, bn.n_tokens, bn.token_dim)
    detector.fit_whitening(fit_content, fit_style)
    baseline.whitener.fit(fit_style.flatten(1))
    print("whitening ok (unfitted guard raises, fit succeeds)")

    terms = detector.log_probs(content, style)
    for name, value in terms.items():
        assert value.shape == (batch,), (name, value.shape)
        assert torch.isfinite(value).all(), name
    lp_gauss = baseline.log_prob(content, style)
    assert lp_gauss.shape == (batch,) and torch.isfinite(lp_gauss).all()
    print("log_probs ok: style_given_content, content, gaussian baseline")

    # conditioning is real: permuting the content must change p(style | content)
    permuted = detector.style_flow.log_prob(content.roll(1, dims=0), style)
    assert not torch.allclose(terms["style_given_content"], permuted)
    print("conditioning ok (log_prob depends on content)")

    # factorized score decomposition at two alphas
    for alpha in (0.0, 0.7):
        score = detector.score(content, style, alpha=alpha)
        expected = terms["style_given_content"] + alpha * terms["content"]
        assert torch.allclose(score, expected, atol=1e-5), alpha
    assert torch.allclose(
        detector.score(content, style),
        terms["style_given_content"] + config.flow.alpha * terms["content"],
        atol=1e-5,
    )
    print("factorized score ok (alpha=0, 0.7, config default)")

    loss = -(terms["style_given_content"].mean() + terms["content"].mean())
    loss = loss - lp_gauss.mean()
    loss.backward()
    assert detector.style_flow.pooler.query.grad is not None
    n_grads = sum(
        p.grad is not None for p in detector.parameters() if p.requires_grad
    )
    n_trainable = sum(1 for p in detector.parameters() if p.requires_grad)
    assert n_grads == n_trainable, f"{n_grads}/{n_trainable} params got grads"
    assert baseline.pooler.query.grad is not None
    print(f"backward ok ({n_grads}/{n_trainable} detector params got grads)")

    with torch.no_grad():
        context = detector.style_flow.pooler(content)
        sample = detector.style_flow.flow(context).sample()
        assert sample.shape == (batch, bn.n_tokens * bn.token_dim)
        assert torch.isfinite(sample).all()
        lp_sample = detector.style_flow.flow(context).log_prob(sample)
        assert torch.isfinite(lp_sample).all()
    print("sampling ok (inverse pass finite, round-trip log_prob finite)")

    scores = detector.score(content, style).detach()
    assert torch.equal(aggregate_scores(scores, "min"), scores.min())
    q5 = aggregate_scores(scores, "percentile", 5.0)
    assert torch.isfinite(q5) and q5 <= scores.median()
    print("aggregation ok (min and low percentile)")
    return detector


@torch.no_grad()
def check_end_to_end(config: TrainConfig, detector: FlowDetector) -> None:
    torch.manual_seed(1)
    encoder = DisentanglementModel(config)
    encoder.requires_grad_(False)
    encoder.eval()
    hidden = torch.randn(2, config.mert.n_hidden_states, 60, config.mert.dim)
    content, style, _, _ = encoder.encode(hidden)
    score = detector.score(content, style)
    assert score.shape == (2,) and torch.isfinite(score).all()
    print(f"end-to-end ok: frozen encoder -> detector score {score.tolist()}")


def main() -> None:
    config = TrainConfig()
    small_flow_config(config)
    detector = check_detector(config)
    check_end_to_end(config, detector)
    print("PASS")


if __name__ == "__main__":
    main()
