#!/usr/bin/env python3
"""Smoke test of the representation-learning model stack.

Default (fast, no MERT): random-tensor forwards through every module with
shape asserts, all losses computed and backpropagated, parameter counts
checked (decoder must be smaller than one encoder branch).

--with-mert additionally decodes two real val windows, runs frozen MERT, and
backprops the full objective — verifying the HF integration end to end.

Examples:
    python scripts/smoke_test_model.py
    python scripts/smoke_test_model.py --with-mert
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from shs100k_meta import DEFAULT_DATA_ROOT  # noqa: E402
from src.config import TrainConfig  # noqa: E402
from src.data import WindowConfig, read_pairs, read_tracks  # noqa: E402
from src.data.windows import decode_window  # noqa: E402
from src.models import DisentanglementModel, MertExtractor  # noqa: E402
from src.training import (  # noqa: E402
    compute_losses,
    extract_mixes,
    pooled_targets,
)


def count_parameters(*modules: torch.nn.Module) -> int:
    return sum(p.numel() for module in modules for p in module.parameters())


def check_shapes_and_losses(config: TrainConfig) -> None:
    torch.manual_seed(0)
    model = DisentanglementModel(config)
    bn = config.bottleneck
    batch, n_frames = 4, 90  # 2 pairs (P=2, K=1) -> layout [a, a, b, b]
    p, k = 2, 1

    hidden = torch.randn(
        batch, config.mert.n_hidden_states, n_frames, config.mert.dim
    )
    content, style, content_mix, style_mix = model.encode(hidden)
    assert content.shape == (batch, bn.n_tokens, bn.token_dim), content.shape
    assert style.shape == (batch, bn.n_tokens, bn.token_dim), style.shape
    assert content_mix.shape == (batch, n_frames, config.mert.dim)
    print(f"encode ok: content/style {tuple(content.shape)}")

    pred_content, pred_style = model.decode(content, style, n_frames)
    assert pred_content.shape == (batch, n_frames, config.mert.dim)
    assert pred_style.shape == (batch, n_frames, config.mert.dim)
    short_content, _ = model.decode(content, style, 60)
    assert short_content.shape == (batch, 60, config.mert.dim)
    print("decode ok: matched-length and cross-length reconstruction")

    content_target, style_target = pooled_targets(config.loss, content_mix, style_mix)
    expected = n_frames // max(config.loss.recon_pool, 1)
    assert content_target.shape == (batch, expected, config.mert.dim)
    print(f"pooled target ok: {n_frames} -> {expected} frames")

    model.content_std.update(content_target)
    model.style_std.update(style_target)
    total, losses = compute_losses(
        model, config.loss, content, style, content_target, style_target, p, k
    )
    for name, value in losses.items():
        assert torch.isfinite(value), f"{name} not finite"
    assert len(losses) == 5, f"expected 5 loss terms, got {list(losses)}"
    total.backward()
    assert model.content_mix.weights.grad is not None
    assert model.style_mix.weights.grad is not None
    parts = "  ".join(f"{n}={float(v):.4f}" for n, v in losses.items())
    print(f"losses ok (backward ran, layer-mix grads present): {parts}")

    encoder_params = count_parameters(model.content_encoder, model.content_bottleneck)
    decoder_params = count_parameters(model.decoder)
    total_params = count_parameters(model)
    print(
        f"params: encoder branch {encoder_params / 1e6:.1f}M, "
        f"decoder {decoder_params / 1e6:.1f}M, total {total_params / 1e6:.1f}M"
    )
    assert decoder_params < encoder_params, "decoder must be smaller than encoders"


def check_with_mert(config: TrainConfig, data_root: Path, split: str) -> None:
    tracks = read_tracks(data_root, split)
    pairs = read_pairs(data_root, split)
    by_key = {t.key: t for t in tracks}
    pair = pairs[0]
    window = WindowConfig(window_seconds=5.0, sample_rate=config.mert.sample_rate)
    waves = torch.stack(
        [
            decode_window(data_root / by_key[pair.key_a].audio, 30.0, window),
            decode_window(data_root / by_key[pair.key_b].audio, 30.0, window),
        ]
    )
    print(f"decoded {pair.key_a} + {pair.key_b}: waves {tuple(waves.shape)}")

    t0 = time.perf_counter()
    mert = MertExtractor(config.mert)
    model = DisentanglementModel(config)
    print(f"models loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    content_mix, style_mix = extract_mixes(mert, model, waves, micro_batch=2)
    expected_frames = content_mix.shape[1]
    assert content_mix.shape == (2, expected_frames, config.mert.dim)
    print(
        f"MERT + mixes ok in {time.perf_counter() - t0:.1f}s: "
        f"{tuple(content_mix.shape)} (~{expected_frames / 5.0:.0f} frames/s)"
    )

    content_target, style_target = pooled_targets(config.loss, content_mix, style_mix)
    model.content_std.update(content_target)
    model.style_std.update(style_target)
    content, style = model.encode_mixes(content_mix, style_mix)
    total, losses = compute_losses(
        model, config.loss, content, style, content_target, style_target, 1, 1
    )
    assert torch.isfinite(total)
    total.backward()
    assert model.content_mix.weights.grad is not None
    parts = "  ".join(f"{n}={float(v):.4f}" for n, v in losses.items())
    print(f"full pipeline ok (P=1, K=1): {parts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-mert", action="store_true")
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    config = TrainConfig()
    check_shapes_and_losses(config)
    if args.with_mert:
        check_with_mert(config, args.data_root, args.split)
    print("PASS")


if __name__ == "__main__":
    main()
