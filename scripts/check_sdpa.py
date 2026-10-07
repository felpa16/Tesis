#!/usr/bin/env python3
"""Which SDPA backend does MultiHeadAttention actually get?

`F.scaled_dot_product_attention` (src/models/attention.py:89) picks a backend
itself, silently, per call. Flash and the memory-efficient kernel never
materialize the (B, H, N, N) score matrix; the MATH fallback does, and at
N = 1500 that is ~34 MB per window per layer. Whether the fast kernels are
actually being reached therefore decides a multi-GiB slice of the memory
budget, which is what caps --batch-pairs.

This asks each candidate backend whether it accepts the project's real shapes,
times forward+backward on each, and then reports what the dispatcher chooses
when left alone. Compare that last line against the others:

  DEFAULT ~= FLASH   attention is already free; the batch ceiling is elsewhere
  DEFAULT ~= MATH    the fast kernels are not being reached -- fix that first

Needs CUDA: the fast kernels do not exist on CPU or MPS, so running this
locally proves nothing.

Example:
    python scripts/check_sdpa.py
    python scripts/check_sdpa.py --batch 32   # how far can the batch go?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from src.config import load_config  # noqa: E402
from src.models.attention import MultiHeadAttention  # noqa: E402

BACKENDS = {
    SDPBackend.FLASH_ATTENTION: "FLASH_ATTENTION",
    SDPBackend.EFFICIENT_ATTENTION: "EFFICIENT_ATTENTION",
    SDPBackend.CUDNN_ATTENTION: "CUDNN_ATTENTION",
    SDPBackend.MATH: "MATH (materializes B,H,N,N)",
}


def time_backward(layer, x, iterations: int = 10) -> tuple[float, float]:
    """Mean ms per forward+backward, and peak allocated GiB."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        layer(x).sum().backward()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        layer(x).sum().backward()
    end.record()
    torch.cuda.synchronize()
    return (
        start.elapsed_time(end) / iterations,
        torch.cuda.max_memory_allocated() / 1024**3,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="windows per step (default: the config's 2*batch_pairs + "
        "batch_tracks, i.e. what one training step really pushes)",
    )
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA; the fast kernels do not exist on CPU/MPS")

    config = load_config(args.config)
    batch = args.batch or config.data.batch_pairs * 2 + config.data.batch_tracks
    frames = int(config.data.window_seconds * config.mert.frame_rate)
    d_model, heads = config.encoder.d_model, config.encoder.n_heads

    print(f"device {torch.cuda.get_device_name()}, torch {torch.__version__}")
    print(
        f"shapes: {batch} windows x {frames} frames, d_model {d_model}, "
        f"{heads} heads, head_dim {d_model // heads}"
    )
    print(
        f"score matrix if materialized: "
        f"{frames * frames * heads * 2 / 1024**2:.0f} MiB per window per layer\n"
    )

    layer = MultiHeadAttention(d_model, heads, use_rope=True)
    layer = layer.cuda().to(torch.bfloat16)
    x = torch.randn(
        batch, frames, d_model, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )

    print("one attention layer, forward + backward:")
    for backend, name in BACKENDS.items():
        try:
            with sdpa_kernel([backend]):
                milliseconds, peak = time_backward(layer, x)
            print(f"  {name:28s} {milliseconds:7.1f} ms   peak {peak:5.2f} GiB")
        except RuntimeError as error:
            reason = str(error).strip().splitlines()[0][:72]
            print(f"  {name:28s} refused: {reason}")

    milliseconds, peak = time_backward(layer, x)
    print(f"\n  {'DEFAULT (what training gets)':28s} {milliseconds:7.1f} ms "
          f"  peak {peak:5.2f} GiB")


if __name__ == "__main__":
    main()
