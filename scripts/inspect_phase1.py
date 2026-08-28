#!/usr/bin/env python3
"""Phase-1 checkpoint audit: is the layer mix meaningful, and is `recon` real?

Three questions, answered before the layer weights are frozen for phase 2.

1. Did the layer-mix weights move away from their uniform initialisation, and
   did the two branches settle on *different* layers? A per-epoch cosine of
   1.000000 is also what you get when nothing ever moved, so stationarity
   alone does not justify freezing.

2. What is the trivial-predictor floor for `recon`? The target is per-dimension
   standardised and `recon` sums the content and style terms, so predicting the
   dataset mean already scores ~2.0. The raw loss only means something against
   that number.

3. How much standardised variance lives *between* windows (which 16 latent
   tokens could plausibly carry) versus *within* a window (frame detail that a
   ~750x compression cannot represent)? The gap between those two bounds is the
   only range the latents were ever able to compete in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from src.config import config_from_dict  # noqa: E402
from src.data import TrackWindowDataset, WindowConfig, read_tracks  # noqa: E402
from src.models import DisentanglementModel, MertExtractor  # noqa: E402
from src.training import extract_mixes, pick_device, worker_init  # noqa: E402


def report_layer_weights(weights: dict[str, torch.Tensor], n_states: int) -> None:
    uniform = torch.full((n_states,), 1.0 / n_states)
    print(f"\n{'=' * 72}\nLAYER MIX  ({n_states} MERT hidden states, 0 = embedding)\n{'=' * 72}")
    for branch, w in weights.items():
        entropy = -(w * w.clamp_min(1e-12).log()).sum()
        top = torch.topk(w, 5)
        print(f"\n[{branch}]")
        print(f"  entropy        {entropy:.4f} nats  (uniform = {torch.log(torch.tensor(float(n_states))):.4f})")
        print(f"  max weight     {w.max():.4f}  (uniform = {1.0 / n_states:.4f})")
        print(f"  cos to uniform {F.cosine_similarity(w, uniform, dim=0):.6f}")
        print(f"  top-5 layers   " + ", ".join(f"{i:02d}:{v:.4f}" for v, i in zip(top.values, top.indices)))
        for layer, value in enumerate(w.tolist()):
            bar = "#" * int(round(value / max(w.max().item(), 1e-9) * 50))
            print(f"    {layer:02d} {value:.4f} {bar}")
    cos = F.cosine_similarity(weights["content"], weights["style"], dim=0)
    print(f"\ncos(content, style) = {cos:.6f}")
    print("  -> ~1.0 means both branches chose the same layers: the two-stream")
    print("     premise (and caching two streams in phase 2) buys nothing.")


@torch.no_grad()
def measure_baselines(mert, model, loader, device, n_batches, micro_batch, autocast):
    """Compare the real recon against the two trivial predictors, same batches."""
    totals = {k: 0.0 for k in ("recon", "global_mean", "window_mean", "pred_std")}
    seen = 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        waves = batch["wave"].to(device)
        with autocast:
            content_mix, style_mix = extract_mixes(mert, model, waves, micro_batch)
            content, style = model.encode_mixes(content_mix, style_mix)
            pred_content, pred_style = model.decode(content, style, content_mix.shape[1])
        for pred, mix, std in (
            (pred_content, content_mix, model.content_std),
            (pred_style, style_mix, model.style_std),
        ):
            z = std.normalize(mix.float())
            totals["recon"] += F.mse_loss(pred.float(), z).item()
            # predict the dataset mean -> 0 in standardised space
            totals["global_mean"] += z.pow(2).mean().item()
            # predict each window's own mean vector (best per-window constant)
            totals["window_mean"] += (z - z.mean(dim=1, keepdim=True)).pow(2).mean().item()
            # how much does the decoder output actually move? ~0 = constant output
            totals["pred_std"] += pred.float().std().item()
        seen += 1
        print(f"  batch {i + 1}/{n_batches}", end="\r", flush=True)
    return {k: v / max(seen, 1) for k, v in totals.items()}, seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/last.pt"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights-only", action="store_true", help="skip the data pass")
    args = parser.parse_args()

    device = pick_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = config_from_dict(checkpoint["config"])
    model = DisentanglementModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"checkpoint: {args.checkpoint}  epoch={checkpoint['epoch']}  step={checkpoint['step']}")

    report_layer_weights(model.layer_weight_summary(), config.mert.n_hidden_states)

    var = model.content_std.var
    print(f"\nstandardiser (content): var min={var.min():.3e}  median={var.median():.3e}  max={var.max():.3e}")
    print("  -> a min many orders below the median means near-constant MERT dims")
    print("     get amplified into pure noise by standardisation.")

    if args.weights_only:
        return

    tracks = read_tracks(args.data_root, args.split)
    window = WindowConfig(
        window_seconds=config.data.window_seconds,
        sample_rate=config.mert.sample_rate,
        n_candidates=1,
    )
    loader = DataLoader(
        TrackWindowDataset(tracks, args.data_root, window),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init,
    )
    mert = MertExtractor(config.mert).to(device)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )

    print(f"\n{'=' * 72}\nRECON vs. TRIVIAL PREDICTORS ({args.batches} batches)\n{'=' * 72}")
    scores, seen = measure_baselines(
        mert, model, loader, device, args.batches, config.mert.micro_batch, autocast
    )
    ceiling, floor, actual = scores["global_mean"], scores["window_mean"], scores["recon"]
    print(f"\n  predict dataset mean      {ceiling:.4f}   <- learning nothing")
    print(f"  MODEL recon               {actual:.4f}")
    print(f"  predict per-window mean   {floor:.4f}   <- a single 1024-d vector per window")
    span = ceiling - floor
    if span > 1e-6:
        print(f"\n  variance explained vs. dataset mean : {100 * (ceiling - actual) / ceiling:5.1f}%")
        print(f"  fraction of the between-window range: {100 * (ceiling - actual) / span:5.1f}%")
        print("\n  100%+ of the range means the latents beat a per-window constant.")
        print("  ~0% means the decoder has not moved off the dataset mean.")
    print(f"\n  decoder output std        {scores['pred_std'] / 2:.4f}   (target std = 1.0)")
    print("  -> near 0 with cycle ~0 is the steganography failure mode: a tiny")
    print("     perturbation on the mean, carrying latents the encoder reads back.")


if __name__ == "__main__":
    main()
