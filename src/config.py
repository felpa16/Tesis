"""Configuration for the representation-learning stage.

All tensor dimensions and hyperparameters live here (see CLAUDE.md coding
guidelines). Configs are plain nested dataclasses; `load_config` reads an
optional JSON file whose nested keys override the defaults, and the training
script layers CLI overrides on top of that.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MertConfig:
    model_name: str = "m-a-p/MERT-v1-330M"
    n_hidden_states: int = 25  # 24 layers + embedding layer
    dim: int = 1024
    sample_rate: int = 24000
    frame_rate: float = 75.0
    micro_batch: int = 4  # windows per MERT forward (memory control)


@dataclass
class EncoderConfig:
    d_model: int = 1024  # must equal MertConfig.dim: (N,1024) -> (N,1024)
    n_layers: int = 4
    n_heads: int = 8
    ffn_ratio: int = 4
    dropout: float = 0.0
    share_encoder: bool = False  # ablation: one Transformer for both branches


@dataclass
class BottleneckConfig:
    n_tokens: int = 8
    token_dim: int = 256
    n_heads: int = 8


@dataclass
class DecoderConfig:
    d_model: int = 512
    n_layers: int = 4
    n_heads: int = 8
    ffn_ratio: int = 4


@dataclass
class LossConfig:
    recon_weight: float = 1.0  # 2a plain reconstruction (main weight)
    contrastive_weight: float = 1.0  # content contrastive (MIL-NCE)
    swap_weight: float = 0.25  # 2b cover-swap reconstruction (low weight)
    cycle_weight: float = 0.25  # 2c latent cycle-consistency
    decorrelation_weight: float = 0.1
    temperature: float = 0.1  # contrastive temperature
    cosine_weight: float = 0.0  # optional cosine term in reconstruction
    cycle_fraction: float = 0.5  # fraction of the batch used for 2c
    decorrelation: str = "xcorr"  # "xcorr" | "hsic"


@dataclass
class NsfConfig:
    """One neural spline flow: ActNorm -> LU linear -> RQS coupling per step."""

    n_transforms: int = 10  # flow steps (CLAUDE.md: 8-12 for the conditional flow)
    n_bins: int = 8  # rational-quadratic spline bins
    hidden_features: tuple[int, ...] = (512, 512)  # coupling conditioner MLP widths


@dataclass
class FlowConfig:
    context_dim: int = 512  # pooled content context vector c
    pool_heads: int = 8  # heads in the 1-query attention pooling
    style: NsfConfig = field(default_factory=NsfConfig)  # p(style | content)
    content: NsfConfig = field(  # small unconditional p(content)
        default_factory=lambda: NsfConfig(n_transforms=6, hidden_features=(256, 256))
    )
    alpha: float = 1.0  # score = log p(style | content) + alpha * log p(content)
    whiten_windows: int = 4096  # train windows used to fit fixed whitening stats


@dataclass
class DataConfig:
    data_root: str = ""  # empty -> shs100k_meta.DEFAULT_DATA_ROOT
    train_split: str = "train"
    val_split: str = "val"  # "none" disables validation
    window_seconds: float = 20.0
    batch_pairs: int = 8  # aligned pairs per step
    batch_tracks: int = 8  # extra plain-reconstruction windows per step
    n_candidates: int = 1  # B-windows per pair (MIL-NCE when > 1)
    num_workers: int = 2
    val_max_batches: int = 0  # cap validation batches; 0 = full pass


@dataclass
class OptimConfig:
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1  # cosine decay floor as a fraction of lr
    grad_clip: float = 1.0


@dataclass
class TrainConfig:
    mert: MertConfig = field(default_factory=MertConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    bottleneck: BottleneckConfig = field(default_factory=BottleneckConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    epochs: int = 10
    max_steps: int = 0  # 0 = no cap
    device: str = "auto"  # auto | cuda | mps | cpu
    seed: int = 0
    log_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 500  # steps; 0 = only at epoch end
    log_every: int = 10  # steps
    freeze_layer_weights: bool = False  # phase-1 freeze switch
    standardizer_momentum: float = 0.99

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _update_dataclass(obj, data: dict):
    for key, value in data.items():
        if not hasattr(obj, key):
            raise KeyError(f"unknown config key: {key!r}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: Path | None = None) -> TrainConfig:
    """Default config, optionally overridden by a nested JSON file."""
    config = TrainConfig()
    if path is not None:
        with open(path, encoding="utf-8") as f:
            _update_dataclass(config, json.load(f))
    return config


def config_from_dict(data: dict) -> TrainConfig:
    """Rebuild a config from a checkpoint's saved `to_dict()` output."""
    return _update_dataclass(TrainConfig(), data)
