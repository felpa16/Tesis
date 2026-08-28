"""Model components.

The flow names are imported lazily: src.models.flow depends on zuko, which is
only needed for the phase-3 detector. Importing it eagerly would make phase-1
representation learning fail on a box that has no flow library installed.
"""

from typing import Any

from src.models.attention import MultiHeadAttention, RotaryEmbedding
from src.models.bottleneck import CrossAttentionBottleneck
from src.models.decoder import PerceiverDecoder
from src.models.mert import LayerMix, MertExtractor
from src.models.model import DisentanglementModel, Standardizer
from src.models.transformer import TransformerEncoder

__all__ = [
    "ConditionalGaussian",
    "ConditionalStyleFlow",
    "ContentPooler",
    "CrossAttentionBottleneck",
    "DisentanglementModel",
    "FlowDetector",
    "LayerMix",
    "MarginalFlow",
    "MertExtractor",
    "MultiHeadAttention",
    "PerceiverDecoder",
    "RotaryEmbedding",
    "Standardizer",
    "TransformerEncoder",
    "Whitener",
    "aggregate_scores",
    "build_nsf",
]


_FLOW_NAMES = frozenset(
    {
        "ConditionalGaussian",
        "ConditionalStyleFlow",
        "ContentPooler",
        "FlowDetector",
        "MarginalFlow",
        "Whitener",
        "aggregate_scores",
        "build_nsf",
    }
)


def __getattr__(name: str) -> Any:
    if name in _FLOW_NAMES:
        from src.models import flow

        return getattr(flow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
