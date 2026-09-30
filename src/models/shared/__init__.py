"""Shared model building blocks used across model families.

Re-exports the common embedding / encoder modules so callers can write
``from src.models.shared import LabelEmbedding`` rather than digging into
the per-file submodule path.
"""
from src.models.shared.embeddings import (
    ControlEmbedding,
    LabelEmbedding,
    SinusoidalTimeEmbedding,
)
from src.models.shared.encodec_wrapper import EncodecWrapper
from src.models.shared.prev_segment_encoder import PrevSegmentEncoder

__all__ = [
    "ControlEmbedding",
    "LabelEmbedding",
    "SinusoidalTimeEmbedding",
    "EncodecWrapper",
    "PrevSegmentEncoder",
]
