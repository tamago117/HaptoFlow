"""``MetadataOnlyDatasetView``: a dataset-shaped object built from ``meta.json`` alone.

Used by ``BaseInference.init_from_model_dir`` so that ``Strategy.build()`` can
construct the model graph at production-load time without actually reading the
dataset off disk. The view exposes only the attributes that ``Strategy.build``
is allowed to touch:

- ``classes``, ``label_to_int``, ``sample_rate``, ``waveform_target_len``

Anything beyond these is intentionally absent. If a future strategy starts
reading ``len(dataset)`` or per-sample arrays inside ``build()``, attribute
access fails loudly here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class MetadataOnlyDatasetView:
    """A minimal dataset surrogate populated from ``meta.json``."""
    classes: List[str] = field(default_factory=list)
    label_to_int: Dict[str, int] = field(default_factory=dict)
    sample_rate: int = 0
    waveform_target_len: int = 0

    @classmethod
    def from_meta(cls, meta: Dict[str, Any]) -> "MetadataOnlyDatasetView":
        """Build a view from the dict returned by ``load_meta_json`` / ``load_model_meta``."""
        ds = meta.get("dataset", {}) or {}
        classes = list(ds.get("classes", []) or [])
        label_to_int = dict(ds.get("label_to_int", {}) or {})
        if not label_to_int and classes:
            label_to_int = {str(c): i for i, c in enumerate(classes)}
        return cls(
            classes=classes,
            label_to_int=label_to_int,
            sample_rate=int(ds.get("sample_rate", 0) or 0),
            waveform_target_len=int(meta.get("waveform_length", 0) or 0),
        )
