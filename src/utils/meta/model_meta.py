from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf

from src.utils.hf_checkpoint import resolve_model_dir


def load_model_meta(model_dir: str) -> Tuple[Dict[str, Any], OmegaConf, int, str]:
    """
    Load meta.json from the model directory, including the training config and
    waveform_length (samples per segment).

    Returns: (meta, cfg, waveform_length, meta_path)
    """
    model_dir = resolve_model_dir(model_dir)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in model directory: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta: Dict[str, Any] = json.load(f)

    if "config" not in meta:
        raise KeyError(f'"config" not found in meta.json: {meta_path}')

    cfg_dict = meta["config"]
    cfg = OmegaConf.create(cfg_dict)

    if "waveform_length" not in meta:
        raise KeyError(
            f'"waveform_length" not found in meta.json: {meta_path}. '
            "Please ensure meta.json contains waveform_length."
        )

    waveform_length = int(meta["waveform_length"])
    return meta, cfg, waveform_length, meta_path


def require_meta(meta: Dict[str, Any], meta_path: str, keys: List[str]) -> None:
    """
    Check that a nested key path exists, e.g. keys = ['dataset', 'sample_rate'].
    """
    cur: Any = meta
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            dotted = ".".join(keys)
            raise KeyError(
                f'"{dotted}" not found in meta.json: {meta_path}. '
                "Please re-train or re-save meta.json with scripts/train.py (newer version)."
            )
        cur = cur[k]


def extract_dataset_meta(
    meta: Dict[str, Any],
    meta_path: str,
) -> Tuple[List[str], Dict[str, int], int, Optional[int]]:
    """
    Extract dataset metadata.

    Returns: (classes, label_to_int, sample_rate, num_classes or None)
    """
    require_meta(meta, meta_path, ["dataset", "sample_rate"])
    ds = meta.get("dataset", {}) if isinstance(meta.get("dataset"), dict) else {}
    classes = ds.get("classes") or []
    label_to_int = ds.get("label_to_int") or {}
    sample_rate = int(float(ds["sample_rate"]))
    num_classes: Optional[int] = None
    if "num_classes" in ds:
        num_classes = int(ds["num_classes"])
    return list(classes), {str(k): int(v) for k, v in dict(label_to_int).items()}, sample_rate, num_classes


def extract_control_stats(meta: Dict[str, Any], meta_path: str) -> Dict[str, Dict[str, float]]:
    """
    Extract mean/std/min/max from control_stats as floats.
    """
    require_meta(meta, meta_path, ["control_stats"])
    cs = meta["control_stats"]
    out: Dict[str, Dict[str, float]] = {}
    for k in ("force", "velocity_x", "velocity_y"):
        if k not in cs:
            continue
        d = cs[k]
        row: Dict[str, float] = {
            "mean": float(d["mean"]),
            "std": float(d["std"]),
            "min": float(d["min"]),
            "max": float(d["max"]),
        }
        out[k] = row
    return out

