"""Config loader with `_base:` deep-merge support."""
from __future__ import annotations

import os
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(path: str) -> DictConfig:
    """Load a YAML config, deep-merging on top of an optional `_base:` chain.

    If the YAML contains a top-level `_base: <relative_path>` key, that file
    is loaded first (relative to the directory of `path`, recursively, so a
    base may itself have a `_base:`) and the main YAML is merged on top of it.
    The `_base` key is stripped from the result. A cycle raises ValueError.
    """
    return OmegaConf.create(_load_raw(path, ()))


def _load_raw(path: str, seen: tuple) -> dict:
    path = os.path.abspath(path)
    if path in seen:
        chain = " -> ".join(seen + (path,))
        raise ValueError(f"Cyclic _base chain: {chain}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base_ref = raw.pop("_base", None)
    if not base_ref:
        return raw

    base_path = os.path.join(os.path.dirname(path), str(base_ref))
    base_raw = _load_raw(base_path, seen + (path,))
    merged = OmegaConf.merge(OmegaConf.create(base_raw), OmegaConf.create(raw))
    return OmegaConf.to_container(merged)


def _strip_unused_top_keys(cfg: Any) -> Any:
    """Remove the `defaults` key from a loaded cfg (no-op if absent)."""
    if hasattr(cfg, "defaults"):
        del cfg.defaults
    return cfg
