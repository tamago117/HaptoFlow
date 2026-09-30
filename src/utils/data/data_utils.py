"""Data preparation utilities for training."""
import os
import signal
import json
from typing import Tuple, Dict, Any, Optional, Iterable
from collections import defaultdict
from math import sqrt
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.datasets.haptic_waveform_dataset import HapticWaveformDataset, split_windows


def prepare_datasets(
    cfg: DictConfig,
    context_segments: int = 1,
    control_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset, HapticWaveformDataset]:
    """Build the window dataset and split it into train / val / test.

    Args:
        cfg: Configuration object (``data.root``, ``data.meta``, ``data.window_hop``, ``data.split``).
        context_segments: Back-to-back segments returned before each target window.
        control_stats: Control normalization stats from stats.jsonl.

    Returns:
        Tuple of (train_ds, val_ds, test_ds, full_ds).
    """
    full_ds = HapticWaveformDataset(
        meta_path=cfg.data.meta,
        root_dir=cfg.data.root,
        window_hop=float(cfg.data.window_hop),
        context_segments=context_segments,
        control_stats=control_stats,
    )
    train_ds, val_ds, test_ds = split_windows(full_ds, list(cfg.data.split))
    return train_ds, val_ds, test_ds, full_ds


def _ignore_sigint_in_worker(worker_id: int) -> None:
    """Make DataLoader workers ignore SIGINT.

    A terminal Ctrl+C is delivered to the whole foreground process group, so the
    workers would die alongside the trainer and the main process's next
    ``next(iterator)`` would raise "DataLoader worker exited unexpectedly"
    *before* the trainer reaches a safe point and can save. Ignoring SIGINT here
    leaves shutdown to the parent. SIGTERM is deliberately left at its default:
    DataLoader teardown and the memory guard's hard tier both rely on it.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def create_dataloaders(
    train_ds: HapticWaveformDataset,
    val_ds: HapticWaveformDataset,
    test_ds: HapticWaveformDataset,
    cfg: DictConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for training, validation, and testing.
    
    Args:
        train_ds: Training dataset.
        val_ds: Validation dataset.
        test_ds: Test dataset.
        cfg: Configuration object.
    
    Returns:
        Tuple of (train_dl, val_dl, test_dl).
    """
    num_workers = int(cfg.data.num_workers)
    pin_memory = bool(cfg.data.pin_memory)

    # PyTorch requires num_workers > 0 for persistent workers
    persistent_workers = num_workers > 0

    prefetch_factor = getattr(cfg.data, "prefetch_factor", None)
    dl_common_kwargs = {
        "batch_size": int(cfg.data.batch_size),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        dl_common_kwargs["worker_init_fn"] = _ignore_sigint_in_worker
    if prefetch_factor is not None and num_workers > 0:
        dl_common_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        **dl_common_kwargs,
    )
    val_dl = DataLoader(
        val_ds,
        shuffle=False,
        **dl_common_kwargs,
    )
    test_dl = DataLoader(
        test_ds,
        shuffle=False,
        **dl_common_kwargs,
    )
    
    return train_dl, val_dl, test_dl


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_column_stats_from_jsonl(
    stats_path: str,
    target_columns: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-column statistics from stats.jsonl.

    Each line holds the stats of one file (columns.{col}.min/max/mean, etc.).
    Means are weighted by num_rows. Synthetic windows (silence padding) are
    skipped since they would skew the real-data distribution. When std is not
    recorded, (max - min) / 6 is used as an approximation.
    """
    stats_path = str(stats_path)
    if not os.path.exists(stats_path):
        return {}
    
    target_set = set(target_columns) if target_columns is not None else None
    
    accum: Dict[str, Dict[str, float]] = defaultdict(lambda: {"sum": 0.0, "sumsq": 0.0, "count": 0.0, "min": None, "max": None})
    has_std_any = False
    
    with open(stats_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            if rec.get("synthetic"):
                continue
            num_rows = _safe_float(rec.get("num_rows"))
            if num_rows is None or num_rows <= 0:
                continue
            
            cols = rec.get("columns", {})
            for col_name, col_stats in cols.items():
                if target_set is not None and col_name not in target_set:
                    continue
                
                mean_v = _safe_float(col_stats.get("mean"))
                min_v = _safe_float(col_stats.get("min"))
                max_v = _safe_float(col_stats.get("max"))
                std_v = _safe_float(col_stats.get("std"))
                
                if mean_v is None:
                    continue
                
                acc = accum[col_name]
                acc["sum"] += mean_v * num_rows
                acc["count"] += num_rows
                
                if min_v is not None:
                    acc["min"] = min_v if acc["min"] is None else min(acc["min"], min_v)
                if max_v is not None:
                    acc["max"] = max_v if acc["max"] is None else max(acc["max"], max_v)
                
                # Weighted E[x^2] for the pooled variance
                if std_v is not None:
                    has_std_any = True
                    acc["sumsq"] += ((std_v ** 2) + (mean_v ** 2)) * num_rows
    
    result: Dict[str, Dict[str, float]] = {}
    for col, acc in accum.items():
        if acc["count"] <= 0:
            continue
        mean = acc["sum"] / acc["count"]
        
        std: float
        if has_std_any and acc["sumsq"] > 0:
            var = acc["sumsq"] / acc["count"] - mean ** 2
            if var < 0:
                var = 0.0
            std = sqrt(var)
        else:
            # No std recorded: approximate with (max - min) / 6, falling back to 1
            span = 0.0
            if acc["min"] is not None and acc["max"] is not None:
                span = float(acc["max"]) - float(acc["min"])
            std = span / 6.0 if span > 0 else 1.0
            if std == 0.0:
                std = 1.0
        
        result[col] = {
            "mean": float(mean),
            "std": float(std),
        }
        if acc["min"] is not None:
            result[col]["min"] = float(acc["min"])
        if acc["max"] is not None:
            result[col]["max"] = float(acc["max"])
    
    return result

