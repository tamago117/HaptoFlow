"""``dataset_info.json`` builder used by ``BaseConverter.write_dataset_info``.

The schema produced here is the contract that downstream training/eval code
relies on (label list, raw signal format, segment length, peak-normalize stats,
etc.). Keeping the builder out of base.py lets the schema be
inspected and changed in one place.
"""
from __future__ import annotations

import json
from typing import Any, Dict


def build_dataset_info(converter) -> Dict[str, Any]:
    """Build the dataset_info dict for ``converter``.

    Reads attributes from the converter (set by ``BaseConverter.__init__`` /
    populated by ``run()``). Extras are merged from
    ``converter.get_dataset_info_extras()``.
    """
    import numpy as np

    info: Dict[str, Any] = {
        "output_dir": converter.output_dir,
        "source": converter.source,
        "group_by_label": converter.group_by_label,
        "segmentation": {
            "duration": converter.segment_duration,
            "segment_len": converter.segment_len,
        },
        "stats": {
            "num_recordings": len(converter.recording_lengths),
            "recording_length": {
                "min": int(min(converter.recording_lengths)) if converter.recording_lengths else 0,
                "max": int(max(converter.recording_lengths)) if converter.recording_lengths else 0,
                "mean": float(np.mean(converter.recording_lengths)) if converter.recording_lengths else 0,
                "total": int(sum(converter.recording_lengths)),
            },
        },
        "labels": sorted(converter.labels),
    }
    info["preprocessing"] = {
        "trim_start": converter.trim_start,
        "trim_end": converter.trim_end,
        "global_peak_normalize": converter.global_peak_normalize,
        "global_peak_value": (
            float(converter.global_peak_value) if converter.global_peak_value is not None else None
        ),
        "global_peak_epsilon": converter.global_peak_epsilon,
        "global_peak_ignore_outliers": converter.global_peak_ignore_outliers,
        "global_peak_outlier_percentile": converter.global_peak_outlier_percentile,
        "lowpass_enabled": converter.lowpass_enabled,
        "lowpass_cutoff_hz": (
            float(converter.lowpass_cutoff_hz) if converter.lowpass_cutoff_hz is not None else None
        ),
        "lowpass_order": converter.lowpass_order,
        "lowpass_type": converter.lowpass_type,
    }
    info["silence_padding"] = converter.silence_padding.info()
    info.update(converter.get_dataset_info_extras())
    return info


def write_dataset_info(converter, path: str) -> None:
    """Build the dataset_info dict and dump it to ``path`` as pretty JSON."""
    info = build_dataset_info(converter)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"Wrote dataset info to {path}")
