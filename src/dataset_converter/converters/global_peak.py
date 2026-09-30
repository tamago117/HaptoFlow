"""Dataset-wide peak amplitude scanner.

Used by ``BaseConverter.run()`` when ``preprocessing.global_peak_normalize``
is set — we sweep every input file once to find the global max |amp|, then
normalize each waveform by it during the main pass.

The scan applies the *same* trim / lowpass / resampling pipeline as the main
pass so the measured peak matches what will be normalized. Parallel execution
is supported.
"""
from __future__ import annotations

from multiprocessing import Pool, cpu_count
from typing import List

import numpy as np
from tqdm import tqdm

from src.dataset_converter.waveform import (
    estimate_sample_rate,
    resample_waveform,
    trim_waveform,
)


def _percentile_or_max(per_stream_peaks: List[float], ignore_outliers: bool, percentile: float) -> float:
    """Reduce per-stream peaks to a single value, optionally clipping by percentile."""
    if not per_stream_peaks:
        return 0.0
    if not ignore_outliers:
        return float(max(per_stream_peaks))

    try:
        p = float(percentile)
    except (TypeError, ValueError):
        p = 99.9
    if p <= 0:
        return 0.0
    if p > 100:
        p = 100.0

    try:
        return float(np.percentile(per_stream_peaks, p))
    except Exception as e:
        print(f"Warning: Failed to compute percentile-based global peak ({e}); falling back to max.")
        return float(max(per_stream_peaks))


def _scan_streams_for_peaks(converter, inputs: List[str]) -> List[float]:
    """Sequential scan: read each file, apply the prep pipeline, record max |amp|."""
    per_stream_peaks: List[float] = []
    desc = f"Scanning global peak ({converter.__class__.__name__})"
    for file_path in tqdm(inputs, desc=desc, unit="file"):
        try:
            streams = converter.read_input(file_path)
        except Exception as e:
            print(f"Warning: Peak scan skipped {file_path} - {e}")
            continue

        if not streams:
            continue

        for waveform, timestamps, metadata, suffix in streams:
            wav = waveform
            ts = timestamps
            meta = metadata

            current_sample_rate = converter.sample_rate
            original_sr = None
            try:
                original_sr = estimate_sample_rate(ts)
                current_sample_rate = original_sr
            except Exception:
                pass

            # Lowpass before resampling (use original sample rate)
            if converter.lowpass_enabled:
                try:
                    wav = converter._apply_lowpass(
                        wav,
                        sample_rate=original_sr if original_sr is not None else converter.sample_rate,
                    )
                except Exception as e:
                    print(f"Warning: Peak scan lowpass failed on {file_path}{suffix}: {e}")
                    continue

            if converter.target_sample_rate is not None and original_sr is not None:
                try:
                    if abs(original_sr - converter.target_sample_rate) > 1e-6:
                        wav, ts, meta = resample_waveform(
                            wav, ts, original_sr, converter.target_sample_rate, meta
                        )
                        current_sample_rate = converter.target_sample_rate
                except Exception as e:
                    print(f"Warning: Peak scan resampling failed for {file_path}{suffix}: {e}")
                    # Continue with original sample rate

            if converter.trim_start > 0 or converter.trim_end > 0:
                try:
                    wav, ts, meta = trim_waveform(
                        wav, ts, current_sample_rate,
                        converter.trim_start, converter.trim_end, meta,
                    )
                except ValueError as e:
                    print(f"Warning: Peak scan skip {file_path}{suffix} - {e}")
                    continue

            if wav.size == 0:
                continue

            try:
                peak = float(np.max(np.abs(wav)))
            except Exception as e:
                print(f"Warning: Peak scan failed on {file_path}{suffix}: {e}")
                continue

            per_stream_peaks.append(peak)
    return per_stream_peaks


def compute_global_peak(converter, inputs: List[str]) -> float:
    """Scan ``inputs`` for the global max |amplitude| under ``converter``'s pipeline.

    Branches on ``cfg['parallel']`` for multi-worker scanning; the worker
    function lives in ``src.dataset_converter.parallel`` and accepts the
    ``(file_path, cfg, converter_class_name)`` triple.
    """
    if not inputs:
        return 0.0

    parallel_cfg = converter.cfg.get("parallel", {})
    num_workers = parallel_cfg.get("num_workers") or cpu_count()
    use_parallel = parallel_cfg.get("enabled", True) and len(inputs) > 1

    if use_parallel and num_workers > 1:
        # Imported here to avoid a circular import (parallel.py imports the concrete converters)
        from src.dataset_converter.parallel import _compute_file_peak_parallel

        converter_class_name = converter.__class__.__name__
        args_list = [(file_path, converter.cfg, converter_class_name) for file_path in inputs]
        desc = (
            f"Scanning global peak ({converter.__class__.__name__}) "
            f"[Parallel: {num_workers} workers]"
        )
        per_stream_peaks: List[float] = []
        with Pool(processes=num_workers) as pool:
            for file_peaks in tqdm(
                pool.imap(_compute_file_peak_parallel, args_list),
                total=len(inputs),
                desc=desc,
                unit="file",
            ):
                per_stream_peaks.extend(file_peaks)
    else:
        per_stream_peaks = _scan_streams_for_peaks(converter, inputs)

    return _percentile_or_max(
        per_stream_peaks,
        ignore_outliers=converter.global_peak_ignore_outliers,
        percentile=converter.global_peak_outlier_percentile,
    )
