"""Shared helper: generate visualization samples by driving a strategy's inference engine.

Training-time visualization and standalone inference both go through
``XxxInference.generate_from_control(...)``.

Output:
    Tuple[List[Tensor[1,T]], List[Tensor[1,T]], List[Optional[Tensor[1,T]]], float]
    = (gen_waveforms, true_waveforms, prev_waveforms, mean_generate_time_ms)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _per_sample_control_raw(batch: Dict[str, torch.Tensor], idx: int) -> Tuple[float, float, float]:
    """Return raw (force, vx, vy) for one batch element.

    Prefers ``*_raw`` columns: the inference engine normalizes internally, so
    passing the normalized fields would silently double-normalize.
    """
    force = 1.0
    vx = 0.0
    vy = 0.0
    for key in ("force_mean_raw", "force_mean", "force"):
        t = batch.get(key)
        if t is not None:
            try:
                force = float(t[idx].view(-1)[0].item())
                break
            except Exception:
                pass
    for key in ("velocity_mean_raw", "velocity_mean"):
        t = batch.get(key)
        if t is not None:
            try:
                vals = t[idx].view(-1)
                if vals.numel() >= 2:
                    vx = float(vals[0].item())
                    vy = float(vals[1].item())
                    break
            except Exception:
                pass
    return force, vx, vy


def generate_samples_via_inference(
    strategy,
    bundle,
    ctx,
    batch: Dict[str, torch.Tensor],
    num_samples: int,
    render_prev_waveform: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Optional[torch.Tensor]], float]:
    """Build the strategy's inference engine and generate per-sample waveforms.

    Always uses raw control columns (``*_raw``) so the engine's internal
    ``_normalize_control_values`` runs exactly once. Each output tensor is
    ``[1, T]`` (channel-first).
    """
    iface = strategy.build_inference(bundle, ctx)
    waveforms = batch.get("waveform")
    if waveforms is None:
        return [], [], [], 0.0
    batch_size = int(waveforms.shape[0])
    n = min(int(num_samples), batch_size)

    gen_waveforms: List[torch.Tensor] = []
    true_waveforms: List[torch.Tensor] = []
    prev_waveforms: List[Optional[torch.Tensor]] = []
    total_ms = 0.0

    label_idx = batch["label_idx"]

    with torch.inference_mode():
        for i in range(n):
            label_val = int(label_idx[i].item())
            force, vx, vy = _per_sample_control_raw(batch, i)

            prev_wf_tensor: Optional[torch.Tensor] = None
            if "prev_waveform" in batch and batch["prev_waveform"] is not None:
                try:
                    prev_wf_tensor = batch["prev_waveform"][i]
                except Exception:
                    prev_wf_tensor = None

            t0 = time.time()
            samples, _sr, _dur = iface.generate_from_control(
                velocity_x=vx,
                velocity_y=vy,
                force=force,
                label=label_val,
                method=None,
                prev_waveform=prev_wf_tensor,
            )
            t1 = time.time()
            total_ms += (t1 - t0) * 1000.0

            pred_np = np.asarray(samples, dtype=np.float32)
            pred_t = torch.from_numpy(pred_np)
            if pred_t.dim() == 1:
                pred_t = pred_t.unsqueeze(0)
            gen_waveforms.append(pred_t)

            true_wf = waveforms[i].detach().cpu()
            if true_wf.dim() == 1:
                true_wf = true_wf.unsqueeze(0)
            true_waveforms.append(true_wf)

            if render_prev_waveform and prev_wf_tensor is not None:
                p = prev_wf_tensor.detach().cpu()
                if p.dim() == 1:
                    p = p.unsqueeze(0)
                prev_waveforms.append(p)
            else:
                prev_waveforms.append(None)

    mean_ms = total_ms / max(1, n)
    return gen_waveforms, true_waveforms, prev_waveforms, mean_ms
