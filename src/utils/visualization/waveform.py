"""Waveform plot rendering.

Per-sample plot helpers (``render_waveform_overlay``, ``render_waveform_plot_only``)
and the batch-level aggregator (``create_waveform_images``) used by training
visualization.
"""
import math
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from src.utils.visualization.style import (
    _apply_legend,
    _apply_tick_style,
    _set_axis_labels,
    _set_axis_title,
)


def _join_prev_to_current(
    prev: np.ndarray, current: np.ndarray, sample_rate: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Place prev before t=0 and extend it to current[0] so the lines connect."""
    prev_len = len(prev)
    tt_prev = (np.arange(prev_len) - prev_len) / max(1, sample_rate)
    if len(current) == 0:
        return tt_prev, prev
    return np.append(tt_prev, 0.0), np.append(prev, current[0])


def render_waveform_overlay(
    gt_wf: np.ndarray,
    pr_wf: np.ndarray,
    label_str: str,
    sample_rate: int,
    *,
    prev_gt_wf: Optional[np.ndarray] = None,
    plot_prev: bool = False,
    prev_before_current: bool = False,
    info_text: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show_x_axis_label: bool = True,
    show_y_axis_label: bool = True,
    show_legend: bool = True,
    show_title: bool = True,
    figure_size: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    line_alpha: float = 1.0,
    prev_alpha: float = 1.0,
) -> Image.Image:
    """Render ground truth and prediction waveforms as an overlay plot."""
    gt_color = "#ff7f0e"
    L = int(min(len(gt_wf), len(pr_wf)))
    if L <= 0:
        L = int(max(len(gt_wf), len(pr_wf)))
    gt = gt_wf[:L]
    pr = pr_wf[:L]
    if figure_size is None:
        fig_w, fig_h = 6, 2
    else:
        fig_w, fig_h = figure_size
    dpi_val = 100 if dpi is None else int(dpi)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi_val)
    ax = fig.add_subplot(111)
    tt = np.arange(L) / max(1, sample_rate)
    ax.plot(tt, gt, label="GT", linewidth=1.0, color=gt_color, alpha=line_alpha)
    ax.plot(tt, pr, label="PRED", linewidth=1.0, alpha=line_alpha)
    if plot_prev and prev_gt_wf is not None and len(prev_gt_wf) > 0:
        prev_len = len(prev_gt_wf)
        prev = prev_gt_wf[:prev_len]
        tt_prev = np.arange(prev_len) / max(1, sample_rate)
        if prev_before_current:
            tt_prev, prev = _join_prev_to_current(prev, gt, sample_rate)
        ax.plot(
            tt_prev,
            prev,
            linewidth=1.0,
            linestyle="--",
            color=gt_color,
            alpha=prev_alpha,
        )
    if info_text is None:
        info_text = ""
    if show_title:
        _set_axis_title(ax, f"label: {label_str}  {info_text}")
    _set_axis_labels(
        ax,
        xlabel="time [s]",
        ylabel="amp",
        show_x=bool(show_x_axis_label),
        show_y=bool(show_y_axis_label),
    )
    _apply_tick_style(ax)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if show_legend:
        _apply_legend(ax, loc="upper right")
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]
    plt.close(fig)
    return Image.fromarray(img)


def truncate_prev_waveform(
    prev_np: Optional[np.ndarray],
    prev_waveform_ratio: float = 1.0,
) -> Optional[np.ndarray]:
    """Truncate previous waveform to keep only the specified ratio from the end."""
    if prev_np is None or prev_waveform_ratio <= 0.0:
        return None
    length = int(prev_np.shape[-1])
    if length <= 0:
        return None
    keep = max(1, int(math.ceil(length * prev_waveform_ratio)))
    keep = min(keep, length)
    return prev_np[..., -keep:]


def render_waveform_plot_only(
    gen_wf_np: np.ndarray,
    label_str: str,
    sample_rate: int,
    *,
    prev_wf_np: Optional[np.ndarray] = None,
    include_prev: bool = False,
    prev_before_current: bool = False,
    info_text: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show_x_axis_label: bool = True,
    show_y_axis_label: bool = True,
    show_legend: bool = True,
    show_title: bool = True,
    figure_size: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    line_alpha: float = 1.0,
    prev_alpha: float = 1.0,
) -> Image.Image:
    """Render only the generated waveform (and optionally previous waveform) as a plot."""
    gt_color = "#ff7f0e"
    if figure_size is None:
        fig_w, fig_h = 6, 2
    else:
        fig_w, fig_h = figure_size
    dpi_val = 100 if dpi is None else int(dpi)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi_val)
    ax = fig.add_subplot(111)
    tt = np.arange(len(gen_wf_np)) / max(1, sample_rate)
    ax.plot(tt, gen_wf_np, label="PRED", linewidth=1.0, color="blue", alpha=line_alpha)
    if include_prev and prev_wf_np is not None:
        prev_len = len(prev_wf_np)
        tt_prev = np.arange(prev_len) / max(1, sample_rate)
        prev = prev_wf_np
        if prev_before_current:
            tt_prev, prev = _join_prev_to_current(prev, gen_wf_np, sample_rate)
        ax.plot(
            tt_prev,
            prev,
            linewidth=1.0,
            linestyle="--",
            color=gt_color,
            alpha=prev_alpha,
        )
    if info_text is None:
        info_text = ""
    if show_title:
        _set_axis_title(ax, f"waveform | label: {label_str}  {info_text}")
    _set_axis_labels(
        ax,
        xlabel="time [s]",
        ylabel="amp",
        show_x=bool(show_x_axis_label),
        show_y=bool(show_y_axis_label),
    )
    _apply_tick_style(ax)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if show_legend:
        _apply_legend(ax, loc="upper right")
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]
    plt.close(fig)
    return Image.fromarray(img)


def create_waveform_images(
    gen_waveforms: List[torch.Tensor],
    true_waveforms: List[torch.Tensor],
    prev_waveforms: List[Optional[torch.Tensor]],
    labels: torch.Tensor,
    full_ds,
    render_prev_waveform: bool,
    prev_waveform_ratio: float,
    *,
    force_means_raw: Optional[torch.Tensor] = None,
    velocity_means_raw: Optional[torch.Tensor] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show_x_axis_label: bool = True,
    show_y_axis_label: bool = True,
    show_legend: bool = True,
    show_title: bool = True,
    figure_size: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    waveform_alpha: float = 1.0,
    prev_waveform_alpha: float = 1.0,
) -> Tuple[List[Image.Image], List[Image.Image]]:
    """Create waveform visualization images.

    Returns:
        Tuple of (waveform_images, waveform_images_prev).
    """
    waveform_images = []
    waveform_images_prev = []

    num_samples = len(gen_waveforms)
    for i in range(num_samples):
        gen_wf = gen_waveforms[i]
        if isinstance(gen_wf, torch.Tensor):
            gen_wf_np = gen_wf.squeeze().numpy()
        else:
            gen_wf_np = gen_wf.squeeze()

        label_idx = int(labels[i].cpu().item())
        label_str = full_ds.classes[label_idx] if 0 <= label_idx < len(full_ds.classes) else str(label_idx)
        info_text = ""
        if force_means_raw is not None and velocity_means_raw is not None:
            force_vec = force_means_raw[i].detach().cpu().view(-1)
            vel_vec = velocity_means_raw[i].detach().cpu().view(-1)
            if force_vec.numel() > 0 and vel_vec.numel() >= 2:
                force_val = float(force_vec[0].item())
                vel_x = float(vel_vec[0].item())
                vel_y = float(vel_vec[1].item())
                info_text = f"force: {force_val:.2f} | vel: ({vel_x:.2f}, {vel_y:.2f})"
        prev_wf_np = None
        if render_prev_waveform and i < len(prev_waveforms):
            prev_entry = prev_waveforms[i]
            if prev_entry is not None:
                prev_np_full = prev_entry.squeeze().numpy()
                prev_wf_np = truncate_prev_waveform(prev_np_full, prev_waveform_ratio)

        if i < len(true_waveforms):
            true_wf = true_waveforms[i]
            if isinstance(true_wf, torch.Tensor):
                true_wf_np = true_wf.squeeze().numpy()
            else:
                true_wf_np = true_wf.squeeze()

            waveform_img = render_waveform_overlay(
                gt_wf=true_wf_np,
                pr_wf=gen_wf_np,
                label_str=label_str,
                sample_rate=full_ds.sample_rate,
                plot_prev=False,
                info_text=info_text,
                ylim=ylim,
                show_x_axis_label=show_x_axis_label,
                show_y_axis_label=show_y_axis_label,
                show_legend=show_legend,
                show_title=show_title,
                figure_size=figure_size,
                dpi=dpi,
                line_alpha=waveform_alpha,
                prev_alpha=prev_waveform_alpha,
            )
            waveform_images.append(waveform_img)

            if render_prev_waveform and prev_wf_np is not None:
                waveform_img_prev = render_waveform_overlay(
                    gt_wf=true_wf_np,
                    pr_wf=gen_wf_np,
                    label_str=label_str,
                    sample_rate=full_ds.sample_rate,
                    prev_gt_wf=prev_wf_np,
                    plot_prev=True,
                    prev_before_current=True,
                    info_text=info_text,
                    ylim=ylim,
                    show_x_axis_label=show_x_axis_label,
                    show_y_axis_label=show_y_axis_label,
                    show_legend=show_legend,
                    show_title=show_title,
                    figure_size=figure_size,
                    dpi=dpi,
                    line_alpha=waveform_alpha,
                    prev_alpha=prev_waveform_alpha,
                )
                waveform_images_prev.append(waveform_img_prev)
        else:
            waveform_img = render_waveform_plot_only(
                gen_wf_np,
                label_str=label_str,
                sample_rate=full_ds.sample_rate,
                include_prev=False,
                info_text=info_text,
                ylim=ylim,
                show_x_axis_label=show_x_axis_label,
                show_y_axis_label=show_y_axis_label,
                show_legend=show_legend,
                show_title=show_title,
                figure_size=figure_size,
                dpi=dpi,
                line_alpha=waveform_alpha,
                prev_alpha=prev_waveform_alpha,
            )
            waveform_images.append(waveform_img)

            if render_prev_waveform and prev_wf_np is not None:
                waveform_img_prev = render_waveform_plot_only(
                    gen_wf_np,
                    label_str=label_str,
                    sample_rate=full_ds.sample_rate,
                    prev_wf_np=prev_wf_np,
                    include_prev=True,
                    prev_before_current=True,
                    info_text=info_text,
                    ylim=ylim,
                    show_x_axis_label=show_x_axis_label,
                    show_y_axis_label=show_y_axis_label,
                    show_legend=show_legend,
                    show_title=show_title,
                    figure_size=figure_size,
                    dpi=dpi,
                    line_alpha=waveform_alpha,
                    prev_alpha=prev_waveform_alpha,
                )
                waveform_images_prev.append(waveform_img_prev)

    return waveform_images, waveform_images_prev
