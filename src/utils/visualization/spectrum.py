"""STFT magnitude spectrum (1D) rendering.

These functions average frame magnitudes to produce a single ``magnitude vs frequency``
curve. The 2D spectrogram (time × frequency heatmap) lives in ``spectrogram.py``.
"""
import os
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from src.utils.visualization.style import (
    _apply_legend,
    _apply_tick_style,
    _set_axis_labels,
    _set_axis_title,
)


def _compute_fft_spectrum_all_frames(
    wf: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FFT magnitude spectrum using all frames (averaged over time, linear scale)."""
    wf_1d = wf.squeeze()
    if wf_1d.ndim == 0:
        wf_1d = wf_1d.reshape(1)
    L = wf_1d.shape[-1]
    if L <= 0:
        freqs = np.linspace(0.0, float(sample_rate) / 2.0, n_fft // 2 + 1)
        return freqs, np.zeros_like(freqs)

    win_length = int(win_length)
    if win_length <= 0 or win_length > L:
        win_length = min(L, n_fft)
    hop = int(hop_length)
    if hop <= 0:
        hop = max(1, win_length // 4)

    if L <= win_length:
        frames = wf_1d[None, :]
        if frames.shape[-1] < win_length:
            pad = win_length - frames.shape[-1]
            frames = np.pad(frames, ((0, 0), (0, pad)))
    else:
        starts = np.arange(0, L - win_length + 1, hop, dtype=int)
        if len(starts) == 0:
            starts = np.array([0], dtype=int)
        frames = np.stack([wf_1d[s : s + win_length] for s in starts], axis=0)

    window = np.hanning(win_length).astype(frames.dtype)
    frames_win = frames * window[None, :]
    spec = np.fft.rfft(frames_win, n=n_fft, axis=-1)  # [F, freq]
    mag = np.abs(spec)  # [F, freq]
    mag_mean = mag.mean(axis=0).astype(np.float32)  # [freq]

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / max(1, sample_rate))
    return freqs, mag_mean


def render_stft_spectrum_overlay(
    gt_wf: np.ndarray,
    pr_wf: np.ndarray,
    label_str: str,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    *,
    info_text: Optional[str] = None,
    show_x_axis_label: bool = True,
    show_y_axis_label: bool = True,
    show_legend: bool = True,
    show_title: bool = True,
    figure_size: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    line_alpha: float = 1.0,
) -> Image.Image:
    """Render 1D STFT magnitude (spectrum) comparison for GT and prediction (normalized per waveform)."""
    freqs, gt_mag_raw = _compute_fft_spectrum_all_frames(gt_wf, sample_rate, n_fft, hop_length, win_length)
    _, pr_mag_raw = _compute_fft_spectrum_all_frames(pr_wf, sample_rate, n_fft, hop_length, win_length)

    eps = 1e-8
    gt_max = float(gt_mag_raw.max()) if gt_mag_raw.size > 0 else 0.0
    pr_max = float(pr_mag_raw.max()) if pr_mag_raw.size > 0 else 0.0
    gt_mag = gt_mag_raw / (gt_max + eps) if gt_max > 0.0 else np.zeros_like(gt_mag_raw)
    pr_mag = pr_mag_raw / (pr_max + eps) if pr_max > 0.0 else np.zeros_like(pr_mag_raw)

    # Zero out tiny components so they sit on the baseline
    min_amp = 1e-3
    gt_mag[gt_mag < min_amp] = 0.0
    pr_mag[pr_mag < min_amp] = 0.0

    if figure_size is None:
        fig_w, fig_h = 6, 2
    else:
        fig_w, fig_h = figure_size
    dpi_val = 100 if dpi is None else int(dpi)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi_val)
    ax = fig.add_subplot(111)
    ax.plot(freqs, gt_mag, label="GT", linewidth=1.0, color="#ff7f0e", alpha=line_alpha)
    ax.plot(freqs, pr_mag, label="PRED", linewidth=1.0, color="blue", alpha=line_alpha)
    ax.set_ylim(0.0, 1.0)
    _set_axis_labels(
        ax,
        xlabel="frequency [Hz]",
        ylabel="normalized magnitude",
        show_x=bool(show_x_axis_label),
        show_y=bool(show_y_axis_label),
    )
    _apply_tick_style(ax)
    if info_text is None:
        info_text = ""
    if show_title:
        _set_axis_title(ax, f"STFT spectrum | label: {label_str}  {info_text}")
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


def render_stft_spectrum_overlay_raw(
    gt_wf: np.ndarray,
    pr_wf: np.ndarray,
    label_str: str,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    *,
    info_text: Optional[str] = None,
    show_x_axis_label: bool = True,
    show_y_axis_label: bool = True,
    show_legend: bool = True,
    show_title: bool = True,
    figure_size: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    line_alpha: float = 1.0,
    dump_dir: Optional[str] = None,
    dump_prefix: Optional[str] = None,
) -> Image.Image:
    """Render 1D STFT magnitude spectrum without normalization.

    Y-axis upper limit is based on GT max (*1.2). PRED may exceed the limit.
    """
    freqs, gt_mag = _compute_fft_spectrum_all_frames(gt_wf, sample_rate, n_fft, hop_length, win_length)
    _, pr_mag = _compute_fft_spectrum_all_frames(pr_wf, sample_rate, n_fft, hop_length, win_length)

    if dump_dir is not None and dump_prefix is not None:
        try:
            os.makedirs(dump_dir, exist_ok=True)
            freqs_path = os.path.join(dump_dir, "freqs.npy")
            if not os.path.exists(freqs_path):
                np.save(freqs_path, freqs.astype(np.float32))
            # [2, F]: GT, PRED
            mags = np.stack(
                [
                    np.asarray(gt_mag, dtype=np.float32),
                    np.asarray(pr_mag, dtype=np.float32),
                ],
                axis=0,
            )
            np.save(os.path.join(dump_dir, f"{dump_prefix}.npy"), mags)
        except Exception:
            # A failed dump must not break visualization
            pass

    if figure_size is None:
        fig_w, fig_h = 6, 2
    else:
        fig_w, fig_h = figure_size
    dpi_val = 100 if dpi is None else int(dpi)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi_val)
    ax = fig.add_subplot(111)
    ax.plot(freqs, gt_mag, label="GT", linewidth=1.0, color="#ff7f0e", alpha=line_alpha)
    ax.plot(freqs, pr_mag, label="PRED", linewidth=1.0, color="blue", alpha=line_alpha)
    gt_max = float(gt_mag.max()) if getattr(gt_mag, "size", 0) else 0.0
    y_max = max(1e-8, gt_max * 1.2)
    y_min = -0.05 * y_max
    ax.set_ylim(y_min, y_max)
    _set_axis_labels(
        ax,
        xlabel="frequency [Hz]",
        ylabel="magnitude",
        show_x=bool(show_x_axis_label),
        show_y=bool(show_y_axis_label),
    )
    _apply_tick_style(ax)
    if info_text is None:
        info_text = ""
    if show_title:
        _set_axis_title(ax, f"STFT spectrum (raw) | label: {label_str}  {info_text}")
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
