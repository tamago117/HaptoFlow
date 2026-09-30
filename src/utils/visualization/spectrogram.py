"""2D spectrogram rendering (linear + log-mel) and the STFT/spectrogram aggregator.

The aggregator (``create_stft_and_spectrogram_images``) drives both the spectrum
(1D) and spectrogram (2D) renderers.
"""
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from src.utils.visualization.spectrum import (
    render_stft_spectrum_overlay,
    render_stft_spectrum_overlay_raw,
)
from src.utils.visualization.style import (
    _apply_tick_style,
    _set_axis_labels,
    _set_figure_suptitle,
)


def render_spectrogram_side_by_side(
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
    show_title: bool = True,
    figure_size: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
) -> Image.Image:
    """Render GT and prediction spectrograms side by side (no overlay)."""
    gt_1d = gt_wf.squeeze()
    pr_1d = pr_wf.squeeze()

    if figure_size is None:
        fig_w, fig_h = 8, 3
    else:
        fig_w, fig_h = figure_size
    dpi_val = 100 if dpi is None else int(dpi)
    fig, axs = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=dpi_val, sharey=True)
    noverlap = max(0, int(win_length) - int(hop_length))
    axs[0].specgram(
        gt_1d,
        NFFT=int(n_fft),
        Fs=sample_rate,
        noverlap=noverlap,
        cmap="magma",
    )
    if show_title:
        axs[0].set_title("GT")
    _set_axis_labels(
        axs[0],
        xlabel="time [s]",
        ylabel="frequency [Hz]",
        show_x=bool(show_x_axis_label),
        show_y=bool(show_y_axis_label),
    )
    _apply_tick_style(axs[0])

    axs[1].specgram(
        pr_1d,
        NFFT=int(n_fft),
        Fs=sample_rate,
        noverlap=noverlap,
        cmap="magma",
    )
    if show_title:
        axs[1].set_title("PRED")
    axs[1].set_xlabel("time [s]" if bool(show_x_axis_label) else "")
    if not bool(show_y_axis_label):
        axs[1].set_ylabel("")
    _apply_tick_style(axs[1])

    if info_text is None:
        info_text = ""
    if show_title:
        _set_figure_suptitle(fig, f"Spectrogram | label: {label_str}  {info_text}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=0, w_pad=0, h_pad=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]
    plt.close(fig)
    return Image.fromarray(img)


def render_logmel_spectrogram_side_by_side(
    gt_wf: np.ndarray,
    pr_wf: np.ndarray,
    label_str: str,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    *,
    info_text: Optional[str] = None,
    show_x_axis_label: bool = True,
    show_y_axis_label: bool = True,
    show_title: bool = True,
    figure_size: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
) -> Image.Image:
    """Render GT and prediction Log-Mel Spectrograms side by side."""
    gt_1d = gt_wf.squeeze()
    pr_1d = pr_wf.squeeze()

    if f_max is None:
        f_max = sample_rate / 2.0

    S_gt = librosa.feature.melspectrogram(
        y=gt_1d,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
    )
    S_db_gt = librosa.power_to_db(S_gt, ref=np.max)

    S_pr = librosa.feature.melspectrogram(
        y=pr_1d,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
    )
    S_db_pr = librosa.power_to_db(S_pr, ref=np.max)

    n_frames_gt = S_db_gt.shape[1]
    n_frames_pr = S_db_pr.shape[1]

    n_frames_max = max(n_frames_gt, n_frames_pr)

    times_unified = librosa.frames_to_time(np.arange(n_frames_max), sr=sample_rate, hop_length=hop_length)
    time_max_unified = times_unified[-1] if len(times_unified) > 0 else (n_frames_max * hop_length / sample_rate)

    # Pad the shorter spectrogram with its min dB so both share one time axis
    if n_frames_gt < n_frames_max:
        pad_width = n_frames_max - n_frames_gt
        min_db = S_db_gt.min()
        S_db_gt_aligned = np.pad(S_db_gt, ((0, 0), (0, pad_width)), mode='constant', constant_values=min_db)
    else:
        S_db_gt_aligned = S_db_gt

    if n_frames_pr < n_frames_max:
        pad_width = n_frames_max - n_frames_pr
        min_db = S_db_pr.min()
        S_db_pr_aligned = np.pad(S_db_pr, ((0, 0), (0, pad_width)), mode='constant', constant_values=min_db)
    else:
        S_db_pr_aligned = S_db_pr

    if figure_size is None:
        fig_w, fig_h = 8, 3
    else:
        fig_w, fig_h = figure_size
    dpi_val = 100 if dpi is None else int(dpi)
    fig, axs = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=dpi_val, sharey=True, sharex=True)

    librosa.display.specshow(
        S_db_gt_aligned,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        fmin=f_min,
        fmax=f_max,
        ax=axs[0],
        cmap="magma",
        x_coords=times_unified,
    )
    if show_title:
        axs[0].set_title("GT (Log-Mel)")
    axs[0].label_outer()
    axs[0].set_xlim(0, time_max_unified)
    plt.setp(axs[0].xaxis.get_majorticklabels(), fontsize=8)

    librosa.display.specshow(
        S_db_pr_aligned,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        fmin=f_min,
        fmax=f_max,
        ax=axs[1],
        cmap="magma",
        x_coords=times_unified,
    )
    if show_title:
        axs[1].set_title("PRED (Log-Mel)")
    axs[1].label_outer()
    axs[1].set_xlim(0, time_max_unified)
    plt.setp(axs[1].xaxis.get_majorticklabels(), fontsize=8)

    # specshow sets its own axis labels; clear them when disabled
    for ax in axs:
        if not bool(show_x_axis_label):
            ax.set_xlabel("")
        if not bool(show_y_axis_label):
            ax.set_ylabel("")
        _apply_tick_style(ax)

    if info_text is None:
        info_text = ""
    if show_title:
        _set_figure_suptitle(fig, f"Log-Mel Spectrogram | label: {label_str}  {info_text}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=0, w_pad=0, h_pad=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = canvas.buffer_rgba()
    img_pil = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]
    plt.close(fig)

    return Image.fromarray(img_pil)


def create_stft_and_spectrogram_images(
    gen_waveforms: List[torch.Tensor],
    true_waveforms: List[torch.Tensor],
    labels: torch.Tensor,
    full_ds,
    n_fft: int,
    hop_length: int,
    win_length: int,
    *,
    force_means_raw: Optional[torch.Tensor] = None,
    velocity_means_raw: Optional[torch.Tensor] = None,
    enable_mel_spectrogram: bool = False,
    mel_n_mels: int = 80,
    mel_f_min: float = 0.0,
    mel_f_max: Optional[float] = None,
    show_x_axis_label: bool = True,
    show_y_axis_label: bool = True,
    show_legend: bool = True,
    show_title: bool = True,
    figure_size_spectrum: Optional[Tuple[float, float]] = None,
    figure_size_spectrogram: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    spectrum_alpha: float = 1.0,
    stft_dump_dir: Optional[str] = None,
) -> Tuple[List[Image.Image], List[Image.Image], List[Image.Image], List[Image.Image]]:
    """Create STFT spectrum (normalized & raw), spectrogram, and optionally Log-Mel comparison images.

    Returns:
        Tuple of (stft_spectrum_images_normalized, stft_spectrum_images_raw,
                  spectrogram_images, mel_spectrogram_images).
    """
    stft_images_normalized: List[Image.Image] = []
    stft_images_raw: List[Image.Image] = []
    spec_images: List[Image.Image] = []
    mel_images: List[Image.Image] = []

    num_samples = min(len(gen_waveforms), len(true_waveforms))
    if num_samples == 0:
        return stft_images_normalized, stft_images_raw, spec_images, mel_images

    for i in range(num_samples):
        gen_wf = gen_waveforms[i]
        true_wf = true_waveforms[i]
        if isinstance(gen_wf, torch.Tensor):
            gen_wf_np = gen_wf.squeeze().numpy()
        else:
            gen_wf_np = gen_wf.squeeze()
        if isinstance(true_wf, torch.Tensor):
            true_wf_np = true_wf.squeeze().numpy()
        else:
            true_wf_np = true_wf.squeeze()

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

        stft_img_norm = render_stft_spectrum_overlay(
            gt_wf=true_wf_np,
            pr_wf=gen_wf_np,
            label_str=label_str,
            sample_rate=full_ds.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            info_text=info_text,
            show_x_axis_label=show_x_axis_label,
            show_y_axis_label=show_y_axis_label,
            show_legend=show_legend,
            show_title=show_title,
            figure_size=figure_size_spectrum,
            dpi=dpi,
            line_alpha=spectrum_alpha,
        )
        stft_images_normalized.append(stft_img_norm)

        stft_img_raw = render_stft_spectrum_overlay_raw(
            gt_wf=true_wf_np,
            pr_wf=gen_wf_np,
            label_str=label_str,
            sample_rate=full_ds.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            info_text=info_text,
            show_x_axis_label=show_x_axis_label,
            show_y_axis_label=show_y_axis_label,
            show_legend=show_legend,
            show_title=show_title,
            figure_size=figure_size_spectrum,
            dpi=dpi,
            line_alpha=spectrum_alpha,
            dump_dir=stft_dump_dir,
            dump_prefix=f"spectrum_{i:04d}" if stft_dump_dir is not None else None,
        )
        stft_images_raw.append(stft_img_raw)

        spec_img = render_spectrogram_side_by_side(
            gt_wf=true_wf_np,
            pr_wf=gen_wf_np,
            label_str=label_str,
            sample_rate=full_ds.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            info_text=info_text,
            show_x_axis_label=show_x_axis_label,
            show_y_axis_label=show_y_axis_label,
            show_title=show_title,
            figure_size=figure_size_spectrogram,
            dpi=dpi,
        )
        spec_images.append(spec_img)

        if enable_mel_spectrogram:
            mel_img = render_logmel_spectrogram_side_by_side(
                gt_wf=true_wf_np,
                pr_wf=gen_wf_np,
                label_str=label_str,
                sample_rate=full_ds.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=mel_n_mels,
                f_min=mel_f_min,
                f_max=mel_f_max,
                info_text=info_text,
                show_x_axis_label=show_x_axis_label,
                show_y_axis_label=show_y_axis_label,
                show_title=show_title,
                figure_size=figure_size_spectrogram,
                dpi=dpi,
            )
            mel_images.append(mel_img)

    return stft_images_normalized, stft_images_raw, spec_images, mel_images
