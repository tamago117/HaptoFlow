"""Parsed visualization config used by training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from omegaconf import OmegaConf


def _to_xy_tuple(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    container = OmegaConf.to_container(value, resolve=True) if hasattr(value, "_content") else value
    if isinstance(container, (list, tuple)) and len(container) == 2:
        return (float(container[0]), float(container[1]))
    return None


def _to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float_or(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class VizConfig:
    """Parsed visualization config — used by train.py."""
    render_prev_waveform: bool = False
    prev_waveform_ratio: float = 1.0
    enable_stft_spectrum: bool = True
    enable_spectrogram: bool = True
    enable_mel_spectrogram: bool = False
    waveform_ylim: Optional[Tuple[float, float]] = None
    show_x_axis_label: bool = True
    show_y_axis_label: bool = True
    show_legend: bool = True
    show_title: bool = True
    figure_size_waveform: Optional[Tuple[float, float]] = None
    figure_size_spectrum: Optional[Tuple[float, float]] = None
    figure_size_spectrogram: Optional[Tuple[float, float]] = None
    figure_dpi: Optional[int] = None
    line_alpha: float = 1.0
    font_axis_label_size: Optional[int] = None
    font_title_size: Optional[int] = None
    font_legend_size: Optional[int] = None
    font_tick_size: Optional[int] = None

    @classmethod
    def from_cfg(
        cls,
        cfg: Any,
        *,
        default_waveform_ylim: Optional[Tuple[float, float]] = None,
    ) -> "VizConfig":
        """Build VizConfig from a cfg.visualization section (or its absence)."""
        viz = getattr(cfg, "visualization", None)
        if viz is None:
            return cls(waveform_ylim=default_waveform_ylim)

        return cls(
            render_prev_waveform=bool(getattr(viz, "render_prev_waveform", False)),
            prev_waveform_ratio=max(0.0, min(1.0, _to_float_or(getattr(viz, "prev_waveform_ratio", 1.0), 1.0))),
            enable_stft_spectrum=bool(getattr(viz, "enable_stft_spectrum", True)),
            enable_spectrogram=bool(getattr(viz, "enable_spectrogram", True)),
            enable_mel_spectrogram=bool(getattr(viz, "enable_mel_spectrogram", False)),
            waveform_ylim=_to_xy_tuple(getattr(viz, "waveform_ylim", None)) or default_waveform_ylim,
            show_x_axis_label=bool(getattr(viz, "show_x_axis_label", True)),
            show_y_axis_label=bool(getattr(viz, "show_y_axis_label", True)),
            show_legend=bool(getattr(viz, "show_legend", True)),
            show_title=bool(getattr(viz, "show_title", True)),
            figure_size_waveform=_to_xy_tuple(getattr(viz, "figure_size_waveform", None)),
            figure_size_spectrum=_to_xy_tuple(getattr(viz, "figure_size_spectrum", None)),
            figure_size_spectrogram=_to_xy_tuple(getattr(viz, "figure_size_spectrogram", None)),
            figure_dpi=_to_int_or_none(getattr(viz, "figure_dpi", None)),
            line_alpha=_to_float_or(getattr(viz, "line_alpha", 1.0), 1.0),
            font_axis_label_size=getattr(viz, "font_size_axis_label", None),
            font_title_size=getattr(viz, "font_size_title", None),
            font_legend_size=getattr(viz, "font_size_legend", None),
            font_tick_size=getattr(viz, "font_size_tick", None),
        )
