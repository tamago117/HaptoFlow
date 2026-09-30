"""Visualization helpers: waveform / STFT spectrum / spectrogram rendering, fonts, viz config.

Modules:

- ``style``: font configuration, axis/tick/legend helpers, generic image utils
- ``waveform``: 1D waveform overlay plots + batch aggregator
- ``spectrum``: 1D STFT magnitude spectrum plots
- ``spectrogram``: 2D linear and log-mel spectrogram plots + STFT/spectrogram aggregator
- ``viz_config``: parsed VizConfig value object used by training

The public API is re-exported here.
"""
from src.utils.visualization.spectrogram import (
    create_stft_and_spectrogram_images,
    render_logmel_spectrogram_side_by_side,
    render_spectrogram_side_by_side,
)
from src.utils.visualization.spectrum import (
    render_stft_spectrum_overlay,
    render_stft_spectrum_overlay_raw,
)
from src.utils.visualization.style import (
    configure_visualization_fonts,
    render_side_by_side,
    tensor_to_pil,
)
from src.utils.visualization.viz_config import VizConfig
from src.utils.visualization.waveform import (
    create_waveform_images,
    render_waveform_overlay,
    render_waveform_plot_only,
    truncate_prev_waveform,
)

__all__ = [
    "configure_visualization_fonts",
    "tensor_to_pil",
    "render_side_by_side",
    "render_waveform_overlay",
    "render_waveform_plot_only",
    "truncate_prev_waveform",
    "create_waveform_images",
    "render_stft_spectrum_overlay",
    "render_stft_spectrum_overlay_raw",
    "render_spectrogram_side_by_side",
    "render_logmel_spectrogram_side_by_side",
    "create_stft_and_spectrogram_images",
    "VizConfig",
]
