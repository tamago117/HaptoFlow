"""Font / tick / legend style helpers shared by waveform, spectrum, and spectrogram renderers.

Also hosts the generic image utilities (``tensor_to_pil``, ``render_side_by_side``).
"""
from typing import Optional

import matplotlib
import torch
from PIL import Image, ImageDraw

matplotlib.use("Agg")


# Mutated by ``configure_visualization_fonts``; other modules read these through
# the helpers below so they always see the latest values.
_LABEL_FONT_FAMILY = "Arial"
_LABEL_FONT_WEIGHT = "bold"
_LABEL_FONT_SIZE: Optional[float] = None
_TITLE_FONT_SIZE: Optional[float] = None
_LEGEND_FONT_SIZE: Optional[float] = None
_TICK_FONT_SIZE: Optional[float] = None


def configure_visualization_fonts(
    *,
    family: Optional[str] = None,
    weight: Optional[str] = None,
    axis_label_size: Optional[float] = None,
    title_size: Optional[float] = None,
    legend_size: Optional[float] = None,
    tick_size: Optional[float] = None,
) -> None:
    """Configure font style used in this module's matplotlib figures.

    Notes:
    - If Arial is not installed, matplotlib will fallback to a available font.
    - Sizes are in points, following matplotlib conventions.
    """
    global _LABEL_FONT_FAMILY, _LABEL_FONT_WEIGHT
    global _LABEL_FONT_SIZE, _TITLE_FONT_SIZE, _LEGEND_FONT_SIZE, _TICK_FONT_SIZE

    if family is not None:
        _LABEL_FONT_FAMILY = str(family)
    if weight is not None:
        _LABEL_FONT_WEIGHT = str(weight)
    _LABEL_FONT_SIZE = None if axis_label_size is None else float(axis_label_size)
    _TITLE_FONT_SIZE = None if title_size is None else float(title_size)
    _LEGEND_FONT_SIZE = None if legend_size is None else float(legend_size)
    _TICK_FONT_SIZE = None if tick_size is None else float(tick_size)


def _font_family() -> str:
    return _LABEL_FONT_FAMILY


def _font_weight() -> str:
    return _LABEL_FONT_WEIGHT


def _label_size() -> Optional[float]:
    return _LABEL_FONT_SIZE


def _title_size() -> Optional[float]:
    return _TITLE_FONT_SIZE


def _legend_size() -> Optional[float]:
    return _LEGEND_FONT_SIZE


def _tick_size() -> Optional[float]:
    return _TICK_FONT_SIZE


def _apply_tick_style(ax) -> None:
    # tick_params cannot set the font family; set it on the Text objects
    if _TICK_FONT_SIZE is not None:
        ax.tick_params(axis="both", which="major", labelsize=_TICK_FONT_SIZE, direction="in")
    else:
        ax.tick_params(axis="both", which="major", direction="in")
    for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        t.set_fontname(_LABEL_FONT_FAMILY)
        t.set_fontweight(_LABEL_FONT_WEIGHT)


def _set_axis_labels(
    ax,
    *,
    xlabel: str,
    ylabel: str,
    show_x: bool,
    show_y: bool,
) -> None:
    ax.set_xlabel(
        xlabel if show_x else "",
        fontname=_LABEL_FONT_FAMILY,
        fontweight=_LABEL_FONT_WEIGHT,
        fontsize=_LABEL_FONT_SIZE,
    )
    ax.set_ylabel(
        ylabel if show_y else "",
        fontname=_LABEL_FONT_FAMILY,
        fontweight=_LABEL_FONT_WEIGHT,
        fontsize=_LABEL_FONT_SIZE,
    )


def _set_axis_title(ax, title: str) -> None:
    ax.set_title(
        title,
        fontname=_LABEL_FONT_FAMILY,
        fontweight=_LABEL_FONT_WEIGHT,
        fontsize=_TITLE_FONT_SIZE,
    )


def _set_figure_suptitle(fig, title: str) -> None:
    fig.suptitle(
        title,
        fontname=_LABEL_FONT_FAMILY,
        fontweight=_LABEL_FONT_WEIGHT,
        fontsize=_TITLE_FONT_SIZE,
    )


def _apply_legend(ax, **kwargs) -> None:
    prop = {"family": _LABEL_FONT_FAMILY, "weight": _LABEL_FONT_WEIGHT}
    if _LEGEND_FONT_SIZE is not None:
        prop["size"] = _LEGEND_FONT_SIZE
    ax.legend(
        prop=prop,
        **kwargs,
    )


def tensor_to_pil(x01: torch.Tensor) -> Image.Image:
    """Convert tensor in [0, 1] range to PIL Image.

    Args:
        x01: Tensor in [0, 1] range, shape (1, H, W) or (H, W).

    Returns:
        PIL Image in grayscale mode 'L'.
    """
    if x01.dim() == 3:
        x01 = x01.squeeze(0)
    arr = (x01.clamp(0, 1).numpy() * 255.0).astype("uint8")
    # Pillow 10+ infers the mode from the array
    im = Image.fromarray(arr)
    return im


def render_side_by_side(
    gt01: torch.Tensor,
    pr01: torch.Tensor,
    label_str: str,
    img_scale: int = 4,
    canvas_w: int = 640,
    canvas_h: int = 240,
    *,
    info_text: Optional[str] = None,
) -> Image.Image:
    """Render ground truth and prediction images side by side.

    Args:
        gt01: Ground truth tensor in [0, 1] range.
        pr01: Prediction tensor in [0, 1] range.
        label_str: Label string to display.
        img_scale: Scale factor for image size (default: 4).
        canvas_w: Canvas width in pixels (default: 640).
        canvas_h: Canvas height in pixels (default: 240).

    Returns:
        PIL Image with GT and PRED side by side.
    """
    gt_im = tensor_to_pil(gt01.cpu())
    pr_im = tensor_to_pil(pr01.cpu())
    gt_im = gt_im.resize((gt_im.width * img_scale, gt_im.height * img_scale), resample=Image.NEAREST)
    pr_im = pr_im.resize((pr_im.width * img_scale, pr_im.height * img_scale), resample=Image.NEAREST)

    canvas = Image.new("L", (canvas_w, canvas_h), color=255)
    draw = ImageDraw.Draw(canvas)

    pad = 8
    if info_text is None:
        info_text = ""
    title = f"label: {label_str}  |  {info_text}"
    draw.text((pad, pad), title, fill=0)
    y0 = 32
    x0 = pad
    draw.text((x0, y0 - 16), "GT", fill=0)
    canvas.paste(gt_im, (x0, y0))

    x1 = x0 + gt_im.width + pad
    draw.text((x1, y0 - 16), "PRED", fill=0)
    canvas.paste(pr_im, (x1, y0))
    return canvas
