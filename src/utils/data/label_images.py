"""Low-res per-label texture thumbnails for the Gradio app.

Layout of the output dir: ``img<texture_id>.jpg`` + ``labels.json`` ({label: file}).
"""

from __future__ import annotations

import json
import os
from typing import Dict

import openpyxl
from PIL import Image

LABELS_JSON = "labels.json"


def load_texture_names(texture_list_path: str) -> Dict[int, str]:
    """texture_id -> English name (first two columns of texture_list.xlsx)."""
    wb = openpyxl.load_workbook(texture_list_path, data_only=True, read_only=True)
    try:
        rows = wb.active.iter_rows(min_row=2, values_only=True)
        return {int(r[0]): str(r[1]).strip() for r in rows if r[0] is not None and r[1] is not None}
    finally:
        wb.close()


def export_label_images(dataset_root: str, out_dir: str, size: int = 160, quality: int = 70) -> int:
    """Downscale images_scan_area/img<id>.jpg into out_dir. Duplicate names keep the lowest id."""
    names = load_texture_names(os.path.join(dataset_root, "texture_list.xlsx"))
    src_dir = os.path.join(dataset_root, "images", "images_scan_area")
    os.makedirs(out_dir, exist_ok=True)
    labels: Dict[str, str] = {}
    for tid in sorted(names):
        label = names[tid]
        src = os.path.join(src_dir, f"img{tid}.jpg")
        if label in labels or not os.path.isfile(src):
            continue
        fname = f"img{tid}.jpg"
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            im.save(os.path.join(out_dir, fname), quality=quality)
        labels[label] = fname
    with open(os.path.join(out_dir, LABELS_JSON), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=1)
    return len(labels)


def load_label_images(out_dir: str) -> Dict[str, str]:
    """label -> absolute image path; empty if the dir was never exported."""
    path = os.path.join(out_dir, LABELS_JSON)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return {k: os.path.join(out_dir, v) for k, v in labels.items()}
