from __future__ import annotations

import os
import sys
import io
import tempfile
import time
import json
from typing import Optional, Tuple, List, Dict, Any
import glob

import numpy as np
import gradio as gr
import yaml
from omegaconf import OmegaConf
import pandas as pd
import time as _time
from scipy.io import wavfile
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.flow_matching import FlowMatchingInference
from src.utils.hf_checkpoint import HF_MODEL_DIRS
from src.utils.meta.model_meta import load_model_meta
from src.utils.data.label_images import load_label_images


model = FlowMatchingInference()
# Live-updatable generation params (UI state)
model.cur_label = None
model.cur_steps = 3
model.cur_guidance = 1.0
model.cur_ymin = -1.0
model.cur_ymax = 1.0
model.show_fft = False
model.streaming = False
# Control parameters (raw, unnormalized units; used as defaults)
model.cur_force_mean = 1.0  # Default raw force value (N)
model.cur_velocity_x = 0.05  # Default raw velocity x (m/s)
model.cur_velocity_y = 0.0  # Default raw velocity y (m/s)
# Per-segment inference times (s) shared by Generate and Streaming; the Time panel shows the latest
model.infer_times = []
INFER_TIMES_MAX = 50
# Min interval between streaming UI updates; too short and the browser falls behind (Stop lags)
model.cur_ui_interval_s = 0.1


def record_infer_time(el: float) -> None:
    model.infer_times.append(float(el))
    del model.infer_times[:-INFER_TIMES_MAX]


def infer_time_df() -> pd.DataFrame:
    n = len(model.infer_times)
    return pd.DataFrame({"Iteration": np.arange(1, n + 1, dtype=np.int32),
                         "Time": np.asarray(model.infer_times, dtype=np.float32)})


MODEL_SEARCH_ROOT = "checkpoints"
# Written by scripts/dataset/export_label_images.py; missing dir = no thumbnails
LABEL_IMAGES_DIR = os.path.join(PROJECT_ROOT, "assets", "texture_label_images")
LABEL_IMAGES = load_label_images(LABEL_IMAGES_DIR)


def list_model_dirs() -> List[str]:
    """
    The published hf:// checkpoints, then local directories under MODEL_SEARCH_ROOT
    that this app can load: meta.json with a matching model_type plus at least one
    *.safetensors. Local paths are returned relative to PROJECT_ROOT.
    """
    expected = FlowMatchingInference._expected_model_type
    found: List[str] = []
    for meta_path in glob.glob(os.path.join(PROJECT_ROOT, MODEL_SEARCH_ROOT, "**", "meta.json"), recursive=True):
        d = os.path.dirname(meta_path)
        if not glob.glob(os.path.join(d, "*.safetensors")):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                model_type = json.load(f).get("model_type")
        except Exception:
            continue
        if model_type != expected:
            continue
        found.append(os.path.relpath(d, PROJECT_ROOT))
    return list(HF_MODEL_DIRS) + sorted(found)


def load_defaults_from_model_dir(model_dir: str) -> Tuple[str, int, float, bool]:
    """
    load label / steps / guidance from config.generate / config.sampling in meta.json in the model directory.
    The last value tells whether the checkpoint was trained with cond_dropout (needed for guidance).
    """
    model_dir = model_dir.strip()
    try:
        _, cfg, _, _ = load_model_meta(model_dir)
    except Exception:
        return "", 3, 1.0, False

    label = ""
    steps = 3
    if hasattr(cfg, "generate"):
        if getattr(cfg.generate, "label", None) is not None:
            label = str(cfg.generate.label)
    if hasattr(cfg, "sampling") and getattr(cfg.sampling, "steps", None) is not None:
        try:
            steps = int(cfg.sampling.steps)
        except Exception:
            pass
    has_null = float(getattr(getattr(cfg, "train", None), "cond_dropout", 0.0) or 0.0) > 0.0
    guidance = float(getattr(getattr(cfg, "sampling", None), "guidance_scale", 1.0) or 1.0) if has_null else 1.0
    return label, int(steps), guidance, has_null


def apply_guidance(value: float) -> None:
    """Classifier-free guidance is read from the model on every generate call."""
    model.cur_guidance = float(value)
    if model.model is not None:
        model.model.guidance_scale = model.cur_guidance


def set_seed_from_model_dir(model_dir: str) -> None:
    """
    Set random seed from config.seed in meta.json.
    Sets both torch and numpy random seeds for reproducibility.
    """
    model_dir = model_dir.strip()
    try:
        _, cfg, _, _ = load_model_meta(model_dir)
        seed = int(getattr(cfg, "seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Random seed set to {seed} from config")
    except Exception as e:
        print(f"Warning: Failed to set seed from config: {e}")
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)


def crossfade_concat(waves: List[np.ndarray], crossfade_samples: int) -> np.ndarray:
    if not waves:
        return np.array([], dtype=np.float32)
    out = waves[0].astype(np.float32)
    for w in waves[1:]:
        w = w.astype(np.float32)
        if crossfade_samples <= 0:
            out = np.concatenate([out, w], axis=0)
            continue
        cf = min(crossfade_samples, len(out), len(w))
        if cf <= 1:
            out = np.concatenate([out, w], axis=0)
            continue
        fade_out = np.linspace(1.0, 0.0, cf, dtype=np.float32)
        fade_in = 1.0 - fade_out
        left = out[-cf:] * fade_out
        right = w[:cf] * fade_in
        joined = left + right
        out = np.concatenate([out[:-cf], joined, w[cf:]], axis=0)
    return out


def do_generate(model_dir: str, label: str, steps: int, continuous: bool, segments: int, crossfade_ms: int,
                force_mean: Optional[float] = None, 
                velocity_x: Optional[float] = None, velocity_y: Optional[float] = None):
    model_dir = model_dir.strip()
    if not model.initialized or model.model_dir != model_dir:
        set_seed_from_model_dir(model_dir)
        msg, status = model.init_from_model_dir(model_dir)
        if status != "ok":
            raise RuntimeError(msg)
        apply_guidance(load_defaults_from_model_dir(model_dir)[2])
    else:
        set_seed_from_model_dir(model_dir)
    
    # Do not reset model.prev_waveform_buffer here, so "previous" carries across
    # separate button presses (calls to do_generate).
    
    force_val = force_mean if force_mean is not None else model.cur_force_mean
    vel_x_val = velocity_x if velocity_x is not None else model.cur_velocity_x
    vel_y_val = velocity_y if velocity_y is not None else model.cur_velocity_y
    # Temporarily apply the UI steps slider to sampling.steps
    orig_sampling_steps = None
    if getattr(model, "cfg", None) is not None and hasattr(model.cfg, "sampling"):
        try:
            orig_sampling_steps = int(getattr(model.cfg.sampling, "steps", int(steps)))
            model.cfg.sampling.steps = int(steps)
        except Exception:
            pass

    label_id = 0
    try:
        if hasattr(model, "label_to_int") and isinstance(model.label_to_int, dict):
            if label in model.label_to_int:
                label_id = int(model.label_to_int[label])
    except Exception:
        label_id = 0

    print("=" * 60)
    print("Model Input Conditions:")
    print(f"  Model directory: {model_dir}")
    print(f"  Label: {label}")
    print(f"  Steps: {steps}")
    print(f"  Guidance: {model.cur_guidance}")
    print(f"  Continuous: {continuous}")
    if continuous:
        print(f"  Segments: {segments}")
        print(f"  Crossfade: {crossfade_ms} ms")
    print(f"  Use previous segment: {model.use_prev_segment}")
    print(f"  Force mean: {force_val} N")
    print(f"  Velocity X: {vel_x_val} m/s")
    print(f"  Velocity Y: {vel_y_val} m/s")
    print("=" * 60)
    
    if continuous and segments > 1:
        parts: List[np.ndarray] = []
        sr = None
        for i in range(segments):
            t0 = time.time()
            samples, sr, _dur = model.generate_from_control(
                velocity_x=float(vel_x_val),
                velocity_y=float(vel_y_val),
                force=float(force_val),
                label=int(label_id),
                method=None,
            )
            t1 = time.time()
            el = float(t1 - t0)
            record_infer_time(el)
            w = np.asarray(samples, dtype=np.float32)
            parts.append(w)
        cf_samples = int((crossfade_ms / 1000.0) * (sr or 24000))
        final = crossfade_concat(parts, cf_samples)
        sample_rate = int(sr or 24000)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp_npy:
            np.save(tmp_npy, final)
            npy_path = tmp_npy.name
        
        wav_data = np.clip(final, -1.0, 1.0)
        wav_int16 = (wav_data * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            wavfile.write(tmp_wav.name, sample_rate, wav_int16)
            wav_path = tmp_wav.name
        
        return final, npy_path, wav_path, sr
    else:
        t0 = time.time()
        samples, sr, _dur = model.generate_from_control(
            velocity_x=float(vel_x_val),
            velocity_y=float(vel_y_val),
            force=float(force_val),
            label=int(label_id),
            method=None,
        )
        t1 = time.time()
        el = float(t1 - t0)
        record_infer_time(el)
        w = np.asarray(samples, dtype=np.float32)
        sample_rate = int(sr or 24000)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp_npy:
            np.save(tmp_npy, w)
            npy_path = tmp_npy.name
        
        wav_data = np.clip(w, -1.0, 1.0)
        wav_int16 = (wav_data * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            wavfile.write(tmp_wav.name, sample_rate, wav_int16)
            wav_path = tmp_wav.name
        
        # Restore sampling.steps
        if orig_sampling_steps is not None and getattr(model, "cfg", None) is not None and hasattr(model.cfg, "sampling"):
            try:
                model.cfg.sampling.steps = int(orig_sampling_steps)
            except Exception:
                pass
        return w, npy_path, wav_path, sr


# ---- Control pad: velocity (x, y) on a 2D pad + force on a vertical slider ----
# A single gr.HTML component whose value is {"vx", "vy", "force"} (raw units).
# The template renders from `value`; js_on_load moves the knobs directly while
# dragging and pushes `props.value` to the backend (throttled, final on release).
# The ranges are baked into the template/JS as literals: custom gr.HTML props are
# not delivered when the component starts hidden and is revealed later.
CONTROL_VEL_MAX = 0.06  # m/s, pad spans [-max, +max] on both axes
CONTROL_FORCE_MAX = 1.0  # N, slider spans [0, max]

CONTROL_PAD_HTML = """
<div class="cp">
  <div class="col">
    <div class="ttl">Velocity (m/s)</div>
    <div class="pad" title="Drag to set velocity, double-click to reset to (0, 0)">
      <div class="axis h"></div><div class="axis v"></div>
      <span class="tick l">-__VMAX__</span><span class="tick r">+__VMAX__</span>
      <span class="tick t">+__VMAX__</span><span class="tick b">-__VMAX__</span>
      <div class="knob" style="left:${(value.vx / __VMAX__ + 1) * 50}%;top:${(1 - value.vy / __VMAX__) * 50}%"></div>
    </div>
    <div class="ro">X <b class="rvx">${value.vx.toFixed(3)}</b>&nbsp;&nbsp;Y <b class="rvy">${value.vy.toFixed(3)}</b></div>
  </div>
  <div class="col">
    <div class="ttl">Force (N)</div>
    <div class="ftrack" title="Drag to set force">
      <div class="ffill" style="height:${value.force / __FMAX__ * 100}%"></div>
      <div class="fknob" style="bottom:${value.force / __FMAX__ * 100}%"></div>
    </div>
    <div class="ro"><b class="rf">${value.force.toFixed(2)}</b></div>
  </div>
</div>
""".replace("__VMAX__", repr(CONTROL_VEL_MAX)).replace("__FMAX__", repr(CONTROL_FORCE_MAX))

CONTROL_PAD_CSS = """
.cp { display: flex; gap: 20px; justify-content: center; padding: 10px 4px; user-select: none; }
.col { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.ttl { font-size: var(--block-label-text-size); color: var(--block-label-text-color); }
.ro { font-size: var(--text-sm); color: var(--body-text-color-subdued); font-variant-numeric: tabular-nums; }
.ro b { color: var(--body-text-color); font-weight: 600; }
.pad {
  position: relative; width: 200px; height: 200px; cursor: crosshair; touch-action: none;
  border: 1px solid var(--border-color-primary); border-radius: var(--radius-md);
  background-color: var(--background-fill-secondary);
  background-image:
    linear-gradient(var(--border-color-primary) 1px, transparent 1px),
    linear-gradient(90deg, var(--border-color-primary) 1px, transparent 1px);
  background-size: 25% 25%; background-position: -1px -1px;
}
.axis { position: absolute; background: var(--body-text-color-subdued); opacity: 0.6; }
.axis.h { left: 0; right: 0; top: 50%; height: 1px; }
.axis.v { top: 0; bottom: 0; left: 50%; width: 1px; }
.tick { position: absolute; font-size: 10px; color: var(--body-text-color-subdued); pointer-events: none; }
.tick.l { left: 3px; top: calc(50% + 2px); }
.tick.r { right: 3px; top: calc(50% + 2px); }
.tick.t { top: 2px; left: calc(50% + 4px); }
.tick.b { bottom: 2px; left: calc(50% + 4px); }
.knob {
  position: absolute; width: 16px; height: 16px; margin: -8px 0 0 -8px; border-radius: 50%;
  background: var(--color-accent); border: 2px solid white; box-shadow: 0 1px 3px rgba(0,0,0,.4);
  pointer-events: none;
}
.ftrack {
  position: relative; width: 14px; height: 200px; cursor: ns-resize; touch-action: none;
  border: 1px solid var(--border-color-primary); border-radius: 7px;
  background: var(--background-fill-secondary);
}
.ffill { position: absolute; left: 0; right: 0; bottom: 0; border-radius: 7px; background: var(--color-accent); opacity: 0.7; pointer-events: none; }
.fknob {
  position: absolute; left: 50%; width: 22px; height: 22px; margin: 0 0 -11px -11px; border-radius: 50%;
  background: var(--color-accent); border: 2px solid white; box-shadow: 0 1px 3px rgba(0,0,0,.4);
  pointer-events: none;
}
"""

CONTROL_PAD_JS = """
const VM = __VMAX__, FM = __FMAX__;
const VSTEP = 0.001, FSTEP = 0.01;
const cur = Object.assign({ vx: 0, vy: 0, force: 0 }, props.value || {});
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const snap = (v, s) => Number((Math.round(v / s) * s).toFixed(4));
const q = (sel) => element.querySelector(sel);
// Inside a gr.Group the HTML wrapper is transparent and its padding shows the
// group's gap colour; paint it like the neighbouring blocks.
const block = element.closest('.block');
if (block) block.style.background = 'var(--block-background-fill)';

function paint() {
  q('.knob').style.left = ((cur.vx / VM + 1) * 50) + '%';
  q('.knob').style.top = ((1 - cur.vy / VM) * 50) + '%';
  q('.ffill').style.height = (cur.force / FM * 100) + '%';
  q('.fknob').style.bottom = (cur.force / FM * 100) + '%';
  q('.rvx').textContent = cur.vx.toFixed(3);
  q('.rvy').textContent = cur.vy.toFixed(3);
  q('.rf').textContent = cur.force.toFixed(2);
}

let last = 0, timer = null;
function push() { last = performance.now(); props.value = { ...cur }; }
function commit(final) {
  if (timer) { clearTimeout(timer); timer = null; }
  if (final || performance.now() - last > 100) push();
  else timer = setTimeout(() => { timer = null; push(); }, 100);
}

function fromPad(e) {
  const r = q('.pad').getBoundingClientRect();
  const fx = clamp((e.clientX - r.left) / r.width, 0, 1);
  const fy = clamp((e.clientY - r.top) / r.height, 0, 1);
  cur.vx = snap((fx * 2 - 1) * VM, VSTEP);
  cur.vy = snap((1 - fy * 2) * VM, VSTEP);
}
function fromTrack(e) {
  const r = q('.ftrack').getBoundingClientRect();
  cur.force = snap(clamp((r.bottom - e.clientY) / r.height, 0, 1) * FM, FSTEP);
}

let mode = null;
element.addEventListener('pointerdown', (e) => {
  const pad = e.target.closest('.pad'), track = e.target.closest('.ftrack');
  if (!pad && !track) return;
  mode = pad ? fromPad : fromTrack;
  (pad || track).setPointerCapture(e.pointerId);
  mode(e); paint(); commit(false);
  e.preventDefault();
});
element.addEventListener('pointermove', (e) => {
  if (!mode) return;
  mode(e); paint(); commit(false);
});
const end = () => { if (mode) { mode = null; commit(true); } };
element.addEventListener('pointerup', end);
element.addEventListener('pointercancel', end);
element.addEventListener('dblclick', (e) => {
  if (!e.target.closest('.pad')) return;
  cur.vx = 0; cur.vy = 0; paint(); commit(true);
});
""".replace("__VMAX__", repr(CONTROL_VEL_MAX)).replace("__FMAX__", repr(CONTROL_FORCE_MAX))


with gr.Blocks(title="Haptics Waveform Generator") as demo:
    gr.Markdown("## Haptics Waveform Generator")
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=340):
            with gr.Group():
                model_dir = gr.Dropdown(
                    choices=list_model_dirs(),
                    value=HF_MODEL_DIRS[0],
                    label="Model directory (contains meta.json)",
                    allow_custom_value=True,
                )
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh", size="sm", min_width=80)
                    load_btn = gr.Button("Load model", size="sm", min_width=80)

            with gr.Group():
                with gr.Row(equal_height=True):
                    label = gr.Dropdown(choices=[], value=None, label="Label", allow_custom_value=True, scale=3)
                    label_img = gr.Image(
                        value=None, show_label=False, interactive=False, height=96, width=96,
                        container=False, scale=0, min_width=96, buttons=[], visible=bool(LABEL_IMAGES),
                    )
                control_pad = gr.HTML(
                    value={"vx": model.cur_velocity_x, "vy": model.cur_velocity_y, "force": model.cur_force_mean},
                    html_template=CONTROL_PAD_HTML,
                    css_template=CONTROL_PAD_CSS,
                    js_on_load=CONTROL_PAD_JS,
                    visible=False,
                )
            with gr.Group():
                steps = gr.Slider(1, 100, value=3, step=1, label="Steps")
                guidance = gr.Slider(
                    1.0, 5.0, value=1.0, step=0.1, label="Guidance (CFG)",
                    info="1.0 = off; needs a checkpoint trained with train.cond_dropout",
                )

            with gr.Tabs():
                with gr.Tab("Streaming"):
                    with gr.Row():
                        cf_stream_ms = gr.Slider(0, 200, value=50, step=10, label="Stream refresh (ms)")
                        max_buf = gr.Slider(1000, 30000, value=1000, step=1000, label="Max buffer (samples)")
                    ui_ms = gr.Slider(
                        20, 1000, value=100, step=10, label="UI update (ms)",
                        info="Lower = smoother; if Stop/Start lag, raise it",
                    )
                    with gr.Row():
                        stream_btn = gr.Button("Start streaming", variant="primary")
                        stop_btn = gr.Button("Stop", variant="stop")
                with gr.Tab("Generate"):
                    cont = gr.Checkbox(False, label="Continuous")
                    with gr.Row():
                        segs = gr.Slider(1, 200, value=10, step=1, label="Segments")
                        cf_ms = gr.Slider(0, 500, value=30, step=5, label="Crossfade (ms)")
                    gen_btn = gr.Button("Generate", variant="primary")
            streaming = gr.State(False)

            with gr.Accordion("Display", open=False):
                with gr.Row():
                    y_min = gr.Number(value=-1.0, label="Waveform Y min")
                    y_max = gr.Number(value=1.0, label="Waveform Y max")
                show_fft = gr.Checkbox(False, label="Show FFT Spectrum")

        with gr.Column(scale=2):
            out_plot = gr.LinePlot(label="Waveform", x="x", y="y", y_lim=(-1.0, 1.0))
            fft_plot = gr.LinePlot(label="FFT Spectrum", x="Frequency (Hz)", y="Magnitude", visible=False, x_lim=(0.0, 1000.0))
            with gr.Accordion("Inference time", open=False):
                time_plot = gr.LinePlot(label="Inference Time (s)", x="Iteration", y="Time")
            with gr.Accordion("Downloads", open=False):
                with gr.Row():
                    download_npy = gr.File(label="Download .npy")
                    download_wav = gr.File(label="Download .wav")

    def to_lineplot_data(w: np.ndarray):
        w = np.asarray(w).reshape(-1).astype(np.float32)
        sr = model.sample_rate if model.initialized else 24000
        if sr > 0:
            x = np.arange(len(w), dtype=np.float32) / float(sr)
        else:
            x = np.arange(len(w), dtype=np.float32)
        return pd.DataFrame({"x": x, "y": w})

    def to_fft_plot_data(w: np.ndarray):
        w = np.asarray(w).reshape(-1).astype(np.float32)
        if w.size == 0:
            return pd.DataFrame({"Frequency (Hz)": np.array([], dtype=np.float32), "Magnitude": np.array([], dtype=np.float32)})
        sr = model.sample_rate if model.initialized else 24000
        if sr <= 0:
            sr = 24000
        freqs = np.fft.rfftfreq(len(w), d=1.0 / float(sr))
        spectrum = np.abs(np.fft.rfft(w))
        return pd.DataFrame({"Frequency (Hz)": freqs.astype(np.float32), "Magnitude": spectrum.astype(np.float32)})

    def on_load_model(model_dir_path: str):
        """Load weights from model_dir and apply its defaults (label / steps) and feature-dependent UI."""
        model_dir_path = model_dir_path.strip()
        set_seed_from_model_dir(model_dir_path)
        lab, s, g, has_null = load_defaults_from_model_dir(model_dir_path)
        model.cur_label = lab
        model.cur_steps = int(s)
        msg, status = model.init_from_model_dir(model_dir_path)
        print(msg)
        guidance_update = gr.update(value=g, interactive=has_null)
        if status != "ok":
            hidden = gr.update(visible=False)
            return gr.update(choices=[], value=None), int(s), guidance_update, hidden
        apply_guidance(g)
        try:
            classes = list(model.classes)
        except Exception:
            classes = []
        label_value = lab if (lab and lab in classes) else (classes[0] if classes else None)
        model.cur_label = label_value
        control_visible = _update_ui_visibility()
        return (
            gr.update(choices=classes, value=label_value),
            int(s),
            guidance_update,
            gr.update(visible=control_visible),
        )

    load_model_outputs = [label, steps, guidance, control_pad]
    load_btn.click(fn=on_load_model, inputs=[model_dir], outputs=load_model_outputs)
    refresh_btn.click(fn=lambda: gr.update(choices=list_model_dirs()), inputs=None, outputs=[model_dir])

    # Live parameter updates: reflect to model.* for streaming
    def _set_label(l: Optional[str]):
        if l is None:
            return
        model.cur_label = str(l).strip()
    def _set_steps(v: int):
        try:
            model.cur_steps = int(v)
        except Exception:
            pass
    def _set_show_fft(flag: bool):
        model.show_fft = bool(flag)
        return gr.update(visible=model.show_fft)
    def _set_control(v: Optional[Dict[str, float]]):
        if not isinstance(v, dict):
            return
        try:
            model.cur_velocity_x = float(v["vx"])
            model.cur_velocity_y = float(v["vy"])
            model.cur_force_mean = float(v["force"])
        except (KeyError, TypeError, ValueError):
            pass
    def _update_y_lim(ymin: float, ymax: float):
        try:
            ymin_val = float(ymin)
            ymax_val = float(ymax)
        except Exception:
            return gr.update()
        if ymax_val <= ymin_val:
            return gr.update()
        model.cur_ymin = ymin_val
        model.cur_ymax = ymax_val
        return gr.update(y_lim=(ymin_val, ymax_val))
    def _update_ui_visibility():
        """Show the control pad only for models with a control embedding."""
        if not model.initialized:
            return False
        return model.control_embedding is not None

    label.change(fn=_set_label, inputs=[label], outputs=[])
    label.change(fn=LABEL_IMAGES.get, inputs=[label], outputs=[label_img])
    steps.change(fn=_set_steps, inputs=[steps], outputs=[])
    guidance.change(fn=apply_guidance, inputs=[guidance], outputs=[])
    ui_ms.change(fn=lambda v: setattr(model, "cur_ui_interval_s", float(v) / 1000.0), inputs=[ui_ms], outputs=[])
    y_min.change(fn=_update_y_lim, inputs=[y_min, y_max], outputs=[out_plot])
    y_max.change(fn=_update_y_lim, inputs=[y_min, y_max], outputs=[out_plot])
    show_fft.change(fn=_set_show_fft, inputs=[show_fft], outputs=[fft_plot])
    control_pad.change(fn=_set_control, inputs=[control_pad], outputs=[], trigger_mode="always_last")

    def on_generate(model_dir_path, lbl, s, c, n, cf, show_fft_flag, control):
        w, npy_path, wav_path, sr = do_generate(
            model_dir_path, lbl, s, c, n, cf, 
            float(control["force"]), float(control["vx"]), float(control["vy"])
        )
        waveform_update = gr.update(value=to_lineplot_data(w), y_lim=(model.cur_ymin, model.cur_ymax))
        if show_fft_flag:
            fft_df = to_fft_plot_data(w)
            fft_update = gr.update(value=fft_df, visible=True, x_lim=(0.0, 1000.0))
        else:
            fft_update = gr.update(visible=False)
        return waveform_update, fft_update, npy_path, wav_path, infer_time_df()

    gen_btn.click(
        fn=on_generate,
        inputs=[model_dir, label, steps, cont, segs, cf_ms, show_fft, control_pad],
        outputs=[out_plot, fft_plot, download_npy, download_wav, time_plot],
    )

    def start_stream(flag):
        model.streaming = True
        return True

    def stop_stream(flag):
        model.streaming = False
        return False

    def stream_generate(model_dir_path, lbl, s, refresh_ms, max_buffer_samples, streaming_flag):
        model_dir_path = model_dir_path.strip()
        if not model.initialized or model.model_dir != model_dir_path:
            set_seed_from_model_dir(model_dir_path)
            msg, status = model.init_from_model_dir(model_dir_path)
            if status != "ok":
                yield None, None, None, None, gr.update()
                return
            apply_guidance(load_defaults_from_model_dir(model_dir_path)[2])
        else:
            set_seed_from_model_dir(model_dir_path)
        # Cap buffer length to keep the plot light
        try:
            max_len = int(max(1000, float(max_buffer_samples)))
        except Exception:
            max_len = 100000
        buf = np.zeros((0,), dtype=np.float32)
        interval = max(0.0, float(refresh_ms) / 1000.0)

        def ui_update(buf: np.ndarray, show_fft_flag: bool):
            sr_for_time = model.sample_rate if model.initialized else 24000
            x = np.arange(len(buf), dtype=np.float32) / float(sr_for_time if sr_for_time > 0 else 1)
            df = pd.DataFrame({"x": x, "y": buf})
            if show_fft_flag:
                fft_update = gr.update(value=to_fft_plot_data(buf), visible=True, x_lim=(0.0, 1000.0))
            else:
                fft_update = gr.update(visible=False)
            return gr.update(value=df, y_lim=(model.cur_ymin, model.cur_ymax)), fft_update, None, None, infer_time_df()

        last_ui = 0.0
        # model.streaming is shared so the Stop button can interrupt between iterations
        while model.streaming:
            # Read the latest UI state (model.*) every iteration so slider changes
            # take effect on the next segment
            cur_label = model.cur_label if model.cur_label is not None else lbl
            cur_steps = model.cur_steps if model.cur_steps is not None else int(s)
            cur_show_fft = model.show_fft
            cur_force = model.cur_force_mean
            cur_vel_x = model.cur_velocity_x
            cur_vel_y = model.cur_velocity_y

            label_id = 0
            try:
                if hasattr(model, "label_to_int") and isinstance(model.label_to_int, dict):
                    if cur_label in model.label_to_int:
                        label_id = int(model.label_to_int[cur_label])
            except Exception:
                label_id = 0

            # Temporarily override sampling.steps with the streaming step count
            orig_sampling_steps = None
            if getattr(model, "cfg", None) is not None and hasattr(model.cfg, "sampling"):
                try:
                    orig_sampling_steps = int(getattr(model.cfg.sampling, "steps", int(cur_steps)))
                    model.cfg.sampling.steps = int(cur_steps)
                except Exception:
                    pass

            t0 = time.time()
            samples, sr, _dur = model.generate_from_control(
                velocity_x=float(cur_vel_x),
                velocity_y=float(cur_vel_y),
                force=float(cur_force),
                label=int(label_id),
                method=None,
            )
            t1 = time.time()
            el = float(t1 - t0)

            if orig_sampling_steps is not None and getattr(model, "cfg", None) is not None and hasattr(model.cfg, "sampling"):
                try:
                    model.cfg.sampling.steps = int(orig_sampling_steps)
                except Exception:
                    pass

            w = np.asarray(samples, dtype=np.float32)
            buf = np.concatenate([buf, w], axis=0)
            if len(buf) > max_len:
                buf = buf[-max_len:]
            record_infer_time(el)
            # Throttle UI updates: the browser falls behind if every segment is pushed
            if time.time() - last_ui >= model.cur_ui_interval_s:
                last_ui = time.time()
                yield ui_update(buf, cur_show_fft)
            if interval > 0:
                _time.sleep(interval)
        yield ui_update(buf, model.show_fft)

    stream_evt = stream_btn.click(fn=start_stream, inputs=[streaming], outputs=[streaming]).then(
        fn=stream_generate,
        inputs=[model_dir, label, steps, cf_stream_ms, max_buf, streaming],
        outputs=[out_plot, fft_plot, download_npy, download_wav, time_plot],
    )
    # cancels ends the pending stream event, so Start works again right away
    stop_btn.click(fn=stop_stream, inputs=[streaming], outputs=[streaming], cancels=[stream_evt])

    def on_app_load():
        model.show_fft = False
        return on_load_model(model_dir.value)

    demo.load(fn=on_app_load, inputs=None, outputs=load_model_outputs)

if __name__ == "__main__":
    demo.queue().launch(allowed_paths=[LABEL_IMAGES_DIR])


