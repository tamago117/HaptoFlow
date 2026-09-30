"""Per-epoch checkpoint save + symmetric load helpers shared by training and inference."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from safetensors.torch import save_file

from src.strategies import TrainingBundle
from src.utils.paths import to_project_relative


def _collect_module_tensors(bundle: TrainingBundle) -> Dict[str, torch.Tensor]:
    """Flatten bundle.model + nn.Module-typed extras into a prefixed tensor dict.

    Dedups by tensor identity so a module that aliases part of ``bundle.model``
    (e.g. an ``extras["unet"]`` that *is* ``bundle.model.backbone``) does not
    cause safetensors to refuse the save with "tensors share memory". On the
    load side, modules whose entries were skipped here are still populated
    because they share storage with the tensors loaded under ``model.*``.
    """
    tensors: Dict[str, torch.Tensor] = {}
    seen_ptrs: set[int] = set()

    def _add(prefix: str, module: Optional[nn.Module]) -> None:
        if module is None:
            return
        for k, v in module.state_dict().items():
            ptr = v.data_ptr()
            if ptr != 0 and ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            tensors[f"{prefix}.{k}"] = v

    _add("model", bundle.model)
    for name, value in bundle.extras.items():
        if isinstance(value, nn.Module):
            _add(name, value)
    return tensors


# Config paths stored repo-relative so a published meta.json leaks no local paths.
_PORTABLE_PATH_KEYS = (("data", "root"), ("data", "meta"), ("train", "init_from"))


def _portable_config(cfg: Any) -> Dict[str, Any]:
    config = OmegaConf.to_container(cfg, resolve=True)
    for section, key in _PORTABLE_PATH_KEYS:
        block = config.get(section)
        if isinstance(block, dict) and block.get(key):
            block[key] = to_project_relative(block[key])
    return config


def _build_meta(
    *,
    cfg: Any,
    model_type: str,
    full_dataset: Any,
    control_stats: Dict[str, Any],
    num_classes: int,
) -> Dict[str, Any]:
    waveform_length = getattr(full_dataset, "waveform_target_len", None)
    if waveform_length is None:
        waveform_length = getattr(full_dataset, "waveform_length", None)
    duration = (
        float(waveform_length) / float(full_dataset.sample_rate)
        if waveform_length is not None
        else None
    )

    return {
        "model_type": model_type,
        "config": _portable_config(cfg),
        "waveform_length": waveform_length,
        "duration": duration,
        "control_stats": control_stats,
        "dataset": {
            "sample_rate": float(full_dataset.sample_rate),
            "classes": list(full_dataset.classes),
            "label_to_int": dict(full_dataset.label_to_int),
            "num_classes": int(num_classes),
        },
        "encodec_io": {
            "encodec_model_name": str(getattr(cfg.encodec, "model_name", "24khz")) if hasattr(cfg, "encodec") else "none",
            "bandwidth_kbps": float(getattr(cfg.encodec, "bandwidth_kbps", 6.0)) if hasattr(cfg, "encodec") else 0.0,
        },
    }


def save_epoch_checkpoint(
    *,
    epoch: int,
    ckpt_dir: str,
    bundle: TrainingBundle,
    cfg: Any,
    model_type: str,
    full_dataset: Any,
    control_stats: Dict[str, Any],
    num_classes: int,
    dir_name: Optional[str] = None,
    global_step: int = 0,
    steps_per_epoch: int = 0,
    scaler: Optional[Any] = None,
    write_training_state: bool = True,
    ema: Optional[Any] = None,
) -> str:
    """Save model.safetensors + meta.json + training_state.pt under {ckpt_dir}/epoch_{epoch:04d}/.

    With ``ema`` (``ModelEMA``) the safetensors hold the EMA weights and the raw
    weights go to ``training_state.pt``.

    ``dir_name`` overrides the ``epoch_XXXX`` directory name — used by the
    memory-guard abort path, whose checkpoint is *not* a completed epoch and
    must not be wiped by the ``epoch_*`` cleanup at the start of a rerun.

    Returns the checkpoint directory path.
    """
    epoch_dir = os.path.join(ckpt_dir, dir_name or f"epoch_{epoch:04d}")
    os.makedirs(epoch_dir, exist_ok=True)

    tensors = dict(ema.shadow) if ema is not None else _collect_module_tensors(bundle)
    safetensors_path = os.path.join(epoch_dir, "model.safetensors")
    save_file(tensors, safetensors_path)

    meta = _build_meta(
        cfg=cfg,
        model_type=model_type,
        full_dataset=full_dataset,
        control_stats=control_stats,
        num_classes=num_classes,
    )
    meta_path = os.path.join(epoch_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    extra = ""
    if write_training_state:
        state_path = save_training_state(
            epoch_dir,
            bundle=bundle,
            completed_epochs=epoch,
            global_step=global_step,
            steps_per_epoch=steps_per_epoch,
            scaler=scaler,
            ema=ema,
        )
        extra = f", resume state: {state_path}"

    print(
        f"Saved checkpoint in {epoch_dir} (weights: {safetensors_path}, meta: {meta_path}{extra})",
        flush=True,
    )
    return epoch_dir


def load_meta_json(ckpt_dir: str) -> Dict[str, Any]:
    """Load ``ckpt_dir/meta.json`` and validate it has the required ``config`` key."""
    meta_path = os.path.join(ckpt_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in checkpoint dir: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ValueError(f"meta.json must contain a JSON object: {meta_path}")
    if "config" not in meta:
        raise ValueError(f"meta.json missing required key 'config': {meta_path}")
    return meta


def load_prefixed_state_dict(prefix: str, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract a single module's state dict from the flat safetensors tensor map.

    Inverse of ``_collect_module_tensors``: strips the ``<prefix>.`` from each
    key and returns ``{tail: value}`` ready for ``module.load_state_dict``.
    """
    out: Dict[str, torch.Tensor] = {}
    prefix_dot = prefix + "."
    for k, v in tensors.items():
        if k == prefix:
            continue
        if k.startswith(prefix_dot):
            out[k[len(prefix_dot):]] = v
    return out


def load_safetensors_into_bundle(
    ckpt_path: str,
    bundle: TrainingBundle,
    device: torch.device,
) -> None:
    """Symmetric counterpart of ``_collect_module_tensors`` + ``save_file``.

    Loads safetensors from ``ckpt_path`` and dispatches each module's slice
    into ``bundle.model`` (prefix ``model``) and each ``nn.Module``-typed
    entry in ``bundle.extras`` (prefix == extras key). Modules whose prefix
    is absent from the tensor file are left untouched — this matches the
    save path which only writes prefixes for modules that exist.
    """
    from safetensors.torch import load_file
    tensors = load_file(str(ckpt_path), device=str(device))

    def _maybe_load(prefix: str, module: Optional[nn.Module]) -> None:
        if module is None:
            return
        sub = load_prefixed_state_dict(prefix, tensors)
        if sub:
            module.load_state_dict(sub, strict=True)

    _maybe_load("model", bundle.model)
    for name, value in bundle.extras.items():
        if isinstance(value, nn.Module):
            _maybe_load(name, value)


# Optimizer / scheduler / scaler / RNG state lives in its own file: safetensors
# holds flat tensors only, and eval / serving never load it.
TRAINING_STATE_FILENAME = "training_state.pt"


def save_training_state(
    ckpt_path: str,
    *,
    bundle: TrainingBundle,
    completed_epochs: int,
    global_step: int,
    steps_per_epoch: int,
    scaler: Optional[Any] = None,
    ema: Optional[Any] = None,
) -> str:
    """Write ``training_state.pt`` next to the weights so training can resume.

    ``completed_epochs`` counts *fully finished* epochs; ``global_step`` may sit
    mid-epoch (an interrupt or memory abort), in which case the resumed run
    truncates its first epoch to the remaining steps.

    Only the torch CPU/CUDA RNG states are stored: ``runtime/setup.py`` seeds
    torch alone, and numpy's state would not survive ``weights_only=True``.
    """
    state: Dict[str, Any] = {
        "completed_epochs": int(completed_epochs),
        "global_step": int(global_step),
        "steps_per_epoch": int(steps_per_epoch),
        "optimizer": bundle.optimizer.state_dict(),
        "scheduler": bundle.scheduler.state_dict() if bundle.scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "rng": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "ema": ema.training_state() if ema is not None else None,
    }
    out_path = os.path.join(ckpt_path, TRAINING_STATE_FILENAME)
    torch.save(state, out_path)
    return out_path


def load_training_state(ckpt_dir: str) -> Dict[str, Any]:
    """Load ``training_state.pt`` from a checkpoint directory."""
    path = os.path.join(ckpt_dir, TRAINING_STATE_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{TRAINING_STATE_FILENAME} not found in {ckpt_dir}. This checkpoint "
            f"predates resume support (or was saved with resume state disabled); "
            f"it can be evaluated but not resumed."
        )
    return torch.load(path, map_location="cpu", weights_only=True)


def restore_training_state(
    state: Dict[str, Any],
    *,
    bundle: TrainingBundle,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> None:
    """Restore optimizer / scheduler / scaler / RNG from ``load_training_state``.

    Call this *after* the weights are loaded into the bundle and *after*
    ``build_runtime_environment`` (which calls ``torch.manual_seed``), but
    *before* the first ``iter(train_dl)`` — the sampler draws its seed there.
    """
    bundle.optimizer.load_state_dict(state["optimizer"])
    # load_state_dict already moves the AdamW moments to each param's device.

    if bundle.scheduler is not None and state.get("scheduler") is not None:
        # LambdaLR.state_dict() omits the lr_lambda closure: the scheduler
        # rebuilt from cfg keeps its lambda; only last_epoch / _step_count load.
        bundle.scheduler.load_state_dict(state["scheduler"])

    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])

    rng = state.get("rng") or {}
    cpu_state = rng.get("torch_cpu")
    if cpu_state is not None:
        torch.set_rng_state(cpu_state.to(torch.uint8).cpu())
    cuda_states = rng.get("torch_cuda") or []
    if cuda_states and torch.cuda.is_available():
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.to(torch.uint8).cpu() for s in cuda_states])
        else:
            print(
                f"[resume] CUDA device count changed "
                f"({len(cuda_states)} saved vs {torch.cuda.device_count()} now); "
                f"skipping CUDA RNG restore."
            )
