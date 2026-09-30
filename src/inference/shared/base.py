"""
Base interface class for model interfaces.
Contains common functionality shared across all model interface implementations.
"""
from __future__ import annotations

import glob
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from src.inference.shared.initial_prev_utils import build_prev_waveform_index_from_stats
from src.utils.hf_checkpoint import resolve_model_dir
from src.utils.paths import resolve_project_path
from src.utils.meta.model_meta import (
    load_model_meta,
    extract_dataset_meta,
    extract_control_stats,
)

logger = logging.getLogger(__name__)

# Initial previous waveform at inference start (when no prev buffer exists yet):
# - "zero": zero-padding
# - "dataset": nearest segment waveform from the dataset's stats.jsonl
INITIAL_PREV_MODES = ("zero", "dataset")


class BaseInference:
    """
    Base class for all model interfaces.
    Provides common initialization and utility methods.
    """

    def __init__(self) -> None:
        """Initialize common attributes."""
        self.initialized: bool = False
        self.model_dir: Optional[str] = None
        self.device: Optional[torch.device] = None
        self.cfg = None
        self.sample_rate: int = 24000
        self.label_to_int: Dict[str, int] = {}
        self.control_stats: Dict[str, Dict[str, float]] = {}
        self.initial_prev_mode = "zero"
        # Dataset-based previous waveform index (for initial_prev_mode == "dataset")
        self.prev_index_ratio: float = 0.1
        self._prev_index_built: bool = False
        self._prev_stats_path: Optional[str] = None
        self._prev_dataset_root: Optional[str] = None
        self._prev_index_labels: Optional[np.ndarray] = None
        self._prev_index_controls: Optional[np.ndarray] = None
        self._prev_index_windows: Optional[List[Tuple[str, int]]] = None

    # Subclasses can override to assert the meta.json:model_type matches.
    _expected_model_type: Optional[str] = None


    @property
    def initial_prev_mode(self) -> str:
        return self._initial_prev_mode

    @initial_prev_mode.setter
    def initial_prev_mode(self, mode: str) -> None:
        if mode not in INITIAL_PREV_MODES:
            raise ValueError(f"initial_prev_mode must be one of {INITIAL_PREV_MODES}, got {mode!r}")
        self._initial_prev_mode = mode
    def init_from_model_dir(self, model_dir: str) -> Tuple[str, str]:
        """Load a saved checkpoint and become a fully-initialized inference engine.

        Uses the same code path as training-time visualization:
        ``Strategy.build()`` constructs the model graph from cfg + a
        metadata-only dataset view; weights are loaded into the bundle;
        ``Strategy.build_inference()`` then attaches the bundle to a fresh
        engine, whose state is copied into ``self`` so the caller's reference
        stays valid.
        """
        # Local imports avoid a circular import between BaseInference (loaded
        # by Strategy.build_inference) and the strategies registry.
        from src.strategies import BuildContext, StepContext, get as get_strategy
        from src.utils.meta.dataset_view import MetadataOnlyDatasetView
        from src.utils.training.checkpoint import load_safetensors_into_bundle

        model_dir = model_dir.strip()
        if self.initialized and self.model_dir == model_dir:
            return (f"Already initialized: {self.device}", "ok")

        try:
            # hf:// specs load from the local cache; self.model_dir keeps the spec
            local_dir = resolve_model_dir(model_dir)
            meta, cfg, _wtl, meta_path = self._load_model_meta(local_dir)
        except Exception as e:
            return (f"Failed to load model meta from {model_dir}: {e}", "error")

        model_type = meta.get("model_type")
        if model_type is None:
            return ("meta.json missing 'model_type'; cannot route to a strategy.", "error")
        if self._expected_model_type and str(model_type) != self._expected_model_type:
            return (
                f"meta.json model_type={model_type!r} does not match "
                f"{type(self).__name__} (expects {self._expected_model_type!r}).",
                "error",
            )

        ckpt_path, err = self._find_ckpt_from_model_dir(local_dir)
        if err is not None or ckpt_path is None:
            return (err or "Checkpoint path resolution failed.", "error")

        torch.manual_seed(int(cfg.seed))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        view = MetadataOnlyDatasetView.from_meta(meta)

        strategy = get_strategy(str(model_type))
        dparams = strategy.dataset_params(cfg)
        encodec_wrapper = None
        if dparams.use_encodec:
            from src.models.shared.encodec_wrapper import EncodecWrapper
            encodec_io = meta.get("encodec_io", {}) or {}
            encodec_wrapper = EncodecWrapper(
                model_name=str(encodec_io.get("encodec_model_name", cfg.encodec.model_name)),
                bandwidth=float(
                    encodec_io.get("bandwidth_kbps", getattr(cfg.encodec, "bandwidth_kbps", 6.0))
                ),
                device=device,
            )

        bundle = strategy.build(BuildContext(
            cfg=cfg,
            device=device,
            full_dataset=view,
            encodec_wrapper=encodec_wrapper,
        ))

        try:
            load_safetensors_into_bundle(str(ckpt_path), bundle, device)
        except Exception as e:
            return (f"Failed to load weights from {ckpt_path}: {e}", "error")

        control_stats = self._get_control_stats_safe(meta, meta_path)
        ctx = StepContext(
            cfg=cfg, device=device, full_dataset=view,
            encodec_wrapper=encodec_wrapper,
            extras={"control_stats": control_stats},
        )
        fresh = strategy.build_inference(bundle, ctx)

        # Copy the fresh engine's state into ``self`` so external callers that
        # hold a reference to ``self`` see the initialized state.
        for k, v in vars(fresh).items():
            setattr(self, k, v)
        # Bookkeeping that attach_to_bundle doesn't perform (only relevant when
        # the engine was constructed from a disk checkpoint).
        self.model_dir = model_dir
        if hasattr(self, "cfg_path"):
            self.cfg_path = os.path.join(local_dir, "meta.json")
        if hasattr(self, "ckpt_path"):
            self.ckpt_path = str(ckpt_path)

        return (f"Initialized on {self.device}", "ok")

    def _attach_common(
        self,
        *,
        cfg: Any,
        device: torch.device,
        full_dataset: Any,
        control_stats: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """Populate state shared across all ``*Inference.attach_to_bundle()``.

        Sets cfg, device, sample_rate, classes/label_to_int (from ``full_dataset``),
        control_stats, and the dataset-prev index location. ``model_dir`` stays None
        because no on-disk checkpoint is involved when attaching to a live bundle.
        """
        self.cfg = cfg
        self.device = device
        self.sample_rate = int(getattr(full_dataset, "sample_rate", self.sample_rate))
        classes = list(getattr(full_dataset, "classes", []) or [])
        self.label_to_int = {str(c): i for i, c in enumerate(classes)}
        if hasattr(self, "classes"):
            self.classes = classes
        self.control_stats = dict(control_stats) if control_stats else {}
        data_root = None
        try:
            data_root = str(cfg.data.root)
        except Exception:
            data_root = None
        if data_root:
            self._init_prev_dataset_from_root(data_root)

    @staticmethod
    def _find_ckpt_in_model_dir(model_dir: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Find checkpoint file in model directory.

        Only *.safetensors files are supported; multiple candidates are an error.
        """
        st_pattern = os.path.join(model_dir, "*.safetensors")
        st_candidates = sorted(glob.glob(st_pattern))
        if len(st_candidates) == 0:
            return None, f"Checkpoint not found in {model_dir} (*.safetensors)"
        if len(st_candidates) > 1:
            return None, (
                f"Multiple checkpoints in {model_dir}: "
                f"{', '.join(os.path.basename(p) for p in st_candidates)}"
            )
        return st_candidates[0], None

    def _find_ckpt_from_model_dir(self, model_dir: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolve checkpoint path by searching for *.safetensors directly under model_dir.
        """
        return self._find_ckpt_in_model_dir(model_dir)

    def _load_model_meta(self, model_dir: str) -> Tuple[Any, Any, int, str]:
        """
        Load model metadata from model directory.
        Returns (meta, cfg, waveform_target_len, meta_path).
        """
        return load_model_meta(model_dir)

    def _extract_dataset_meta(self, meta: Dict[str, Any], meta_path: str) -> Tuple[List[str], Dict[str, int], int, int]:
        """
        Extract dataset metadata from meta dict.
        Returns (classes, label_to_int, dataset_sample_rate, num_classes).
        """
        return extract_dataset_meta(meta, meta_path)

    @staticmethod
    def _get_control_stats_safe(meta: Dict[str, Any], meta_path: str) -> Dict[str, Dict[str, float]]:
        """Return control_stats from meta, or empty dict (caller uses 0 mean, 1 std)."""
        if "control_stats" not in meta or not isinstance(meta.get("control_stats"), dict):
            return {}
        try:
            return extract_control_stats(meta, meta_path)
        except Exception:
            return {}

    @staticmethod
    def _extract_module_state(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        """Extract state dict for a module by removing the prefix."""
        out = {}
        prefix_dot = f"{prefix}."
        for k, v in state.items():
            if k.startswith(prefix_dot):
                out[k[len(prefix_dot):]] = v
        return out

    def _load_checkpoint_state(self, ckpt_path: str) -> Dict[str, torch.Tensor]:
        """Load checkpoint state dict from a .safetensors file."""
        ckpt_path_str = str(ckpt_path)
        ext = os.path.splitext(ckpt_path_str)[1].lower()

        if ext != ".safetensors":
            raise ValueError(
                f"Unsupported checkpoint extension '{ext}' for {ckpt_path_str}; "
                "only .safetensors is supported."
            )

        return load_file(ckpt_path_str, device=str(self.device))

    def _normalize_control_values(
        self,
        force_val: float,
        velocity_x: float,
        velocity_y: float,
    ) -> Tuple[float, float, float]:
        """
        Normalize control values using control_stats.
        Returns (force_norm, vx_norm, vy_norm).
        """
        fs = self.control_stats.get("force", {})
        vxs = self.control_stats.get("velocity_x", {})
        vys = self.control_stats.get("velocity_y", {})
        force_n = (force_val - float(fs.get("mean", 0.0))) / (float(fs.get("std", 1.0)) or 1.0)
        vx_n = (velocity_x - float(vxs.get("mean", 0.0))) / (float(vxs.get("std", 1.0)) or 1.0)
        vy_n = (velocity_y - float(vys.get("mean", 0.0))) / (float(vys.get("std", 1.0)) or 1.0)
        return force_n, vx_n, vy_n

    def _get_label_id(self, label_value: Any) -> int:
        """Convert a label value (string, number, etc.) to a label ID."""
        key = str(label_value)
        if key in self.label_to_int:
            return self.label_to_int[key]
        return int(label_value)

    # -------------------------------------------------------------------------
    # Dataset-based previous waveform utilities (shared by subclasses)
    # -------------------------------------------------------------------------

    def _init_prev_dataset_from_root(self, dataset_root: str) -> None:
        """
        Initialize dataset-based previous waveform info from dataset root.

        Expects stats.jsonl under the root directory.
        """
        dataset_root = resolve_project_path(dataset_root)
        stats_path = os.path.join(dataset_root, "stats.jsonl")
        if os.path.exists(stats_path):
            self._prev_stats_path = stats_path
            self._prev_dataset_root = dataset_root

    def _ensure_prev_dataset_index(self) -> None:
        """Build index from stats.jsonl if needed (for initial_prev_mode == 'dataset')."""
        if self._prev_index_built:
            return
        if self._prev_stats_path is None or self._prev_dataset_root is None:
            return

        try:
            labels, controls, windows = build_prev_waveform_index_from_stats(
                self._prev_stats_path,
                self.label_to_int,
                self.control_stats,
                subsample_ratio=self.prev_index_ratio,
            )
        except Exception as e:
            logger.warning("Failed to build prev dataset index: %s", e)
            return

        if labels.shape[0] == 0:
            logger.info(
                "Prev dataset index is empty (stats=%s, ratio=%.3f)",
                self._prev_stats_path,
                self.prev_index_ratio,
            )
            return

        self._prev_index_labels = labels
        self._prev_index_controls = controls
        self._prev_index_windows = windows
        self._prev_index_built = True

    def _select_nearest_dataset_waveform(
        self,
        label_id: int,
        force_raw: float,
        vx_raw: float,
        vy_raw: float,
        target_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Select nearest waveform from dataset for initial previous waveform.

        This uses:
          - label_id (int)
          - control values (force_raw, vx_raw, vy_raw)
          - z-score normalization with self.control_stats
          - pre-built index from stats.jsonl

        target_len: final waveform length in samples.
        device: target torch.device.
        """
        if not self._prev_index_built:
            self._ensure_prev_dataset_index()
        if (
            not self._prev_index_built
            or self._prev_index_labels is None
            or self._prev_index_controls is None
            or self._prev_index_windows is None
            or self._prev_dataset_root is None
        ):
            return None

        t0 = time.time()

        # Normalize query controls with the same stats used at training time
        f_n, vx_n, vy_n = self._normalize_control_values(force_raw, vx_raw, vy_raw)
        q = np.asarray([f_n, vx_n, vy_n], dtype=np.float32)

        labels = self._prev_index_labels
        controls = self._prev_index_controls

        mask = labels == int(label_id)
        if not np.any(mask):
            logger.info("No prev dataset candidates for label_id=%s", label_id)
            return None

        cand_controls = controls[mask]
        diff = cand_controls - q[None, :]
        dist2 = np.sum(diff * diff, axis=1)
        local_idx = int(np.argmin(dist2))
        global_indices = np.nonzero(mask)[0]
        global_idx = int(global_indices[local_idx])

        rel_path, start = self._prev_index_windows[global_idx]
        parquet_path = os.path.join(self._prev_dataset_root, rel_path)

        try:
            amp = pd.read_parquet(parquet_path, columns=["amplitude"])["amplitude"].to_numpy(dtype=np.float32)
        except Exception as e:
            logger.warning("Failed to read parquet for prev dataset waveform: %s", e)
            return None

        target_len = int(target_len)
        wave = np.zeros((target_len,), dtype=np.float32)
        window = amp[start:start + target_len]
        wave[: window.size] = window

        elapsed = time.time() - t0
        print(
            f"[prev_dataset_search] search time: {elapsed:.3f} s, "
            f"label_id={label_id}, path={parquet_path}, start={start}"
        )

        # [1, L]
        tensor = torch.from_numpy(wave).unsqueeze(0).to(device, dtype=torch.float32)
        return tensor

    def generate_from_control(
        self,
        velocity_x: float,
        velocity_y: float,
        force: float,
        label: int,
        method: Optional[int] = None,
        prev_waveform: Optional[torch.Tensor] = None,
    ) -> Tuple[List[float], int, float]:
        """
        Generate waveform from control parameters.
        Default implementation that calls generate_once.
        Subclasses can override if needed.

        Args:
            velocity_x: Control value (x velocity).
            velocity_y: Control value (y velocity).
            force: Control value (force).
            label: Label id or label value.
            method: Unused.
            prev_waveform: Optional previous waveform segment to condition on.
                If provided, subclasses can use this instead of internal buffers
                or dataset-based previous segments.
        """
        w, sr = self.generate_once(
            label,
            1,
            force_mean=float(force),
            velocity_x=float(velocity_x),
            velocity_y=float(velocity_y),
            use_prev_segment_flag=None,
            prev_waveform=prev_waveform,
        )
        duration = float(len(w)) / float(sr) if sr > 0 else 0.0
        return [float(x) for x in w.reshape(-1)], sr, duration

    def generate_once(
        self,
        label_value: str,
        steps: int,
        force_mean: Optional[float] = None,
        velocity_x: Optional[float] = None,
        velocity_y: Optional[float] = None,
        use_prev_segment_flag: Optional[bool] = None,
        prev_waveform: Optional[torch.Tensor] = None,
    ):
        """
        Generate one segment of waveform.
        Must be implemented by subclasses.

        Args:
            label_value: Label value (string / number).
            steps: Number of sampling steps.
            force_mean: Optional force mean value.
            velocity_x: Optional x velocity.
            velocity_y: Optional y velocity.
            use_prev_segment_flag: Optional flag to enable/disable previous segment.
            prev_waveform: Optional raw previous waveform tensor to condition on.
        """
        raise NotImplementedError("Subclasses must implement generate_once")
