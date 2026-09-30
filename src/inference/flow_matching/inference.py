"""Flow Matching inference engine."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.models.flow_matching.conditional_flow_matching import ConditionalFlowMatching
from src.models.shared.encodec_wrapper import EncodecWrapper
from src.models.shared.embeddings import ControlEmbedding, SinusoidalTimeEmbedding, LabelEmbedding
from src.models.shared.prev_segment_encoder import PrevSegmentEncoder
from src.utils.training.model_builder import build_prev_features
from src.utils.training.training_utils import build_control_condition
from src.inference.shared.base import BaseInference
from src.inference.flow_matching.cuda_graph import CudaGraphGenerator
from src.inference.flow_matching.generate import generate_segments

logger = logging.getLogger(__name__)


class FlowMatchingInference(BaseInference):
    """
    Lightweight generation state for FlowMatching model.

    ``init_from_model_dir`` is inherited from ``BaseInference``: meta load +
    ``Strategy.build`` + safetensors load + ``attach_to_bundle`` in one shot.
    """

    _expected_model_type = "flow_matching"

    def __init__(self) -> None:
        super().__init__()
        self.classes: List[str] = []
        self.time_embed = None
        self.label_embed = None
        self.encodec_wrapper: Optional[EncodecWrapper] = None
        self.model: Optional[ConditionalFlowMatching] = None
        self.generator: Optional[CudaGraphGenerator] = None
        self.gen_shape: Optional[torch.Size] = None
        self.cfg_path: Optional[str] = None
        self.ckpt_path: Optional[str] = None
        self.waveform_target_len: Optional[int] = None
        self.prev_segment_encoder: Optional[PrevSegmentEncoder] = None
        self.control_embedding = None
        self.prev_waveform_buffer: Optional[torch.Tensor] = None
        self.use_prev_segment: bool = False

    def _fix_waveform_length(self, wave: torch.Tensor) -> torch.Tensor:
        """
        Ensure waveform tensor has length == waveform_target_len.

        Args:
            wave: Tensor of shape [T] or [1, T] or [B, T].

        Returns:
            Tensor with the same leading dimensions but time length
            padded/truncated to self.waveform_target_len.
        """
        if self.waveform_target_len is None:
            return wave
        target_len = int(self.waveform_target_len)
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)  # [1, T]
        cur_len = int(wave.shape[-1])
        if cur_len < target_len:
            pad = target_len - cur_len
            wave = F.pad(wave, (0, pad))
        elif cur_len > target_len:
            wave = wave[..., :target_len]
        return wave

    def attach_to_bundle(
        self,
        *,
        cfg: Any,
        device: torch.device,
        full_dataset: Any,
        waveform_target_len: int,
        model: ConditionalFlowMatching,
        label_embed: LabelEmbedding,
        encodec_wrapper: EncodecWrapper,
        time_embed: Optional[SinusoidalTimeEmbedding] = None,
        prev_segment_encoder: Optional[PrevSegmentEncoder] = None,
        control_embedding: ControlEmbedding,
        use_prev_segment: bool = False,
        control_stats: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """Wire an already-built bundle into this engine (no disk I/O).

        Mirrors the post-checkpoint-load tail of ``init_from_model_dir``:
        sets every module reference, switches them to eval, runs the dummy
        encode that determines ``gen_shape``. After this call, the engine
        is usable through ``generate_from_control`` exactly like a
        disk-initialized one.
        """
        self._attach_common(
            cfg=cfg, device=device, full_dataset=full_dataset, control_stats=control_stats,
        )
        self.waveform_target_len = int(waveform_target_len)
        self.model = model
        self.generator = CudaGraphGenerator(model)
        self.label_embed = label_embed
        self.encodec_wrapper = encodec_wrapper
        self.time_embed = time_embed
        self.prev_segment_encoder = prev_segment_encoder
        self.control_embedding = control_embedding
        self.use_prev_segment = bool(use_prev_segment)

        for mod in (self.model, self.label_embed, self.time_embed,
                    self.prev_segment_encoder, self.control_embedding):
            if mod is not None:
                mod.eval()

        # Dummy-encode to fix gen_shape (encodec stride depends on input length;
        # this is the only safe way to obtain the per-step latent shape).
        dummy_wave = torch.zeros((1, int(waveform_target_len)), dtype=torch.float32)
        enc_out = encodec_wrapper.encode_waveform(
            dummy_wave,
            return_latent=True,
            return_codes=False,
            input_sample_rate=float(self.sample_rate),
        )
        latent = enc_out.get("latent")
        if latent is None:
            raise RuntimeError("Failed to encode dummy waveform to latent for shape inference.")
        self.gen_shape = torch.Size([1] + list(latent.shape[1:]))
        self.initialized = True

    def generate_once(
        self,
        label_value: str,
        steps: int,
        force_mean: float = 1.0,
        velocity_x: float = 0.0,
        velocity_y: float = 0.0,
        use_prev_segment_flag: Optional[bool] = None,
        prev_waveform: Optional[torch.Tensor] = None,
    ):
        """Generate one waveform segment."""
        assert self.initialized
        device = self.device
        encodec_wrapper = self.encodec_wrapper
        gen_shape = self.gen_shape
        target_len = int(self.waveform_target_len) if self.waveform_target_len is not None else None

        label_id = self._get_label_id(label_value)
        labels_tensor = torch.tensor([label_id], device=device)

        # Prepared outside inference_mode since it may load data from disk
        use_prev = use_prev_segment_flag if use_prev_segment_flag is not None else self.use_prev_segment
        prev_waveform_tensor = None
        if use_prev and self.prev_segment_encoder is not None:
            # An explicit prev_waveform takes priority; otherwise use the internal
            # buffer, then the dataset-based or zero initial waveform
            if prev_waveform is not None:
                if isinstance(prev_waveform, torch.Tensor):
                    prev_waveform_tensor = prev_waveform.to(device)
                else:
                    prev_waveform_tensor = torch.as_tensor(prev_waveform, dtype=torch.float32, device=device)
                if prev_waveform_tensor.dim() == 1:
                    prev_waveform_tensor = prev_waveform_tensor.unsqueeze(0)
            elif self.prev_waveform_buffer is not None:
                prev_waveform_tensor = self.prev_waveform_buffer.to(device)
                if prev_waveform_tensor.dim() == 1:
                    prev_waveform_tensor = prev_waveform_tensor.unsqueeze(0)
            else:
                if self.initial_prev_mode == "dataset":
                    prev_waveform_tensor = self._select_nearest_dataset_waveform(
                        label_id=label_id,
                        force_raw=force_mean,
                        vx_raw=velocity_x,
                        vy_raw=velocity_y,
                        target_len=int(self.waveform_target_len),
                        device=device,
                    )
                    if prev_waveform_tensor is None:
                        raise RuntimeError(
                            "Failed to select initial previous waveform from dataset "
                            f"(label_id={label_id}, stats_path={self._prev_stats_path})"
                        )
                else:
                    prev_waveform_tensor = torch.zeros(
                        (1, int(self.waveform_target_len)), device=device, dtype=torch.float32
                    )

        with torch.inference_mode():
            force_val_norm, vel_x_val_norm, vel_y_val_norm = self._normalize_control_values(
                force_mean, velocity_x, velocity_y
            )
            wave_t = generate_segments(
                model=self.generator,
                label_embed=self.label_embed,
                control_embedding=self.control_embedding,
                encodec_wrapper=encodec_wrapper,
                prev_segment_encoder=self.prev_segment_encoder if use_prev else None,
                labels=labels_tensor,
                force_mean=torch.tensor([[force_val_norm]], device=device, dtype=torch.float32),
                velocity_mean=torch.tensor([[vel_x_val_norm, vel_y_val_norm]], device=device, dtype=torch.float32),
                prev_waveform=prev_waveform_tensor,
                steps=int(steps),
                latent_shape=gen_shape,
                sample_rate=float(self.sample_rate),
                target_len=target_len,
                device=device,
            )[0]
            w = wave_t.detach().cpu().numpy().astype(np.float32)
        # Update internal prev waveform buffer only when external prev_waveform is not provided.
        if use_prev and prev_waveform is None:
            self.prev_waveform_buffer = torch.from_numpy(w).to(device)

        return w, int(self.sample_rate)

    def generate_from_control(
        self,
        velocity_x: float,
        velocity_y: float,
        force: float,
        label: int,
        method: Optional[int] = None,
        prev_waveform: Optional[torch.Tensor] = None,
    ):
        """
        Generate one waveform segment from (label, vx, vy, force).
        label is an int class index; steps comes from cfg.sampling. method is unused.
        """
        assert self.initialized and self.cfg is not None
        cfg = self.cfg

        label_val = int(label)

        steps_val = int(cfg.sampling.steps)
        
        if method is not None:
            logger.debug(f"Method parameter received: {method}")

        w, sr = self.generate_once(
            label_val,
            int(steps_val),
            force_mean=float(force),
            velocity_x=float(velocity_x),
            velocity_y=float(velocity_y),
            use_prev_segment_flag=self.use_prev_segment,
            prev_waveform=prev_waveform,
        )
        duration = float(len(w)) / float(sr) if sr > 0 else 0.0
        samples = [float(v) for v in w.reshape(-1)]
        return samples, sr, duration
