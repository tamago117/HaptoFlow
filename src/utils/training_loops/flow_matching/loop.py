"""Training loops for Flow Matching model."""
import time
from typing import Any, Dict, Optional, Tuple
import torch

from src.models.flow_matching.conditional_flow_matching import ConditionalFlowMatching
from src.models.shared.prev_segment_encoder import PrevSegmentEncoder
from src.models.shared.encodec_wrapper import EncodecWrapper
from src.models.shared.embeddings import LabelEmbedding, ControlEmbedding
from src.utils.training.training_losses import encode_latent_batch
from src.utils.training.model_builder import build_prev_features
from src.utils.data.data_augmentation import (
    apply_history_noise_and_mask,
    get_zero_prev_prob,
    shuffle_prev_within_batch,
    zero_prev_within_batch,
)
from src.utils.training.self_forcing import get_self_forcing_params
from src.utils.training_loops.flow_matching.self_forcing import build_self_forcing_batch, make_generate_fn
from src.utils.training.training_utils import (
    normalize_waveform_tensor,
    build_control_condition,
)


def train_step(
    batch: Dict[str, torch.Tensor],
    model: ConditionalFlowMatching,
    prev_segment_encoder: Optional[PrevSegmentEncoder],
    control_embedding: ControlEmbedding,
    label_embed: LabelEmbedding,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    encodec_wrapper: EncodecWrapper,
    device: torch.device,
    sample_rate: int,
    grad_clip: float,
    use_amp: bool,
    data_augmentation_cfg: Any = None,
    sampling_steps: int = 0,
    self_forcing_cfg: Any = None,
) -> Dict[str, float]:
    """Perform one training step.
    
    Args:
        batch: Training batch dictionary.
        model: Flow Matching model.
        prev_segment_encoder: Optional previous segment encoder.
        control_embedding: Control embedding.
        label_embed: Label embedding module.
        optimizer: Optimizer.
        scaler: Gradient scaler for mixed precision.
        encodec_wrapper: EncodecWrapper.
        device: Target device.
        sample_rate: Audio sample rate.
        grad_clip: Gradient clipping value.
        use_amp: Whether to use mixed precision.

    Returns:
        Dictionary of metrics and losses.
    """
    step_start = time.time()

    # Self-forcing (phase 2): flatten chains into (prev, target) pairs
    num_unroll, sf_prob, loss_reduction, loss_pairs = get_self_forcing_params(self_forcing_cfg)
    sf_generated = 0
    chain_batch = "chain_waveform" in batch
    if chain_batch:
        if prev_segment_encoder is None:
            raise ValueError("self_forcing needs model.use_prev_segment: true")
        if loss_reduction != "mean":
            raise ValueError("flow_matching self_forcing averages over the pairs; set loss_reduction: mean")
        generate_fn = make_generate_fn(
            model=model,
            label_embed=label_embed,
            control_embedding=control_embedding,
            prev_segment_encoder=prev_segment_encoder,
            encodec_wrapper=encodec_wrapper,
            steps=int(sampling_steps),
            sample_rate=float(sample_rate),
            device=device,
        )
        batch, sf_generated = build_self_forcing_batch(
            batch, generate_fn, sf_prob,
            # zero_prev acts on the chain start here, not on every pair
            zero_start_prob=get_zero_prev_prob(data_augmentation_cfg),
            loss_pairs=loss_pairs,
        )

    labels = batch["label_idx"].to(device)

    x0 = encode_latent_batch(batch["waveform"], encodec_wrapper, sample_rate, device)
    
    cond = label_embed(labels.long())
    
    cond = torch.cat([cond, build_control_condition(batch, control_embedding, device)], dim=-1)
    
    has_prev_mask_tensor = batch.get("has_prev_segment")

    # Raw previous waveform (augmented before encoding)
    prev_waveform_true = None
    if "prev_waveform" in batch:
        prev_waveform_raw = batch["prev_waveform"].to(device)
        # Shuffle prev_waveform within batch to simulate mismatched conditions
        shuffle_cfg = getattr(data_augmentation_cfg, "shuffle_prev", None) if data_augmentation_cfg is not None else None
        prev_waveform_raw, has_prev_mask_tensor = shuffle_prev_within_batch(
            prev_waveform_raw, has_prev_mask_tensor, shuffle_cfg, device
        )
        if not chain_batch:  # a chain batch applied zero_prev to its chain starts
            prev_waveform_raw = zero_prev_within_batch(prev_waveform_raw, data_augmentation_cfg, device)
        prev_waveform_true = normalize_waveform_tensor(prev_waveform_raw)
        prev_waveform_true = apply_history_noise_and_mask(prev_waveform_true, data_augmentation_cfg, device)

    prev_features = None
    if prev_segment_encoder is not None:
        prev_segment_dict: Dict[str, torch.Tensor] = {}
        if prev_waveform_true is not None:
            prev_segment_dict["latent"] = encode_latent_batch(
                prev_waveform_true.squeeze(1), encodec_wrapper, sample_rate, device
            )

        if prev_segment_dict:
            prev_features = build_prev_features(
                prev_segment_dict,
                has_prev_mask_tensor,
                prev_segment_encoder,
                device,
            )

    encoder_feats = prev_features
    
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', enabled=use_amp):
        out = model(x0, cond, encoder_feats=encoder_feats)
        loss = out["loss"]

    scaler.scale(loss).backward()
    if grad_clip > 0:
        scaler.unscale_(optimizer)
    clip_params = list(model.parameters())
    if prev_segment_encoder is not None:
        clip_params += list(prev_segment_encoder.parameters())
    clip_params += list(control_embedding.parameters())
    torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
    scaler.step(optimizer)
    scaler.update()
    
    lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
    t_tensor = out.get("t")
    t_mean = float(t_tensor.mean().item()) if t_tensor is not None else 0.0
    t_std = float(t_tensor.std().item()) if t_tensor is not None else 0.0
    v_pred = out.get("v_pred")
    v_pred_norm = float(v_pred.detach().float().norm().item()) if v_pred is not None else 0.0
    step_time_ms = (time.time() - step_start) * 1000.0
    mem_mb = 0.0
    if torch.cuda.is_available():
        mem_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
    
    return {
        "loss": float(loss.item()),
        "lr": lr,
        "t_mean": t_mean,
        "t_std": t_std,
        "v_pred_norm": v_pred_norm,
        "step_time_ms": step_time_ms,
        "gpu_mem_mb": mem_mb,
        "self_forcing_prob": sf_prob,
        "self_forcing_steps": num_unroll,
        "self_forcing_generated": sf_generated,
    }


def compute_evaluation_losses(
    batch: Dict[str, torch.Tensor],
    model: ConditionalFlowMatching,
    prev_segment_encoder: Optional[PrevSegmentEncoder],
    control_embedding: ControlEmbedding,
    label_embed: LabelEmbedding,
    encodec_wrapper: EncodecWrapper,
    device: torch.device,
    sample_rate: int,
) -> Tuple[float, float]:
    """Compute the evaluation loss for a batch.
    
    Returns:
        Tuple of (loss, infer_time_ms).
    """
    labels = batch["label_idx"].to(device)
    
    x0 = encode_latent_batch(batch["waveform"], encodec_wrapper, sample_rate, device)
    cond = label_embed(labels.long())

    cond = torch.cat([cond, build_control_condition(batch, control_embedding, device)], dim=-1)
    
    prev_features = None
    if prev_segment_encoder is not None and "prev_waveform" in batch:
        prev_features = build_prev_features(
            {"latent": encode_latent_batch(batch["prev_waveform"], encodec_wrapper, sample_rate, device)},
            batch.get("has_prev_segment"),
            prev_segment_encoder,
            device,
        )
        
    encoder_feats = prev_features
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_start = time.time()
    out = model(x0, cond, encoder_feats=encoder_feats)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_time_ms = (time.time() - infer_start) * 1000.0
    
    return float(out["loss"].item()), infer_time_ms
