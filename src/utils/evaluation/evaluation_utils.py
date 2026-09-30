"""Evaluation utility functions."""
import itertools
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.flow_matching.conditional_flow_matching import ConditionalFlowMatching
from src.models.shared.prev_segment_encoder import PrevSegmentEncoder
from src.models.shared.encodec_wrapper import EncodecWrapper
from src.models.shared.embeddings import LabelEmbedding, ControlEmbedding
from src.utils.training.model_builder import build_prev_features
from src.utils.training.training_losses import encode_latent_batch
from src.utils.training.training_utils import (
    build_control_condition,
    compute_gfc,
    compute_freq_spectrum_rmse,
    _compute_freq_spectrum_from_waveform,
)
from src.utils.audio.encodec_audio import compute_snr_db


def _compute_metrics_from_waveforms(
    gen_waveforms: List[torch.Tensor],
    true_waveforms: List[torch.Tensor],
    sample_rate: int,
    stft_n_fft: int,
    stft_hop_length: int,
    stft_win_length: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Compute evaluation metrics from waveform pairs.
    
    Returns:
        Tuple of (snr_list, rmse_list, gfc_list, freq_spectrum_rmse_list).
    """
    snr_list = []
    rmse_list = []
    gfc_list = []
    freq_spectrum_rmse_list = []
    
    for gen_wf, true_wf in zip(gen_waveforms, true_waveforms):
        min_len = min(gen_wf.shape[-1], true_wf.shape[-1])
        gen_wf_aligned = gen_wf[..., :min_len]
        true_wf_aligned = true_wf[..., :min_len]
        
        snr = compute_snr_db(true_wf_aligned, gen_wf_aligned)
        snr_list.append(snr)
        
        if isinstance(gen_wf_aligned, torch.Tensor):
            gen_wf_np = gen_wf_aligned.squeeze().cpu().numpy()
        else:
            gen_wf_np = gen_wf_aligned.squeeze()
        if isinstance(true_wf_aligned, torch.Tensor):
            true_wf_np = true_wf_aligned.squeeze().cpu().numpy()
        else:
            true_wf_np = true_wf_aligned.squeeze()
        
        mse = np.mean((true_wf_np - gen_wf_np) ** 2)
        rmse = np.sqrt(mse)
        rmse_list.append(rmse)
        
        gfc = compute_gfc(
            true_wf_np,
            gen_wf_np,
            sample_rate,
            stft_n_fft,
            stft_hop_length,
            stft_win_length,
        )
        gfc_list.append(gfc)
        
        freq_spectrum_rmse = compute_freq_spectrum_rmse(
            true_wf_np,
            gen_wf_np,
            sample_rate,
            stft_n_fft,
            stft_hop_length,
            stft_win_length,
        )
        freq_spectrum_rmse_list.append(freq_spectrum_rmse)
    
    return snr_list, rmse_list, gfc_list, freq_spectrum_rmse_list


def compute_evaluation_metrics_on_dataloader(
    dataloader: DataLoader,
    model: ConditionalFlowMatching,
    prev_segment_encoder: Optional[PrevSegmentEncoder],
    control_embedding: Optional[ControlEmbedding],
    label_embed: LabelEmbedding,
    encodec_wrapper: EncodecWrapper,
    device: torch.device,
    sampling_steps: Optional[int] = None,
    stft_n_fft: int = 1024,
    stft_hop_length: int = 256,
    stft_win_length: int = 1024,
    sample_rate: int = 24000,
    return_per_sample_lists: bool = False,
    desc: Optional[str] = None,
    max_batches: Optional[int] = None,
) -> Union[Tuple[float, float, float, float], Tuple[float, float, float, float, List[float], List[float], List[float], List[float]]]:
    """Compute evaluation metrics (SNR, RMSE, GFC, freq_spectrum_rmse) on all batches in dataloader.
    
    Supports ConditionalFlowMatching on EnCodec latents.
    
    When return_per_sample_lists is True, returns (snr_mean, rmse_mean, gfc_mean, freq_spectrum_rmse_mean,
    snr_list, rmse_list, gfc_list, freq_spectrum_rmse_list) for histogram logging.
    
    Args:
        dataloader: DataLoader to iterate over.
        model: ConditionalFlowMatching model.
        prev_segment_encoder: Previous segment encoder (optional, for FlowMatching).
        control_embedding: Control embedding module (optional).
        label_embed: Label embedding module.
        encodec_wrapper: EnCodec wrapper (for FlowMatching).
        device: Target device.
        sampling_steps: Number of sampling steps (required).
        stft_n_fft: STFT FFT window size.
        stft_hop_length: STFT hop length.
        stft_win_length: STFT window length.
        sample_rate: Audio sample rate.
    
    Returns:
        Tuple of (snr_mean, rmse_mean, gfc_mean, freq_spectrum_rmse_mean).
    """
    model.eval()
    if prev_segment_encoder is not None:
        prev_segment_encoder.eval()
    if control_embedding is not None:
        control_embedding.eval()
    
    snr_list = []
    rmse_list = []
    gfc_list = []
    freq_spectrum_rmse_list = []
    
    total = len(dataloader) if hasattr(dataloader, "__len__") else None
    if max_batches is not None and max_batches > 0:
        effective_total = min(total, max_batches) if total is not None else max_batches
        source = itertools.islice(dataloader, max_batches)
    else:
        effective_total = total
        source = dataloader
    iterator = (
        tqdm(source, total=effective_total, desc=desc, unit="batch", dynamic_ncols=True, leave=False)
        if desc is not None
        else source
    )
    with torch.inference_mode():
        for batch in iterator:
            labels = batch["label_idx"].to(device)
            batch_size = labels.size(0)
            gen_waveforms = []
            cond = label_embed(labels.long())
            
            if control_embedding is not None:
                cond = torch.cat([cond, build_control_condition(batch, control_embedding, device)], dim=-1)
            
            if sampling_steps is None:
                raise ValueError("sampling_steps is required for ConditionalFlowMatching model")
            
            x0_true = encode_latent_batch(batch["waveform"], encodec_wrapper, sample_rate, device)
            gen_shape = torch.Size([x0_true.shape[0]] + list(x0_true.shape[1:]))
            
            gen_prev_features = None
            if prev_segment_encoder is not None and "prev_waveform" in batch:
                gen_prev_features = build_prev_features(
                    {"latent": encode_latent_batch(batch["prev_waveform"], encodec_wrapper, sample_rate, device)},
                    batch.get("has_prev_segment"),
                    prev_segment_encoder,
                    device,
                )
            
            gen_encoder_feats = gen_prev_features
            
            gen_continuous = model.generate(
                steps=sampling_steps,
                cond=cond,
                shape=gen_shape,
                encoder_feats=gen_encoder_feats,
            )
            
            gen_codes, _ = encodec_wrapper.quantize_latent(gen_continuous.detach())
            if gen_codes.dim() == 2:
                gen_codes = gen_codes.unsqueeze(0)
            
            gen_waveforms = []
            for i in range(gen_codes.shape[0]):
                codes_i = gen_codes[i]  # [Q, T]
                waveform_i = encodec_wrapper.decode_from_codes(codes_i, target_sample_rate=sample_rate)  # [1, T']
                gen_waveforms.append(waveform_i.cpu())
            
            if "waveform" in batch and len(gen_waveforms) > 0:
                true_waveforms = []
                for i in range(batch_size):
                    if i < len(gen_waveforms):
                        true_wf = batch["waveform"][i].cpu()
                        if true_wf.dim() == 1:
                            true_wf = true_wf.unsqueeze(0)
                        true_waveforms.append(true_wf)
                
                batch_snr_list, batch_rmse_list, batch_gfc_list, batch_freq_spectrum_rmse_list = _compute_metrics_from_waveforms(
                    gen_waveforms,
                    true_waveforms,
                    sample_rate,
                    stft_n_fft,
                    stft_hop_length,
                    stft_win_length,
                )
                snr_list.extend(batch_snr_list)
                rmse_list.extend(batch_rmse_list)
                gfc_list.extend(batch_gfc_list)
                freq_spectrum_rmse_list.extend(batch_freq_spectrum_rmse_list)
    
    snr_mean = sum(snr_list) / len(snr_list) if snr_list else 0.0
    rmse_mean = sum(rmse_list) / len(rmse_list) if rmse_list else 0.0
    gfc_mean = sum(gfc_list) / len(gfc_list) if gfc_list else 0.0
    freq_spectrum_rmse_mean = sum(freq_spectrum_rmse_list) / len(freq_spectrum_rmse_list) if freq_spectrum_rmse_list else 0.0
    
    if return_per_sample_lists:
        return snr_mean, rmse_mean, gfc_mean, freq_spectrum_rmse_mean, snr_list, rmse_list, gfc_list, freq_spectrum_rmse_list
    return snr_mean, rmse_mean, gfc_mean, freq_spectrum_rmse_mean
