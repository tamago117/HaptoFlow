"""EnCodec latent encoding for Flow Matching training batches."""
import torch
from src.models.shared.encodec_wrapper import EncodecWrapper


def encode_latent_batch(
    waveform: torch.Tensor,
    encodec_wrapper: EncodecWrapper,
    sample_rate: float,
    device: torch.device,
) -> torch.Tensor:
    """Encode a batch of dataset waveforms to the EnCodec latent on ``device``.

    Args:
        waveform: Waveforms at ``sample_rate``, shaped [B, T], [B, 1, T] or [T].
        encodec_wrapper: EnCodec wrapper (its device does the resampling + encoding).
        sample_rate: Sample rate of ``waveform`` (the dataset's sample rate).
        device: Device for the returned latent.

    Returns:
        Latent tensor [B, D, T'].
    """
    outputs = encodec_wrapper.encode_waveform(
        waveform,
        return_latent=True,
        return_codes=False,
        input_sample_rate=float(sample_rate),
        move_to_cpu=False,
    )
    return outputs["latent"].to(device)
