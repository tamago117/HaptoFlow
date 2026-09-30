"""Self-forcing (phase-2) batches for Flow Matching.

A chain ``s_0 … s_K`` becomes the pairs ``(s_{t-1}, s_t)``; for samples picked
with ``prob`` the prevs ``s_1 … s_{K-1}`` are the model's own generations.
The chain start is zeroed with ``zero_start_prob`` (``zero_prev``).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

# (labels [n], force_mean [n, 1], velocity_mean [n, 2], prev_waveform [n, L]) -> [n, L]
GenerateFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def build_self_forcing_batch(
    batch: Dict[str, torch.Tensor],
    generate_fn: GenerateFn,
    prob: float,
    zero_start_prob: float = 0.0,
    loss_pairs: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    with_rollout: bool = False,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Flatten a chain batch into (prev, target) pairs with generated prevs.

    Args:
        batch: Dataset batch with ``chain_waveform`` [B, K+1, L],
            ``chain_force_mean`` [B, K+1, 1], ``chain_velocity_mean`` [B, K+1, 2]
            and ``label_idx`` [B].
        generate_fn: Generates one segment per row from its prev and controls.
        prob: Per-sample probability of feeding generated prevs.
        zero_start_prob: Per-sample probability that the chain starts from a zero prev.
        loss_pairs: Keep this many random pairs per chain (None = all K).
        generator: RNG for the zero-start / generated / pair choices (validation
            passes a fixed one so every epoch sees the same prevs).
        with_rollout: Also return, for the P chains with generated prevs, the
            generated chain ``rollout`` [P, K, L] (s_1 … s_K, each from the previous
            generation), its ``rollout_truth`` [P, K, L], ``rollout_start`` [P, L]
            (zero or true prev of s_1) and ``rollout_label_idx`` [P].

    Returns:
        (flat batch of B*M pairs with the keys the Flow Matching step reads,
        number of samples that used generated prevs).
    """
    chain = batch["chain_waveform"]
    force = batch["chain_force_mean"]
    velocity = batch["chain_velocity_mean"]
    labels = batch["label_idx"]
    B, K1, L = chain.shape
    K = K1 - 1

    # prevs[:, j] is the prev of target s_{j+1}
    prevs = chain[:, :-1].clone()
    prevs[torch.rand(B, generator=generator) < zero_start_prob, 0] = 0.0
    picked = torch.nonzero(torch.rand(B, generator=generator) < prob).squeeze(1)
    if K > 1 and picked.numel() > 0:
        prev = prevs[picked, 0]
        for t in range(1, K):
            generated = generate_fn(labels[picked], force[picked, t], velocity[picked, t], prev)
            generated = generated.to(prevs.device, prevs.dtype)
            prevs[picked, t] = generated
            prev = generated

    rollout = {}
    if with_rollout and picked.numel() > 0:
        last = generate_fn(labels[picked], force[picked, K], velocity[picked, K], prevs[picked, K - 1])
        rollout = {
            "rollout": torch.cat([prevs[picked, 1:], last.to(prevs.device, prevs.dtype).unsqueeze(1)], dim=1),
            "rollout_truth": chain[picked, 1:],
            "rollout_start": prevs[picked, 0],
            "rollout_label_idx": labels[picked],
        }
        if "chain_force_mean_raw" in batch:
            rollout["rollout_force_mean_raw"] = batch["chain_force_mean_raw"][picked, 1:].mean(dim=1)
            rollout["rollout_velocity_mean_raw"] = batch["chain_velocity_mean_raw"][picked, 1:].mean(dim=1)

    per_step = {"waveform": chain[:, 1:], "prev_waveform": prevs, "force_mean": force[:, 1:],
                "velocity_mean": velocity[:, 1:]}
    if "chain_force_mean_raw" in batch:
        per_step["force_mean_raw"] = batch["chain_force_mean_raw"][:, 1:]
        per_step["velocity_mean_raw"] = batch["chain_velocity_mean_raw"][:, 1:]
    M = K if loss_pairs is None else max(1, min(int(loss_pairs), K))
    if M < K:
        keep = torch.rand(B, K, generator=generator).argsort(dim=1)[:, :M]
        take = lambda x: torch.gather(x, 1, keep.view(B, M, *[1] * (x.dim() - 2)).expand(B, M, *x.shape[2:]))
        per_step = {k: take(v) for k, v in per_step.items()}

    flat = {k: v.reshape(B * M, *v.shape[2:]) for k, v in per_step.items()}
    flat["label_idx"] = labels.repeat_interleave(M)
    flat["has_prev_segment"] = torch.ones(B * M, dtype=torch.bool)
    flat.update(rollout)
    return flat, int(picked.numel())


def make_generate_fn(
    *,
    model,
    label_embed,
    control_embedding,
    prev_segment_encoder,
    encodec_wrapper,
    steps: int,
    sample_rate: float,
    device: torch.device,
) -> GenerateFn:
    """A :data:`GenerateFn` running ``generate_segments`` as inference does (eval, no grad, fp32)."""
    from src.inference.flow_matching.generate import generate_segments

    modules = [m for m in (model, label_embed, control_embedding, prev_segment_encoder) if m is not None]
    latent_shape = None

    def _generate(labels, force_mean, velocity_mean, prev_waveform):
        nonlocal latent_shape
        prev_waveform = prev_waveform.to(device)
        was_training = [m.training for m in modules]
        for m in modules:
            m.eval()
        try:
            with torch.no_grad():
                if latent_shape is None:
                    latent_shape = encodec_wrapper.encode_waveform(
                        prev_waveform[:1], return_latent=True, return_codes=False,
                        input_sample_rate=float(sample_rate),
                    )["latent"].shape
                return generate_segments(
                    model=model,
                    label_embed=label_embed,
                    control_embedding=control_embedding,
                    encodec_wrapper=encodec_wrapper,
                    prev_segment_encoder=prev_segment_encoder,
                    labels=labels.to(device),
                    force_mean=force_mean.to(device),
                    velocity_mean=velocity_mean.to(device),
                    prev_waveform=prev_waveform,
                    steps=steps,
                    latent_shape=latent_shape,
                    sample_rate=sample_rate,
                    target_len=prev_waveform.shape[-1],
                    device=device,
                )
        finally:
            for m, mode in zip(modules, was_training):
                m.train(mode)

    return _generate
