"""Self-forcing evaluation: generate K-segment chains along real windows, score each step."""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np
import torch

from src.utils.training.training_utils import compute_freq_spectrum_rmse, compute_gfc

# (labels [n], force_mean [n, 1], velocity_mean [n, 2], prev [n, L]) -> [n, L]
GenerateFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

MODES = ("teacher_forced", "self_forced", "self_forced_zero")


def self_forcing_chain_metrics(
    dataset,
    indices: Sequence[int],
    generate_fn: GenerateFn,
    *,
    num_unroll: int,
    num_chains: int,
    sample_rate: int,
    stft_n_fft: int,
    stft_hop_length: int,
    stft_win_length: int,
) -> Dict[str, float]:
    """GFC / frequency-spectrum RMSE of K-step chains ending at windows of ``indices``.

    Modes: ``teacher_forced`` (true prev each step), ``self_forced`` (own output,
    seeded with the true segment before the chain), ``self_forced_zero`` (seeded
    with zeros, as a serving session). Returns ``{mode}/{gfc,freq_rmse}_{mean,last}``.
    """
    K = int(num_unroll)
    picked = [int(i) for i in indices if dataset.has_context(int(i), K)][: int(num_chains)]
    if not picked:
        return {}
    chains = [dataset.chain(i, K) for i in picked]
    wave = torch.stack([c["chain_waveform"] for c in chains])  # [N, K+1, L]
    force = torch.stack([c["chain_force_mean"] for c in chains])  # [N, K+1, 1]
    velocity = torch.stack([c["chain_velocity_mean"] for c in chains])  # [N, K+1, 2]
    labels = torch.stack([c["label_idx"] for c in chains])
    N, _, L = wave.shape

    generated = {
        "teacher_forced": generate_fn(
            labels.repeat_interleave(K),
            force[:, 1:].reshape(N * K, 1),
            velocity[:, 1:].reshape(N * K, 2),
            wave[:, :-1].reshape(N * K, L),
        ).reshape(N, K, L)
    }
    for mode, seed in (("self_forced", wave[:, 0]), ("self_forced_zero", torch.zeros(N, L))):
        prev, steps = seed, []
        for t in range(1, K + 1):
            prev = generate_fn(labels, force[:, t], velocity[:, t], prev).cpu()
            steps.append(prev)
        generated[mode] = torch.stack(steps, dim=1)

    stft = (stft_n_fft, stft_hop_length, stft_win_length)
    truth = wave[:, 1:].numpy()
    out: Dict[str, float] = {}
    for mode in MODES:
        gen = generated[mode].cpu().numpy()
        gfc = np.array([[compute_gfc(truth[n, t], gen[n, t], sample_rate, *stft) for t in range(K)] for n in range(N)])
        frmse = np.array(
            [[compute_freq_spectrum_rmse(truth[n, t], gen[n, t], sample_rate, *stft) for t in range(K)] for n in range(N)]
        )
        out[f"{mode}/gfc_mean"] = float(gfc.mean())
        out[f"{mode}/gfc_last"] = float(gfc[:, -1].mean())
        out[f"{mode}/freq_rmse_mean"] = float(frmse.mean())
        out[f"{mode}/freq_rmse_last"] = float(frmse[:, -1].mean())
    return out
