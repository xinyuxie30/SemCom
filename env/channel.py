"""
AWGN / Rayleigh / Rician channel simulation.

All functions operate on torch tensors. The channel snapshot is part of the
fast-loop state, concatenated with sub-graph embedding, intent vector, and
node embedding.

[Phase 1 Fix]: channel_snapshot default dim changed 16→4 to match CSI_DIM=4.
  Features: [snr_norm, doppler_norm, sinr_est, path_loss_norm]
"""
from __future__ import annotations

from typing import Tuple

import torch


def awgn(x: torch.Tensor, snr_db: float) -> Tuple[torch.Tensor, float]:
    """
    Additive White Gaussian Noise.
    Args:
        x: signal tensor [*].
        snr_db: SNR in dB.
    Returns:
        (noisy_signal, effective_snr_db).
    """
    snr_linear = 10.0 ** (snr_db / 10.0)
    signal_power = x.abs().pow(2).mean()
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(x) * torch.sqrt(noise_power)
    effective_snr = 10.0 * torch.log10(signal_power / (noise.abs().pow(2).mean() + 1e-12))
    return x + noise, effective_snr.item()


def rayleigh_fading(
    x: torch.Tensor,
    snr_db: float,
    doppler_hz: float = 10.0,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, float]:
    """
    Rayleigh fast-fading channel (Jakes-like, simplified).
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    B, T = x.shape[0], x.shape[-1]

    N_sinusoids = 32
    t = torch.arange(T, device=x.device, dtype=x.dtype) * dt
    angles = torch.rand(N_sinusoids, device=x.device) * 2 * torch.pi
    freqs = doppler_hz * torch.cos(
        2 * torch.pi * torch.arange(1, N_sinusoids + 1, device=x.device, dtype=x.dtype) / N_sinusoids
    )
    h = torch.zeros(T, device=x.device, dtype=x.dtype)
    for i in range(N_sinusoids):
        h += torch.cos(2 * torch.pi * freqs[i] * t + angles[i])
    h /= torch.sqrt(torch.tensor(N_sinusoids, dtype=x.dtype))

    snr_linear = 10.0 ** (snr_db / 10.0)
    signal_power = x.abs().pow(2).mean()
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(x) * torch.sqrt(noise_power)

    y = x * h.unsqueeze(0) + noise
    inst_snr = 10.0 * torch.log10(
        (x.abs().pow(2).mean(dim=-1)) / (noise.abs().pow(2).mean(dim=-1) + 1e-12)
    )
    return y.squeeze(0) if B == 1 else y, inst_snr.mean().item()


# [Phase 1 Fix]: dim default = 4, matching CSI_DIM in the global dimension contract.
# Feature layout:
#   [0] snr_norm       ∈ [0, 1]
#   [1] doppler_norm   ∈ [0, 1]
#   [2] sinr_estimate  ∈ [0, 1]  (random placeholder for now)
#   [3] path_loss_norm ∈ [0, 1]  (random placeholder for now)
def channel_snapshot(dim: int = 4, snr_db: float = 15.0, doppler_hz: float = 10.0) -> torch.Tensor:
    """
    Produce a compact channel-state vector consumed by the policy network.
    """
    snr_norm = snr_db / 50.0                               # roughly normalise to [0, 1]
    doppler_norm = min(doppler_hz / 100.0, 1.0)
    sinr_est = 0.3 + 0.7 * torch.rand(1).item()            # [Phase 1 Fix]: stochastic placeholder
    path_loss_norm = 0.1 + 0.5 * torch.rand(1).item()      # [Phase 1 Fix]: stochastic placeholder
    vec = torch.zeros(dim)
    vec[0] = snr_norm
    vec[1] = doppler_norm
    vec[2] = sinr_est
    vec[3] = path_loss_norm
    return vec
