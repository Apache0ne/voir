from __future__ import annotations

import numpy as np
import torch
from scipy import stats


def rescale_sigmas(sigmas: torch.Tensor, start: float = 1.0, end: float = 0.0) -> torch.Tensor:
    """RES4LYF linear min/max rescale, including schedule reversal."""
    if sigmas.ndim != 1:
        raise ValueError("sigmas must be a one-dimensional tensor")
    if sigmas.numel() == 0:
        return sigmas.clone()
    lo, hi = sigmas.min(), sigmas.max()
    if torch.isclose(lo, hi):
        return torch.full_like(sigmas, float(start))
    return ((sigmas - lo) * (float(start) - float(end))) / (hi - lo) + float(end)


def flow_time_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
    if shift <= 0:
        raise ValueError("shift must be > 0")
    return shift * t / (1.0 + (shift - 1.0) * t)


def flow_model_sigmas(
    timesteps: int = 1000,
    shift: float = 6.0,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Comfy ModelSamplingDiscreteFlow sigma table, ascending from low noise to 1."""
    if timesteps < 2:
        raise ValueError("timesteps must be >= 2")
    t = torch.arange(1, timesteps + 1, dtype=torch.float32, device=device) / timesteps
    return flow_time_shift(shift, t).float()


def beta_scheduler_from_model_sigmas(
    model_sigmas: torch.Tensor,
    steps: int,
    alpha: float = 0.6,
    beta: float = 0.8,
) -> torch.Tensor:
    """Exact port of ComfyUI's beta_scheduler selection and duplicate removal."""
    if model_sigmas.ndim != 1 or model_sigmas.numel() < 2:
        raise ValueError("model_sigmas must be a 1D table with at least two entries")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be > 0")

    total_timesteps = model_sigmas.numel() - 1
    probs = 1.0 - np.linspace(0.0, 1.0, steps, endpoint=False, dtype=np.float64)
    indices = np.rint(stats.beta.ppf(probs, alpha, beta) * total_timesteps)
    indices = np.nan_to_num(indices, nan=0.0, posinf=total_timesteps, neginf=0.0)
    indices = np.clip(indices.astype(np.int64), 0, total_timesteps)

    selected: list[torch.Tensor] = []
    last_index = -1
    for index in indices.tolist():
        if index != last_index:
            selected.append(model_sigmas[index])
        last_index = index
    selected.append(model_sigmas.new_zeros(()))
    return torch.stack(selected)


def beta_flow_sigmas(
    steps: int = 4,
    alpha: float = 0.6,
    beta: float = 0.8,
    start: float = 1.0,
    end: float = 0.0,
    shift: float = 6.0,
    train_timesteps: int = 1000,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Standalone Mage flow schedule: discrete-flow table -> beta indices -> RES4LYF rescale."""
    table = flow_model_sigmas(train_timesteps, shift, device="cpu")
    sigmas = beta_scheduler_from_model_sigmas(table, steps, alpha, beta)
    return rescale_sigmas(sigmas, start=start, end=end).to(device)


def flow_percent_to_sigma(percent: float | torch.Tensor, shift: float = 6.0) -> torch.Tensor:
    """Comfy ModelSamplingDiscreteFlow.percent_to_sigma for the normalized flow domain."""
    p = torch.as_tensor(percent, dtype=torch.float32)
    one = torch.ones_like(p)
    zero = torch.zeros_like(p)
    interior = flow_time_shift(shift, 1.0 - p)
    return torch.where(p <= 0.0, one, torch.where(p >= 1.0, zero, interior))
