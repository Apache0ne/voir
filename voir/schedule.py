from __future__ import annotations

import numpy as np
import torch
from scipy import stats


def rescale_sigmas(sigmas: torch.Tensor, start: float = 1.0, end: float = 0.0) -> torch.Tensor:
    """RES4LYF-compatible linear min/max rescale, including schedule reversal."""
    if sigmas.ndim != 1:
        raise ValueError("sigmas must be a one-dimensional tensor")
    if sigmas.numel() == 0:
        return sigmas.clone()
    lo, hi = sigmas.min(), sigmas.max()
    if torch.isclose(lo, hi):
        return torch.full_like(sigmas, float(start))
    return ((sigmas - lo) * (float(start) - float(end))) / (hi - lo) + float(end)


def beta_quantiles(steps: int, alpha: float = 0.6, beta: float = 0.8) -> torch.Tensor:
    """ComfyUI BetaSamplingScheduler quantiles in normalized flow time [1, 0]."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be > 0")
    p = 1.0 - np.linspace(0.0, 1.0, steps, endpoint=False, dtype=np.float64)
    q = stats.beta.ppf(p, alpha, beta)
    q = np.nan_to_num(q, nan=0.0, posinf=1.0, neginf=0.0)
    return torch.from_numpy(q.astype(np.float32))


def static_shift(sigmas: torch.Tensor, shift: float = 6.0) -> torch.Tensor:
    if shift <= 0:
        raise ValueError("shift must be > 0")
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


def beta_flow_sigmas(
    steps: int = 4,
    alpha: float = 0.6,
    beta: float = 0.8,
    start: float = 1.0,
    end: float = 0.0,
    shift: float = 6.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Beta-spaced Mage flow schedule with RES4LYF-style endpoint rescaling."""
    q = static_shift(beta_quantiles(steps, alpha, beta), shift=shift)
    q = torch.cat([q, q.new_zeros(1)])
    return rescale_sigmas(q, start=start, end=end).to(device)
