from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _mask_like(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if tuple(mask.shape[-2:]) != tuple(value.shape[-2:]):
        mask = F.interpolate(mask.float(), size=value.shape[-2:], mode="nearest")
    if mask.shape[1] == 1 and value.shape[1] != 1:
        mask = mask.expand(-1, value.shape[1], -1, -1)
    return mask.to(device=value.device, dtype=value.dtype).clamp(0.0, 1.0)


def _weighted_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = _mask_like(mask, value)
    return (value * weight).sum() / weight.sum().clamp_min(1e-6)


@torch.no_grad()
def global_ssim(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is None:
        mu_pred = pred.mean(dim=(-2, -1))
        mu_target = target.mean(dim=(-2, -1))
        var_pred = ((pred - mu_pred[..., None, None]) ** 2).mean(dim=(-2, -1))
        var_target = ((target - mu_target[..., None, None]) ** 2).mean(dim=(-2, -1))
        covariance = (
            (pred - mu_pred[..., None, None]) * (target - mu_target[..., None, None])
        ).mean(dim=(-2, -1))
    else:
        base_mask = mask.unsqueeze(1) if mask.ndim == 3 else mask
        base_mask = base_mask.to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
        denominator = base_mask.sum(dim=(-2, -1)).clamp_min(1e-6)
        mu_pred = (pred * base_mask).sum(dim=(-2, -1)) / denominator
        mu_target = (target * base_mask).sum(dim=(-2, -1)) / denominator
        var_pred = (((pred - mu_pred[..., None, None]) ** 2) * base_mask).sum(dim=(-2, -1)) / denominator
        var_target = (((target - mu_target[..., None, None]) ** 2) * base_mask).sum(dim=(-2, -1)) / denominator
        covariance = (
            (pred - mu_pred[..., None, None])
            * (target - mu_target[..., None, None])
            * base_mask
        ).sum(dim=(-2, -1)) / denominator
    c1 = 0.01**2
    c2 = 0.03**2
    score = (
        (2 * mu_pred * mu_target + c1)
        * (2 * covariance + c2)
        / ((mu_pred.square() + mu_target.square() + c1) * (var_pred + var_target + c2))
    )
    return float(score.mean())


@torch.no_grad()
def local_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window: int = 7,
    mask: torch.Tensor | None = None,
) -> float:
    kernel = max(1, min(int(window), pred.shape[-2], pred.shape[-1]))
    if kernel % 2 == 0:
        kernel -= 1
    padding = kernel // 2
    mu_pred = F.avg_pool2d(pred, kernel, stride=1, padding=padding)
    mu_target = F.avg_pool2d(target, kernel, stride=1, padding=padding)
    var_pred = F.avg_pool2d(pred.square(), kernel, stride=1, padding=padding) - mu_pred.square()
    var_target = F.avg_pool2d(target.square(), kernel, stride=1, padding=padding) - mu_target.square()
    covariance = F.avg_pool2d(pred * target, kernel, stride=1, padding=padding) - mu_pred * mu_target
    c1 = 0.01**2
    c2 = 0.03**2
    score = (
        (2 * mu_pred * mu_target + c1)
        * (2 * covariance + c2)
        / ((mu_pred.square() + mu_target.square() + c1) * (var_pred + var_target + c2))
    )
    return float(score.mean() if mask is None else _weighted_mean(score, mask))


@torch.no_grad()
def albedo_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    difference = pred - target
    if mask is None:
        mse = float(difference.square().mean())
        mae = float(difference.abs().mean())
    else:
        mse = float(_weighted_mean(difference.square(), mask))
        mae = float(_weighted_mean(difference.abs(), mask))
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    return {
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
        "ssim_7x7": local_ssim(pred, target, window=7, mask=mask),
        "global_ssim": global_ssim(pred, target, mask=mask),
    }
