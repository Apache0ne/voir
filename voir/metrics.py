from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def global_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    mu_pred = pred.mean(dim=(-2, -1))
    mu_target = target.mean(dim=(-2, -1))
    var_pred = ((pred - mu_pred[..., None, None]) ** 2).mean(dim=(-2, -1))
    var_target = ((target - mu_target[..., None, None]) ** 2).mean(dim=(-2, -1))
    covariance = (
        (pred - mu_pred[..., None, None]) * (target - mu_target[..., None, None])
    ).mean(dim=(-2, -1))
    c1 = 0.01**2
    c2 = 0.03**2
    score = (
        (2 * mu_pred * mu_target + c1)
        * (2 * covariance + c2)
        / ((mu_pred.square() + mu_target.square() + c1) * (var_pred + var_target + c2))
    )
    return float(score.mean())


@torch.no_grad()
def local_ssim(pred: torch.Tensor, target: torch.Tensor, window: int = 7) -> float:
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
    return float(score.mean())


@torch.no_grad()
def albedo_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    difference = pred - target
    mse = float(difference.square().mean())
    mae = float(difference.abs().mean())
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    return {
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
        "ssim_7x7": local_ssim(pred, target, window=7),
        "global_ssim": global_ssim(pred, target),
    }
