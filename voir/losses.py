from __future__ import annotations

import torch
import torch.nn.functional as F


def _mask_like(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] not in (1, value.shape[1]):
        raise ValueError("mask must be [B,1,H,W], [B,C,H,W], or [B,H,W]")
    if tuple(mask.shape[-2:]) != tuple(value.shape[-2:]):
        mask = F.interpolate(mask.float(), size=value.shape[-2:], mode="nearest")
    if mask.shape[1] == 1 and value.shape[1] != 1:
        mask = mask.expand(-1, value.shape[1], -1, -1)
    return mask.to(device=value.device, dtype=value.dtype).clamp(0.0, 1.0)


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    weight = _mask_like(mask, value)
    return (value * weight).sum() / weight.sum().clamp_min(eps)


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return (torch.sqrt((pred - target).square() + eps * eps) - eps).mean()


def masked_charbonnier(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    residual = torch.sqrt((pred - target).square() + eps * eps) - eps
    return masked_mean(residual, mask)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pdx = pred[..., :, 1:] - pred[..., :, :-1]
    pdy = pred[..., 1:, :] - pred[..., :-1, :]
    tdx = target[..., :, 1:] - target[..., :, :-1]
    tdy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy)


def masked_gradient_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    pdx = pred[..., :, 1:] - pred[..., :, :-1]
    pdy = pred[..., 1:, :] - pred[..., :-1, :]
    tdx = target[..., :, 1:] - target[..., :, :-1]
    tdy = target[..., 1:, :] - target[..., :-1, :]
    mask_x = torch.minimum(mask[..., :, 1:], mask[..., :, :-1])
    mask_y = torch.minimum(mask[..., 1:, :], mask[..., :-1, :])
    return masked_mean((pdx - tdx).abs(), mask_x) + masked_mean((pdy - tdy).abs(), mask_y)


def multiscale_loss(pred: torch.Tensor, target: torch.Tensor, factor: int = 4) -> torch.Tensor:
    height, width = pred.shape[-2:]
    kernel = max(1, min(int(factor), height, width))
    if kernel == 1:
        return F.l1_loss(pred, target)
    return F.l1_loss(
        F.avg_pool2d(pred, kernel_size=kernel, stride=kernel),
        F.avg_pool2d(target, kernel_size=kernel, stride=kernel),
    )


def masked_multiscale_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    factor: int = 4,
) -> torch.Tensor:
    height, width = pred.shape[-2:]
    kernel = max(1, min(int(factor), height, width))
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
    if kernel == 1:
        return masked_mean((pred - target).abs(), mask)
    denominator = F.avg_pool2d(mask, kernel_size=kernel, stride=kernel).clamp_min(1e-6)
    pred_small = F.avg_pool2d(pred * mask, kernel_size=kernel, stride=kernel) / denominator
    target_small = F.avg_pool2d(target * mask, kernel_size=kernel, stride=kernel) / denominator
    valid = (denominator > 1e-5).to(pred.dtype)
    return masked_mean((pred_small - target_small).abs(), valid)


def _masked_local_moments(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    kernel: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
    padding = kernel // 2
    weight = F.avg_pool2d(mask, kernel, stride=1, padding=padding).clamp_min(1e-6)
    mu_pred = F.avg_pool2d(pred * mask, kernel, stride=1, padding=padding) / weight
    mu_target = F.avg_pool2d(target * mask, kernel, stride=1, padding=padding) / weight
    second_pred = F.avg_pool2d(pred.square() * mask, kernel, stride=1, padding=padding) / weight
    second_target = F.avg_pool2d(target.square() * mask, kernel, stride=1, padding=padding) / weight
    cross = F.avg_pool2d(pred * target * mask, kernel, stride=1, padding=padding) / weight
    var_pred = (second_pred - mu_pred.square()).clamp_min(0.0)
    var_target = (second_target - mu_target.square()).clamp_min(0.0)
    covariance = cross - mu_pred * mu_target
    valid = (weight > 1e-5).to(pred.dtype)
    return mu_pred, mu_target, var_pred, var_target, covariance, valid


def masked_local_ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    window: int = 7,
) -> torch.Tensor:
    kernel = max(1, min(int(window), pred.shape[-2], pred.shape[-1]))
    if kernel % 2 == 0:
        kernel -= 1
    mu_pred, mu_target, var_pred, var_target, covariance, valid = _masked_local_moments(
        pred, target, mask, kernel
    )
    c1 = 0.01**2
    c2 = 0.03**2
    score = (
        (2 * mu_pred * mu_target + c1)
        * (2 * covariance + c2)
        / ((mu_pred.square() + mu_target.square() + c1) * (var_pred + var_target + c2))
    )
    return 1.0 - masked_mean(score, valid)


def masked_chromaticity_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred_log = torch.log(pred.clamp_min(1e-3))
    target_log = torch.log(target.clamp_min(1e-3))
    pred_chroma = pred_log - pred_log.mean(dim=1, keepdim=True)
    target_chroma = target_log - target_log.mean(dim=1, keepdim=True)
    return masked_mean((pred_chroma - target_chroma).abs(), mask)


def masked_color_mean_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
    denominator = mask.sum(dim=(-2, -1)).clamp_min(1e-6)
    pred_mean = (pred * mask).sum(dim=(-2, -1)) / denominator
    target_mean = (target * mask).sum(dim=(-2, -1)) / denominator
    return F.l1_loss(pred_mean, target_mean)


def albedo_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Intrinsic-albedo objective used by the synthetic CPU benchmark."""
    l_rgb = charbonnier(pred, target)
    l_grad = gradient_loss(pred, target)
    l_color = F.l1_loss(pred.mean(dim=(-2, -1)), target.mean(dim=(-2, -1)))
    l_multiscale = multiscale_loss(pred, target)
    total = l_rgb + 0.15 * l_grad + 0.20 * l_color + 0.10 * l_multiscale
    return total, {
        "rgb": float(l_rgb.detach()),
        "gradient": float(l_grad.detach()),
        "color": float(l_color.detach()),
        "multiscale": float(l_multiscale.detach()),
        "total": float(total.detach()),
    }


def masked_albedo_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Real-photo albedo objective using Olbedo's validity mask."""
    l_rgb = masked_charbonnier(pred, target, mask)
    l_grad = masked_gradient_loss(pred, target, mask)
    l_ssim = masked_local_ssim_loss(pred, target, mask)
    l_chroma = masked_chromaticity_loss(pred, target, mask)
    l_color = masked_color_mean_loss(pred, target, mask)
    l_multiscale = masked_multiscale_loss(pred, target, mask)
    total = (
        l_rgb
        + 0.18 * l_grad
        + 0.20 * l_ssim
        + 0.10 * l_chroma
        + 0.12 * l_color
        + 0.10 * l_multiscale
    )
    return total, {
        "rgb": float(l_rgb.detach()),
        "gradient": float(l_grad.detach()),
        "ssim": float(l_ssim.detach()),
        "chroma": float(l_chroma.detach()),
        "color": float(l_color.detach()),
        "multiscale": float(l_multiscale.detach()),
        "total": float(total.detach()),
    }
