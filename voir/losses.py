from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target).square() + eps * eps).mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pdx = pred[..., :, 1:] - pred[..., :, :-1]
    pdy = pred[..., 1:, :] - pred[..., :-1, :]
    tdx = target[..., :, 1:] - target[..., :, :-1]
    tdy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy)


def multiscale_loss(pred: torch.Tensor, target: torch.Tensor, factor: int = 4) -> torch.Tensor:
    height, width = pred.shape[-2:]
    kernel = max(1, min(int(factor), height, width))
    if kernel == 1:
        return F.l1_loss(pred, target)
    return F.l1_loss(
        F.avg_pool2d(pred, kernel_size=kernel, stride=kernel),
        F.avg_pool2d(target, kernel_size=kernel, stride=kernel),
    )


def albedo_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Intrinsic-albedo objective used by the CPU benchmark and Mage readout."""
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
