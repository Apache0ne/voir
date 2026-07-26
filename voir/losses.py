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


def albedo_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    l_rgb = charbonnier(pred, target)
    l_grad = gradient_loss(pred, target)
    l_color = F.l1_loss(pred.mean(dim=(-2, -1)), target.mean(dim=(-2, -1)))
    total = l_rgb + 0.20 * l_grad + 0.10 * l_color
    return total, {
        "rgb": float(l_rgb.detach()),
        "gradient": float(l_grad.detach()),
        "color": float(l_color.detach()),
    }
