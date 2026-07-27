from __future__ import annotations

import torch
import torch.nn.functional as F


AUXILIARY_ALBEDO_CHANNELS = 63


def _as_batch(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if image.ndim == 3:
        if image.shape[0] != 3:
            raise ValueError("image must have three RGB channels")
        return image.unsqueeze(0), True
    if image.ndim == 4 and image.shape[1] == 3:
        return image, False
    raise ValueError("image must be [3,H,W] or [B,3,H,W]")


def _safe_odd_kernel(requested: int, height: int, width: int) -> int:
    limit = max(1, min(height, width))
    if limit % 2 == 0:
        limit -= 1
    return max(1, min(int(requested), limit))


@torch.no_grad()
def fixed_albedo_features(
    image: torch.Tensor,
    size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Build a deterministic 63-channel intrinsic-image feature bank.

    The bank is fixed and has no trainable parameters. It preserves source RGB,
    nonlinear color transforms, chromaticity, multiscale local illumination
    estimates, high-pass residuals, Retinex-style ratios, and image gradients.
    It is intended to complement the frozen Mage trajectory, not replace it.
    """
    x, squeezed = _as_batch(image)
    x = x.detach().float().clamp(0.0, 1.0)
    if size is not None and tuple(x.shape[-2:]) != tuple(size):
        x = F.interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=True)
        x = x.clamp(0.0, 1.0)

    features = [
        x,
        torch.sqrt(x.clamp_min(1e-4)),
        x.square(),
        torch.log(x.clamp_min(1e-3)),
        x / (x.sum(dim=1, keepdim=True) + 1e-3),
    ]
    luminance = x.mean(dim=1, keepdim=True)
    features.extend([luminance, torch.log(luminance.clamp_min(1e-3))])

    height, width = x.shape[-2:]
    for requested in (3, 7, 15, 31):
        kernel = _safe_odd_kernel(requested, height, width)
        padding = kernel // 2
        blur_rgb = F.avg_pool2d(x, kernel, stride=1, padding=padding)
        blur_luminance = F.avg_pool2d(luminance, kernel, stride=1, padding=padding)
        features.extend(
            [
                blur_rgb,
                x - blur_rgb,
                x / (blur_luminance + 0.05),
                blur_luminance,
            ]
        )

    dx = F.pad(x[..., 1:] - x[..., :-1], (0, 1, 0, 0))
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    features.extend([dx, dy])

    out = torch.cat(features, dim=1)
    if out.shape[1] != AUXILIARY_ALBEDO_CHANNELS:
        raise RuntimeError(f"expected {AUXILIARY_ALBEDO_CHANNELS} channels, got {out.shape[1]}")
    return out[0] if squeezed else out
