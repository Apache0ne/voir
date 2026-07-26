from __future__ import annotations

import math

import torch


def synthetic_albedo_batch(batch: int = 8, size: int = 64, seed: int = 0, device: str = "cpu"):
    """Generate shaded RGB inputs with exact diffuse-albedo targets."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    albedo = torch.rand(batch, 3, size, size, generator=generator)
    albedo = torch.nn.functional.avg_pool2d(albedo, 7, stride=1, padding=3)
    y, x = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    light = torch.rand(batch, 2, generator=generator) * 1.5 - 0.75
    shading = []
    for i in range(batch):
        field = 0.55 + 0.45 * torch.clamp(1.0 - ((x-light[i, 0])**2 + (y-light[i, 1])**2), 0, 1)
        field *= 0.80 + 0.20 * torch.cos((x * (i + 1) + y) * math.pi)
        shading.append(field)
    shading = torch.stack(shading).unsqueeze(1)
    observed = torch.clamp(albedo * shading + 0.08 * shading.square(), 0, 1)
    return observed.to(device), albedo.to(device)
