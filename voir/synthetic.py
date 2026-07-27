from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def synthetic_albedo_batch(
    batch: int = 8,
    size: int = 64,
    seed: int = 0,
    device: str = "cpu",
):
    """Small shaded-RGB batch used by fast smoke tests."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    albedo = torch.rand(batch, 3, size, size, generator=generator)
    albedo = F.avg_pool2d(albedo, 7, stride=1, padding=3)
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, size),
        torch.linspace(-1, 1, size),
        indexing="ij",
    )
    light = torch.rand(batch, 2, generator=generator) * 1.5 - 0.75
    shading = []
    for index in range(batch):
        field = 0.55 + 0.45 * torch.clamp(
            1.0 - ((x - light[index, 0]) ** 2 + (y - light[index, 1]) ** 2),
            0,
            1,
        )
        field *= 0.80 + 0.20 * torch.cos((x * (index + 1) + y) * math.pi)
        shading.append(field)
    shading = torch.stack(shading).unsqueeze(1)
    observed = torch.clamp(albedo * shading + 0.08 * shading.square(), 0, 1)
    return observed.to(device), albedo.to(device)


def synthetic_albedo_benchmark_batch(
    batch: int = 256,
    size: int = 48,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard deterministic intrinsic-albedo benchmark with exact targets.

    It combines sharp material boundaries, textured diffuse color, spatially
    varying illumination, channel-dependent color casts, specular highlights,
    exposure changes, and camera gamma. Different seeds form disjoint held-out
    train, validation, and test sets.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    albedo = torch.rand(batch, 3, size, size, generator=generator)
    albedo = F.avg_pool2d(albedo, 7, stride=1, padding=3)

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, size),
        torch.linspace(-1, 1, size),
        indexing="ij",
    )
    yy = yy.unsqueeze(0).unsqueeze(0)
    xx = xx.unsqueeze(0).unsqueeze(0)

    for index in range(batch):
        for _ in range(3):
            center_x = (torch.rand((), generator=generator) * 1.6 - 0.8).item()
            center_y = (torch.rand((), generator=generator) * 1.6 - 0.8).item()
            radius = (torch.rand((), generator=generator) * 0.45 + 0.08).item()
            mask = (((xx - center_x) ** 2 + (yy - center_y) ** 2) < radius**2).float()
            color = torch.rand(1, 3, 1, 1, generator=generator) * 0.9 + 0.05
            albedo[index:index + 1] = albedo[index:index + 1] * (1.0 - mask) + color * mask
    albedo = albedo.clamp(0.02, 0.98)

    shading_fields = []
    for _ in range(batch):
        light_x = (torch.rand((), generator=generator) * 1.6 - 0.8).item()
        light_y = (torch.rand((), generator=generator) * 1.6 - 0.8).item()
        direction_x = (torch.rand((), generator=generator) * 2.0 - 1.0).item()
        direction_y = (torch.rand((), generator=generator) * 2.0 - 1.0).item()
        radial = 0.35 + 0.65 * torch.clamp(
            1.0 - ((xx - light_x) ** 2 + (yy - light_y) ** 2) / 2.5,
            0,
            1,
        )
        gradient = 0.65 + 0.35 * torch.clamp(
            1.0 + xx * direction_x + yy * direction_y,
            0,
            2,
        ) / 2.0
        shading_fields.append((radial * gradient).clamp(0.15, 1.2))
    shading = torch.cat(shading_fields, dim=0)

    color_cast = torch.rand(batch, 3, 1, 1, generator=generator) * 0.35 + 0.75
    observed = albedo * shading * color_cast

    for index in range(batch):
        specular_x = (torch.rand((), generator=generator) * 1.6 - 0.8).item()
        specular_y = (torch.rand((), generator=generator) * 1.6 - 0.8).item()
        spread = (torch.rand((), generator=generator) * 0.08 + 0.015).item()
        strength = (torch.rand((), generator=generator) * 0.4).item()
        specular = torch.exp(
            -((xx - specular_x) ** 2 + (yy - specular_y) ** 2) / spread
        ) * strength
        observed[index:index + 1] += specular

    exposure = torch.rand(batch, 1, 1, 1, generator=generator) * 0.4 + 0.8
    gamma = torch.rand(batch, 1, 1, 1, generator=generator) * 0.35 + 0.85
    observed = (observed * exposure).clamp(0, 1).pow(1.0 / gamma)
    return observed.clamp(0, 1).to(device), albedo.to(device)
