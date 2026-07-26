from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .state import ReservoirState


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(16, channels)
        while channels % groups:
            groups -= 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AlbedoReadout(nn.Module):
    """Trainable dense readout; the reservoir itself remains fixed."""

    def __init__(self, in_channels: int, width: int = 96, depth: int = 4):
        super().__init__()
        self.in_channels = int(in_channels)
        self.width = int(width)
        self.depth = int(depth)
        self.input = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
        self.output = nn.Sequential(
            nn.GroupNorm(min(16, width), width),
            nn.SiLU(),
            nn.Conv2d(width, 3, 3, padding=1),
        )

    def forward(self, features: torch.Tensor, output_size: tuple[int, int] | None = None) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("features must be [B,C,H,W]")
        x = self.output(self.blocks(self.input(features)))
        if output_size is not None and tuple(x.shape[-2:]) != tuple(output_size):
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(x)

    def predict_state(self, state: ReservoirState, device: str | torch.device = "cpu") -> torch.Tensor:
        self.eval()
        with torch.inference_mode():
            return self(state.flattened(device), state.output_size)

    def checkpoint(self) -> dict:
        return {
            "model": self.state_dict(),
            "config": {"in_channels": self.in_channels, "width": self.width, "depth": self.depth},
            "task": "albedo",
        }

    @classmethod
    def from_checkpoint(cls, payload: dict, map_location: str | torch.device = "cpu") -> "AlbedoReadout":
        config = payload["config"]
        model = cls(**config)
        model.load_state_dict(payload["model"])
        return model.to(map_location)
