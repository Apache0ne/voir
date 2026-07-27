from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .state import ReservoirState


def _group_count(channels: int, maximum: int = 16) -> int:
    groups = min(int(maximum), int(channels))
    while groups > 1 and channels % groups:
        groups -= 1
    return groups


class ResidualBlock(nn.Module):
    """Legacy two-convolution block retained for old checkpoints."""

    def __init__(self, channels: int):
        super().__init__()
        groups = _group_count(channels)
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


class DilatedResidualBlock(nn.Module):
    """Large-receptive-field residual block used by the v2 dense readout."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        groups = _group_count(channels, maximum=8)
        # Short names intentionally match the validated CPU checkpoint format.
        self.n1 = nn.GroupNorm(groups, channels)
        self.c1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=int(dilation),
            dilation=int(dilation),
        )
        self.n2 = nn.GroupNorm(groups, channels)
        self.c2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.silu(self.n1(x))
        residual = self.c1(residual)
        residual = F.silu(self.n2(residual))
        residual = self.c2(residual)
        return x + residual


def _dilations(depth: int) -> tuple[int, ...]:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    canonical = (1, 2, 4, 8, 4, 2, 1)
    if depth <= len(canonical):
        return canonical[:depth]
    return tuple(canonical[index % len(canonical)] for index in range(depth))


class AlbedoReadout(nn.Module):
    """Trainable dense readout; every reservoir and auxiliary feature is fixed."""

    def __init__(
        self,
        in_channels: int,
        width: int = 32,
        depth: int = 7,
        architecture: str = "dilated_v2",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.width = int(width)
        self.depth = int(depth)
        self.architecture = str(architecture)

        if self.architecture == "legacy":
            self.input = nn.Conv2d(self.in_channels, self.width, 1)
            self.blocks = nn.Sequential(*[ResidualBlock(self.width) for _ in range(self.depth)])
            self.output = nn.Sequential(
                nn.GroupNorm(_group_count(self.width), self.width),
                nn.SiLU(),
                nn.Conv2d(self.width, 3, 3, padding=1),
            )
        elif self.architecture == "dilated_v2":
            self.inp = nn.Conv2d(self.in_channels, self.width, 1)
            self.blocks = nn.ModuleList(
                [DilatedResidualBlock(self.width, dilation) for dilation in _dilations(self.depth)]
            )
            self.global_fc = nn.Sequential(
                nn.Linear(self.width, self.width),
                nn.SiLU(),
                nn.Linear(self.width, self.width),
            )
            self.out = nn.Sequential(
                nn.GroupNorm(_group_count(self.width, maximum=8), self.width),
                nn.SiLU(),
                nn.Conv2d(self.width, 3, 3, padding=1),
            )
        else:
            raise ValueError(f"unknown readout architecture: {self.architecture}")

    def forward(
        self,
        features: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("features must be [B,C,H,W]")
        if features.shape[1] != self.in_channels:
            raise ValueError(
                f"readout expects {self.in_channels} channels, received {features.shape[1]}"
            )

        if self.architecture == "legacy":
            logits = self.output(self.blocks(self.input(features)))
        else:
            hidden = self.inp(features)
            context = self.global_fc(hidden.mean(dim=(-2, -1))).unsqueeze(-1).unsqueeze(-1)
            hidden = hidden + context
            for block in self.blocks:
                hidden = block(hidden)
            logits = self.out(hidden)

        if output_size is not None and tuple(logits.shape[-2:]) != tuple(output_size):
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)

    def predict_state(
        self,
        state: ReservoirState,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        self.eval()
        with torch.inference_mode():
            return self(state.flattened(device), state.output_size)

    def checkpoint(self) -> dict:
        return {
            "model": self.state_dict(),
            "config": {
                "in_channels": self.in_channels,
                "width": self.width,
                "depth": self.depth,
                "architecture": self.architecture,
            },
            "task": "albedo",
            "format_version": 2,
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: dict,
        map_location: str | torch.device = "cpu",
    ) -> "AlbedoReadout":
        config = dict(payload["config"])
        if "architecture" not in config:
            config["architecture"] = "legacy"
        model = cls(**config)
        model.load_state_dict(payload["model"])
        return model.to(map_location)
