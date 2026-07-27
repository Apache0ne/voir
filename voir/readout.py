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


def _safe_odd_kernel(requested: int, height: int, width: int) -> int:
    limit = max(1, min(height, width))
    if limit % 2 == 0:
        limit -= 1
    return max(1, min(int(requested), limit))


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
    """Large-receptive-field residual block used by the v2/v3 dense readouts."""

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
    canonical = (1, 2, 4, 8, 8, 4, 2, 1)
    if depth <= len(canonical):
        return canonical[:depth]
    return tuple(canonical[index % len(canonical)] for index in range(depth))


class AlbedoReadout(nn.Module):
    """Trainable dense readout; every reservoir and auxiliary feature is fixed.

    ``intrinsic_v3`` separates the variable-width reservoir trajectory from the
    fixed 63-channel image feature bank. This lets a CPU-trained decoder transfer
    to Mage states: only ``trajectory_proj`` changes shape, while the auxiliary
    branch, intrinsic correction trunk, and output heads are reusable.
    """

    def __init__(
        self,
        in_channels: int,
        width: int = 32,
        depth: int = 7,
        architecture: str = "dilated_v2",
        trajectory_channels: int | None = None,
        auxiliary_channels: int = 63,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.width = int(width)
        self.depth = int(depth)
        self.architecture = str(architecture)
        self.auxiliary_channels = int(auxiliary_channels)
        self.trajectory_channels = (
            int(trajectory_channels)
            if trajectory_channels is not None
            else max(0, self.in_channels - self.auxiliary_channels)
        )

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
        elif self.architecture == "intrinsic_v3":
            if self.auxiliary_channels < 3:
                raise ValueError("intrinsic_v3 needs at least three auxiliary RGB channels")
            if self.trajectory_channels < 1:
                raise ValueError("intrinsic_v3 needs at least one trajectory channel")
            if self.trajectory_channels + self.auxiliary_channels != self.in_channels:
                raise ValueError("trajectory_channels + auxiliary_channels must equal in_channels")
            branch_width = max(8, self.width // 2)
            self.trajectory_proj = nn.Sequential(
                nn.Conv2d(self.trajectory_channels, branch_width, 1),
                nn.GroupNorm(_group_count(branch_width, maximum=8), branch_width),
                nn.SiLU(),
            )
            self.auxiliary_proj = nn.Sequential(
                nn.Conv2d(self.auxiliary_channels, branch_width, 1),
                nn.GroupNorm(_group_count(branch_width, maximum=8), branch_width),
                nn.SiLU(),
            )
            self.fuse = nn.Conv2d(branch_width * 2, self.width, 1)
            self.blocks = nn.ModuleList(
                [DilatedResidualBlock(self.width, dilation) for dilation in _dilations(self.depth)]
            )
            self.global_fc = nn.Sequential(
                nn.Linear(self.width, self.width),
                nn.SiLU(),
                nn.Linear(self.width, self.width * 2),
            )
            head_norm = _group_count(self.width, maximum=8)
            self.head_norm = nn.GroupNorm(head_norm, self.width)
            self.correction_head = nn.Conv2d(self.width, 6, 3, padding=1)
            self.direct_head = nn.Conv2d(self.width, 3, 3, padding=1)
            self.gate_head = nn.Conv2d(self.width, 1, 1)
            nn.init.zeros_(self.correction_head.weight)
            nn.init.zeros_(self.correction_head.bias)
            nn.init.zeros_(self.direct_head.weight)
            nn.init.zeros_(self.direct_head.bias)
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, 2.0)
        else:
            raise ValueError(f"unknown readout architecture: {self.architecture}")

    def _forward_intrinsic_v3(self, features: torch.Tensor) -> torch.Tensor:
        trajectory = features[:, : self.trajectory_channels]
        auxiliary = features[:, self.trajectory_channels :]
        source_rgb = auxiliary[:, :3].clamp(1e-4, 1.0 - 1e-4)
        hidden = torch.cat(
            [self.trajectory_proj(trajectory), self.auxiliary_proj(auxiliary)],
            dim=1,
        )
        hidden = self.fuse(hidden)
        context = self.global_fc(hidden.mean(dim=(-2, -1)))
        scale, bias = context.chunk(2, dim=1)
        hidden = hidden * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(-1).unsqueeze(-1))
        hidden = hidden + bias.unsqueeze(-1).unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = F.silu(self.head_norm(hidden))

        correction = self.correction_head(hidden)
        low_frequency, detail = correction.chunk(2, dim=1)
        kernel = _safe_odd_kernel(31, *low_frequency.shape[-2:])
        low_frequency = F.avg_pool2d(
            low_frequency,
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )
        detail = 0.75 * torch.tanh(detail)
        source_logits = torch.logit(source_rgb)
        corrected = torch.sigmoid(source_logits + low_frequency + detail)
        direct = torch.sigmoid(self.direct_head(hidden))
        gate = torch.sigmoid(self.gate_head(hidden))
        return gate * corrected + (1.0 - gate) * direct

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
            output = torch.sigmoid(self.output(self.blocks(self.input(features))))
        elif self.architecture == "dilated_v2":
            hidden = self.inp(features)
            context = self.global_fc(hidden.mean(dim=(-2, -1))).unsqueeze(-1).unsqueeze(-1)
            hidden = hidden + context
            for block in self.blocks:
                hidden = block(hidden)
            output = torch.sigmoid(self.out(hidden))
        else:
            output = self._forward_intrinsic_v3(features)

        if output_size is not None and tuple(output.shape[-2:]) != tuple(output_size):
            output = F.interpolate(output, size=output_size, mode="bilinear", align_corners=False)
        return output

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
                "trajectory_channels": self.trajectory_channels,
                "auxiliary_channels": self.auxiliary_channels,
            },
            "task": "albedo",
            "format_version": 3,
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
