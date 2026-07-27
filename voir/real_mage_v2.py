from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int, maximum: int = 8) -> int:
    groups = min(int(channels), int(maximum))
    while groups > 1 and channels % groups:
        groups -= 1
    return groups


class GatedSeparableBlock(nn.Module):
    """Efficient residual block with a wider pointwise nonlinear path."""

    def __init__(self, channels: int, dilation: int = 1, expansion: int = 2):
        super().__init__()
        hidden = int(channels) * int(expansion)
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            3,
            padding=int(dilation),
            dilation=int(dilation),
            groups=channels,
            bias=False,
        )
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.project = nn.Conv2d(hidden, channels, 1)
        self.gate = nn.Parameter(torch.tensor(-1.5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = F.silu(self.norm(value))
        residual = self.depthwise(residual)
        residual = self.project(F.silu(self.expand(residual)))
        return value + torch.sigmoid(self.gate) * residual


class SourceDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(_groups(out_channels), out_channels)
        self.block = GatedSeparableBlock(out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.silu(self.norm(self.down(value)))
        return self.block(value)


class SourcePyramid(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.channels = (
            max(24, width // 4),
            max(32, width // 3),
            max(48, width // 2),
            max(64, width * 2 // 3),
            width,
        )
        c0, c1, c2, c3, c4 = self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, 3, padding=1),
            nn.GroupNorm(_groups(c0), c0),
            nn.SiLU(),
            GatedSeparableBlock(c0),
        )
        self.down1 = SourceDownBlock(c0, c1)
        self.down2 = SourceDownBlock(c1, c2)
        self.down3 = SourceDownBlock(c2, c3)
        self.down4 = SourceDownBlock(c3, c4)

    def forward(self, source: torch.Tensor) -> list[torch.Tensor]:
        s0 = self.stem(source.float())
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        s4 = self.down4(s3)
        return [s0, s1, s2, s3, s4]


class UpFuseBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.pre = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pre_norm = nn.GroupNorm(_groups(out_channels), out_channels)
        self.fuse = nn.Conv2d(out_channels + skip_channels, out_channels, 1)
        self.block1 = GatedSeparableBlock(out_channels)
        self.block2 = GatedSeparableBlock(out_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        value = F.silu(self.pre_norm(self.pre(value)))
        value = self.fuse(torch.cat([value, skip], dim=1))
        value = self.block1(value)
        return self.block2(value)


class RealMageAlbedoNetV2(nn.Module):
    """Multi-scale source-fusion decoder for frozen Mage trajectory caches.

    Compared with v1, the source image is encoded at every spatial scale and fused
    into every decoder stage. This preserves high-frequency material boundaries while
    the frozen Mage trajectory branches provide global illumination/edit context.
    """

    FORMAT = "voir_real_mage_albedo_v2"

    def __init__(
        self,
        hidden_channels: int,
        sampler_channels: int,
        auxiliary_channels: int = 63,
        width: int = 96,
        depth: int = 10,
        upsample_stages: int = 4,
    ):
        super().__init__()
        if int(upsample_stages) != 4:
            raise ValueError("RealMageAlbedoNetV2 currently requires exactly four 2x upsample stages")
        self.hidden_channels = int(hidden_channels)
        self.sampler_channels = int(sampler_channels)
        self.auxiliary_channels = int(auxiliary_channels)
        self.width = int(width)
        self.depth = int(depth)
        self.upsample_stages = int(upsample_stages)

        branch_width = self.width
        auxiliary_width = max(32, self.width // 2)
        self.hidden_proj = nn.Sequential(
            nn.Conv2d(self.hidden_channels, branch_width, 1, bias=False),
            nn.GroupNorm(_groups(branch_width), branch_width),
            nn.SiLU(),
        )
        self.sampler_proj = nn.Sequential(
            nn.Conv2d(self.sampler_channels, branch_width, 1, bias=False),
            nn.GroupNorm(_groups(branch_width), branch_width),
            nn.SiLU(),
        )
        self.auxiliary_proj = nn.Sequential(
            nn.Conv2d(self.auxiliary_channels, auxiliary_width, 1, bias=False),
            nn.GroupNorm(_groups(auxiliary_width), auxiliary_width),
            nn.SiLU(),
        )

        self.source_pyramid = SourcePyramid(self.width)
        source_channels = self.source_pyramid.channels
        self.low_fuse = nn.Conv2d(
            branch_width * 2 + auxiliary_width + source_channels[-1],
            self.width,
            1,
        )
        self.low_norm = nn.GroupNorm(_groups(self.width), self.width)
        dilations = (1, 2, 4, 8, 4, 2, 1)
        self.low_blocks = nn.ModuleList(
            [GatedSeparableBlock(self.width, dilations[index % len(dilations)]) for index in range(self.depth)]
        )
        self.context = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width * 2),
        )

        decoder_channels = (
            self.width,
            max(64, self.width * 3 // 4),
            max(48, self.width // 2),
            max(32, self.width // 3),
        )
        self.up3 = UpFuseBlock(self.width, source_channels[3], decoder_channels[0])
        self.up2 = UpFuseBlock(decoder_channels[0], source_channels[2], decoder_channels[1])
        self.up1 = UpFuseBlock(decoder_channels[1], source_channels[1], decoder_channels[2])
        self.up0 = UpFuseBlock(decoder_channels[2], source_channels[0], decoder_channels[3])
        final_channels = decoder_channels[3]
        self.refine = nn.Sequential(
            GatedSeparableBlock(final_channels),
            GatedSeparableBlock(final_channels),
        )
        self.head = nn.Sequential(
            nn.GroupNorm(_groups(final_channels), final_channels),
            nn.SiLU(),
            nn.Conv2d(final_channels, max(32, final_channels), 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(max(32, final_channels), 7, 3, padding=1),
        )

        self.register_buffer("hidden_mean", torch.zeros(self.hidden_channels), persistent=True)
        self.register_buffer("hidden_std", torch.ones(self.hidden_channels), persistent=True)
        self.register_buffer("sampler_mean", torch.zeros(self.sampler_channels), persistent=True)
        self.register_buffer("sampler_std", torch.ones(self.sampler_channels), persistent=True)
        self.register_buffer("auxiliary_mean", torch.zeros(self.auxiliary_channels), persistent=True)
        self.register_buffer("auxiliary_std", torch.ones(self.auxiliary_channels), persistent=True)

        final = self.head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[6] = 2.0

    def config(self) -> dict[str, int]:
        return {
            "hidden_channels": self.hidden_channels,
            "sampler_channels": self.sampler_channels,
            "auxiliary_channels": self.auxiliary_channels,
            "width": self.width,
            "depth": self.depth,
            "upsample_stages": self.upsample_stages,
        }

    def set_normalization(self, statistics: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name in (
                "hidden_mean",
                "hidden_std",
                "sampler_mean",
                "sampler_std",
                "auxiliary_mean",
                "auxiliary_std",
            ):
                target = getattr(self, name)
                value = statistics[name].to(device=target.device, dtype=target.dtype)
                if value.shape != target.shape:
                    raise ValueError(f"normalization shape mismatch for {name}: {value.shape} vs {target.shape}")
                target.copy_(value)

    @staticmethod
    def _normalize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (value.float() - mean[None, :, None, None]) / std[None, :, None, None].clamp_min(1e-4)

    def forward(
        self,
        hidden: torch.Tensor,
        sampler: torch.Tensor,
        auxiliary: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._normalize(hidden, self.hidden_mean, self.hidden_std)
        sampler = self._normalize(sampler, self.sampler_mean, self.sampler_std)
        auxiliary = self._normalize(auxiliary, self.auxiliary_mean, self.auxiliary_std)
        source_features = self.source_pyramid(source)
        low_source = source_features[-1]
        target_grid = hidden.shape[-2:]
        if tuple(low_source.shape[-2:]) != tuple(target_grid):
            low_source = F.interpolate(low_source, size=target_grid, mode="bilinear", align_corners=False)
        value = torch.cat(
            [self.hidden_proj(hidden), self.sampler_proj(sampler), self.auxiliary_proj(auxiliary), low_source],
            dim=1,
        )
        value = F.silu(self.low_norm(self.low_fuse(value)))
        scale, bias = self.context(value.mean(dim=(-2, -1))).chunk(2, dim=1)
        value = value * (1.0 + 0.15 * torch.tanh(scale)[:, :, None, None])
        value = value + bias[:, :, None, None]
        for block in self.low_blocks:
            value = block(value)
        value = self.up3(value, source_features[3])
        value = self.up2(value, source_features[2])
        value = self.up1(value, source_features[1])
        value = self.up0(value, source_features[0])
        value = self.refine(value)
        if tuple(value.shape[-2:]) != tuple(source.shape[-2:]):
            value = F.interpolate(value, size=source.shape[-2:], mode="bilinear", align_corners=False)
        raw = self.head(value)
        delta = 2.0 * torch.tanh(raw[:, :3])
        direct = torch.sigmoid(raw[:, 3:6])
        gate = torch.sigmoid(raw[:, 6:7])
        source_logits = torch.logit(source.float().clamp(1e-4, 1.0 - 1e-4))
        corrected = torch.sigmoid(source_logits + delta)
        return gate * corrected + (1.0 - gate) * direct

    def checkpoint(self, **metadata: Any) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "model": {name: value.detach().cpu() for name, value in self.state_dict().items()},
            "config": self.config(),
            **metadata,
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: dict[str, Any],
        device: str | torch.device = "cpu",
    ) -> "RealMageAlbedoNetV2":
        if payload.get("format") != cls.FORMAT:
            raise ValueError(f"unsupported checkpoint format: {payload.get('format')}")
        model = cls(**payload["config"])
        model.load_state_dict(payload["model"])
        return model.to(device)
