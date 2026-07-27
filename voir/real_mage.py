from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def _pil_rgb(path: Path, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _pil_mask(path: Path, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.NEAREST)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).contiguous()


def _token_trajectory_to_map(value: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """Convert [T,1,N,C] or [T,N,C] tokens into [T*C,H,W]."""
    value = value.detach().cpu()
    if value.ndim == 4:
        if value.shape[1] != 1:
            raise ValueError(f"expected singleton packed batch axis, got {tuple(value.shape)}")
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError(f"expected token trajectory [T,N,C], got {tuple(value.shape)}")
    steps, tokens, channels = value.shape
    height, width = grid
    if tokens != height * width:
        raise ValueError(f"token count {tokens} does not match target grid {grid}")
    return (
        value.float()
        .reshape(steps, height, width, channels)
        .permute(0, 3, 1, 2)
        .reshape(steps * channels, height, width)
        .contiguous()
    )


def load_real_mage_cache(cache_path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    """Load one v1 cache into the tensors used by the real readout.

    The model consumes all projected transformer states and all three saved sampler
    trajectories: latent, denoised, and rectified-flow velocity. It also consumes
    the fixed 63-channel intrinsic feature bank and the full-resolution source RGB.
    """
    cache_path = Path(cache_path)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "voir_mage_real16_cache_v1":
        raise ValueError(f"unsupported cache format in {cache_path}")

    state = payload["reservoir_state"]
    sampler = payload["sampler_cache"]
    row = dict(payload.get("manifest_record") or {})
    dataset_record = dict(payload.get("dataset_record") or {})

    hidden = state["features"].float().contiguous()
    if hidden.ndim != 5:
        raise ValueError(f"projected states must be [T,L,C,H,W], got {tuple(hidden.shape)}")
    steps, layers, channels, grid_h, grid_w = hidden.shape
    hidden = hidden.reshape(steps * layers * channels, grid_h, grid_w).contiguous()

    grid = tuple(int(value) for value in sampler["target_grid"])
    if grid != (grid_h, grid_w):
        raise ValueError(f"sampler grid {grid} does not match hidden grid {(grid_h, grid_w)}")
    sampler_features = torch.cat(
        [
            _token_trajectory_to_map(sampler["eval_latents"], grid),
            _token_trajectory_to_map(sampler["eval_denoised"], grid),
            _token_trajectory_to_map(sampler["eval_velocity"], grid),
        ],
        dim=0,
    ).contiguous()

    auxiliary = state.get("aux_features")
    if auxiliary is None:
        raise ValueError("real Mage cache is missing the 63-channel auxiliary feature bank")
    auxiliary = auxiliary.float().contiguous()
    if tuple(auxiliary.shape[-2:]) != grid:
        auxiliary = F.interpolate(auxiliary.unsqueeze(0), size=grid, mode="bilinear", align_corners=False)[0]

    output_size = tuple(int(value) for value in state["output_size"])
    base = Path(root) if root is not None else cache_path.parents[1]
    source_path = base / row["input"]
    target_path = base / row["albedo"]
    mask_path = base / row["mask"]
    source = _pil_rgb(source_path, output_size)
    target = _pil_rgb(target_path, output_size)
    mask = _pil_mask(mask_path, output_size)

    for name, tensor in {
        "hidden": hidden,
        "sampler": sampler_features,
        "auxiliary": auxiliary,
        "source": source,
        "target": target,
        "mask": mask,
    }.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values in {cache_path}")

    return {
        "hidden": hidden,
        "sampler": sampler_features,
        "auxiliary": auxiliary,
        "source": source,
        "target": target,
        "mask": mask,
        "grid": grid,
        "output_size": output_size,
        "cache_path": str(cache_path),
        "index": int(row.get("index", dataset_record.get("index", -1))),
        "subset": str(dataset_record.get("subset", row.get("subset", "train"))),
        "scene": dataset_record.get("scene", row.get("scene")),
        "source_row": dataset_record.get("source_row", row.get("source_row")),
    }


class RealMageCacheDataset(Dataset):
    """Preloaded actual-Mage cache dataset with aligned grid/full-resolution crops."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        patch_grid: tuple[int, int] | None = (16, 24),
        augment: bool = True,
        repeats: int = 4,
    ):
        self.root = Path(root)
        self.split = str(split)
        self.patch_grid = patch_grid
        self.augment = bool(augment)
        self.repeats = max(1, int(repeats)) if self.split == "train" else 1
        manifest_path = self.root / "manifest.jsonl"
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        samples = []
        for row in rows:
            sample = load_real_mage_cache(self.root / row["cache"], root=self.root)
            if self.split == "all" or sample["subset"] == self.split:
                samples.append(sample)
        if not samples:
            raise ValueError(f"no {self.split!r} samples found under {self.root}")
        shapes = {
            (
                tuple(sample["hidden"].shape),
                tuple(sample["sampler"].shape),
                tuple(sample["auxiliary"].shape),
            )
            for sample in samples
        }
        if len(shapes) != 1:
            raise ValueError(f"cache feature shapes are inconsistent: {sorted(shapes)}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples) * self.repeats

    @property
    def base_length(self) -> int:
        return len(self.samples)

    @property
    def channels(self) -> dict[str, int]:
        sample = self.samples[0]
        return {
            "hidden": int(sample["hidden"].shape[0]),
            "sampler": int(sample["sampler"].shape[0]),
            "auxiliary": int(sample["auxiliary"].shape[0]),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index % len(self.samples)]
        hidden = sample["hidden"]
        sampler = sample["sampler"]
        auxiliary = sample["auxiliary"]
        source = sample["source"]
        target = sample["target"]
        mask = sample["mask"]
        grid_h, grid_w = hidden.shape[-2:]
        out_h, out_w = source.shape[-2:]

        if self.patch_grid is not None:
            patch_h = min(int(self.patch_grid[0]), grid_h)
            patch_w = min(int(self.patch_grid[1]), grid_w)
            if self.augment:
                top = int(torch.randint(0, grid_h - patch_h + 1, (1,)).item())
                left = int(torch.randint(0, grid_w - patch_w + 1, (1,)).item())
            else:
                top = (grid_h - patch_h) // 2
                left = (grid_w - patch_w) // 2
            bottom = top + patch_h
            right = left + patch_w
            y0 = round(top * out_h / grid_h)
            y1 = round(bottom * out_h / grid_h)
            x0 = round(left * out_w / grid_w)
            x1 = round(right * out_w / grid_w)
            hidden = hidden[:, top:bottom, left:right]
            sampler = sampler[:, top:bottom, left:right]
            auxiliary = auxiliary[:, top:bottom, left:right]
            source = source[:, y0:y1, x0:x1]
            target = target[:, y0:y1, x0:x1]
            mask = mask[:, y0:y1, x0:x1]

        if self.augment:
            if bool(torch.randint(0, 2, (1,)).item()):
                hidden = hidden.flip(-1)
                sampler = sampler.flip(-1)
                auxiliary = auxiliary.flip(-1)
                source = source.flip(-1)
                target = target.flip(-1)
                mask = mask.flip(-1)
            if bool(torch.randint(0, 2, (1,)).item()):
                hidden = hidden.flip(-2)
                sampler = sampler.flip(-2)
                auxiliary = auxiliary.flip(-2)
                source = source.flip(-2)
                target = target.flip(-2)
                mask = mask.flip(-2)

        return {
            "hidden": hidden.contiguous(),
            "sampler": sampler.contiguous(),
            "auxiliary": auxiliary.contiguous(),
            "source": source.contiguous(),
            "target": target.contiguous(),
            "mask": mask.contiguous(),
            "index": torch.tensor(int(sample["index"]), dtype=torch.long),
        }


def compute_branch_normalization(dataset: RealMageCacheDataset) -> dict[str, torch.Tensor]:
    """Compute per-channel mean/std from the non-augmented training cache tensors."""
    result: dict[str, torch.Tensor] = {}
    for branch in ("hidden", "sampler", "auxiliary"):
        channels = int(dataset.samples[0][branch].shape[0])
        total = torch.zeros(channels, dtype=torch.float64)
        square = torch.zeros(channels, dtype=torch.float64)
        count = 0
        for sample in dataset.samples:
            value = sample[branch].double()
            total += value.sum(dim=(-2, -1))
            square += value.square().sum(dim=(-2, -1))
            count += int(value.shape[-2] * value.shape[-1])
        mean = total / max(count, 1)
        variance = (square / max(count, 1) - mean.square()).clamp_min(1e-8)
        std = variance.sqrt().clamp_min(1e-4)
        result[f"{branch}_mean"] = mean.float()
        result[f"{branch}_std"] = std.float()
    return result


def _group_count(channels: int, maximum: int = 8) -> int:
    groups = min(int(channels), int(maximum))
    while groups > 1 and channels % groups:
        groups -= 1
    return groups


class SeparableResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        groups = _group_count(channels)
        self.norm = nn.GroupNorm(groups, channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            3,
            padding=int(dilation),
            dilation=int(dilation),
            groups=channels,
        )
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = F.silu(self.norm(value))
        residual = self.pointwise(self.depthwise(residual))
        return value + torch.tanh(self.gate) * residual


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.block = SeparableResidualBlock(out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, scale_factor=2.0, mode="bilinear", align_corners=False)
        value = F.silu(self.norm(self.conv(value)))
        return self.block(value)


class RealMageAlbedoNet(nn.Module):
    """Full-resolution albedo decoder trained on actual frozen Mage trajectories."""

    FORMAT = "voir_real_mage_albedo_v1"

    def __init__(
        self,
        hidden_channels: int,
        sampler_channels: int,
        auxiliary_channels: int = 63,
        width: int = 64,
        depth: int = 6,
        upsample_stages: int = 4,
    ):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.sampler_channels = int(sampler_channels)
        self.auxiliary_channels = int(auxiliary_channels)
        self.width = int(width)
        self.depth = int(depth)
        self.upsample_stages = int(upsample_stages)
        branch_width = max(24, self.width * 3 // 4)
        aux_width = max(16, self.width // 2)

        self.hidden_proj = nn.Sequential(
            nn.Conv2d(self.hidden_channels, branch_width, 1, bias=False),
            nn.GroupNorm(_group_count(branch_width), branch_width),
            nn.SiLU(),
        )
        self.sampler_proj = nn.Sequential(
            nn.Conv2d(self.sampler_channels, branch_width, 1, bias=False),
            nn.GroupNorm(_group_count(branch_width), branch_width),
            nn.SiLU(),
        )
        self.auxiliary_proj = nn.Sequential(
            nn.Conv2d(self.auxiliary_channels, aux_width, 1, bias=False),
            nn.GroupNorm(_group_count(aux_width), aux_width),
            nn.SiLU(),
        )
        self.fuse = nn.Conv2d(branch_width * 2 + aux_width, self.width, 1)
        dilations = (1, 2, 4, 8, 4, 2, 1)
        self.low_blocks = nn.ModuleList(
            [SeparableResidualBlock(self.width, dilations[index % len(dilations)]) for index in range(self.depth)]
        )
        self.context = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width * 2),
        )

        stage_channels = [self.width, self.width, max(32, self.width * 3 // 4), max(24, self.width // 2)]
        if self.upsample_stages > len(stage_channels):
            stage_channels.extend([stage_channels[-1]] * (self.upsample_stages - len(stage_channels)))
        stage_channels = stage_channels[: self.upsample_stages]
        upsample = []
        previous = self.width
        for channels in stage_channels:
            upsample.append(UpsampleBlock(previous, channels))
            previous = channels
        self.upsample = nn.ModuleList(upsample)
        self.source_stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(previous + 16, max(24, previous), 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(max(24, previous), 7, 3, padding=1),
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
        value = torch.cat(
            [self.hidden_proj(hidden), self.sampler_proj(sampler), self.auxiliary_proj(auxiliary)],
            dim=1,
        )
        value = self.fuse(value)
        scale, bias = self.context(value.mean(dim=(-2, -1))).chunk(2, dim=1)
        value = value * (1.0 + 0.1 * torch.tanh(scale)[:, :, None, None])
        value = value + bias[:, :, None, None]
        for block in self.low_blocks:
            value = block(value)
        for block in self.upsample:
            value = block(value)
        if tuple(value.shape[-2:]) != tuple(source.shape[-2:]):
            value = F.interpolate(value, size=source.shape[-2:], mode="bilinear", align_corners=False)
        raw = self.head(torch.cat([value, self.source_stem(source.float())], dim=1))
        delta = 1.5 * torch.tanh(raw[:, :3])
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
    ) -> "RealMageAlbedoNet":
        if payload.get("format") != cls.FORMAT:
            raise ValueError(f"unsupported checkpoint format: {payload.get('format')}")
        model = cls(**payload["config"])
        model.load_state_dict(payload["model"])
        return model.to(device)


def cosine_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max(1e-3, float(step + 1) / max(warmup_steps, 1))
    progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
