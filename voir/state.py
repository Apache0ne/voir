from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class ReservoirState:
    """Detached fixed-reservoir features in [T, L, C, H, W] format."""

    features: torch.Tensor
    output_size: tuple[int, int]
    sigmas: torch.Tensor
    layer_indices: tuple[int, ...]
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    aux_features: torch.Tensor | None = None

    def validate(self) -> "ReservoirState":
        if self.features.ndim != 5:
            raise ValueError(f"features must be [T,L,C,H,W], got {tuple(self.features.shape)}")
        if self.sigmas.ndim != 1:
            raise ValueError("sigmas must be one-dimensional")
        if self.features.shape[0] != self.sigmas.numel():
            raise ValueError("one model-evaluation sigma is required for every captured state")
        if self.features.shape[1] != len(self.layer_indices):
            raise ValueError("layer_indices does not match feature layer axis")
        if self.aux_features is not None:
            if self.aux_features.ndim != 3:
                raise ValueError("aux_features must be [C,H,W]")
            if not torch.isfinite(self.aux_features).all():
                raise ValueError("aux_features contain non-finite values")
        if not torch.isfinite(self.features).all():
            raise ValueError("features contain non-finite values")
        return self

    @property
    def trajectory_channels(self) -> int:
        return int(self.features.shape[0] * self.features.shape[1] * self.features.shape[2])

    @property
    def auxiliary_channels(self) -> int:
        return 0 if self.aux_features is None else int(self.aux_features.shape[0])

    @property
    def readout_channels(self) -> int:
        return self.trajectory_channels + self.auxiliary_channels

    def flattened(self, device: str | torch.device | None = None) -> torch.Tensor:
        """Return [1,C,H,W] with fixed source features first, then trajectory states."""
        self.validate()
        t, l, c, h, w = self.features.shape
        trajectory = self.features.reshape(1, t * l * c, h, w)
        if self.aux_features is None:
            x = trajectory
        else:
            aux = self.aux_features.unsqueeze(0)
            if tuple(aux.shape[-2:]) != (h, w):
                aux = F.interpolate(aux.float(), size=(h, w), mode="bilinear", align_corners=False)
            x = torch.cat([aux.to(dtype=trajectory.dtype), trajectory], dim=1)
        return x.to(device) if device is not None else x

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["features"] = self.features.detach().cpu()
        payload["sigmas"] = self.sigmas.detach().cpu()
        if self.aux_features is not None:
            payload["aux_features"] = self.aux_features.detach().cpu()
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "ReservoirState":
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        payload["output_size"] = tuple(payload["output_size"])
        payload["layer_indices"] = tuple(payload["layer_indices"])
        payload.setdefault("aux_features", None)
        return cls(**payload).validate()
