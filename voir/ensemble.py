from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from .readout import AlbedoReadout


class AlbedoReadoutEnsemble(nn.Module):
    """Average several tiny readouts while sharing one frozen feature tensor."""

    def __init__(self, models: Iterable[AlbedoReadout], weights: Iterable[float] | None = None):
        super().__init__()
        model_list = list(models)
        if not model_list:
            raise ValueError("at least one readout is required")
        in_channels = {model.in_channels for model in model_list}
        if len(in_channels) != 1:
            raise ValueError("all readouts must accept the same feature channel count")
        self.models = nn.ModuleList(model_list)
        if weights is None:
            weight_tensor = torch.ones(len(model_list), dtype=torch.float32)
        else:
            weight_tensor = torch.tensor(list(weights), dtype=torch.float32)
            if weight_tensor.numel() != len(model_list):
                raise ValueError("weights must match the number of models")
            if torch.any(weight_tensor < 0) or float(weight_tensor.sum()) <= 0:
                raise ValueError("weights must be nonnegative with a positive sum")
        self.register_buffer("weights", weight_tensor / weight_tensor.sum())

    def forward(
        self,
        features: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        outputs = [model(features, output_size) for model in self.models]
        stacked = torch.stack(outputs, dim=0)
        weight = self.weights.to(device=stacked.device, dtype=stacked.dtype)
        return (stacked * weight[:, None, None, None, None]).sum(dim=0)

    @classmethod
    def from_checkpoints(
        cls,
        paths: Iterable[str | Path],
        device: str | torch.device = "cpu",
        weights: Iterable[float] | None = None,
    ) -> "AlbedoReadoutEnsemble":
        models = []
        for path in paths:
            payload = torch.load(Path(path), map_location=device, weights_only=False)
            models.append(AlbedoReadout.from_checkpoint(payload, device))
        return cls(models, weights=weights).to(device)
