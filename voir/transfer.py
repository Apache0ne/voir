from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .readout import AlbedoReadout


def transfer_intrinsic_readout(
    checkpoint: dict[str, Any] | str | Path,
    *,
    trajectory_channels: int,
    auxiliary_channels: int = 63,
    device: str | torch.device = "cpu",
    freeze_shared: bool = True,
) -> tuple[AlbedoReadout, dict[str, Any]]:
    """Transfer a CPU-trained ``intrinsic_v3`` decoder to Mage state width.

    All image-feature and decoder weights are copied exactly. The new
    ``trajectory_proj`` input weight is initialized to zero because CPU surrogate
    channels and Mage hidden-state channels do not share a basis. Consequently,
    the transferred model starts as the validated auxiliary-only predictor rather
    than injecting arbitrary Mage-state noise. A short projection-only warm-up can
    then learn the Mage basis before the complete readout is unfrozen.
    """
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    else:
        payload = checkpoint
    config = dict(payload["config"])
    if config.get("architecture") != "intrinsic_v3":
        raise ValueError("only intrinsic_v3 checkpoints support trajectory-width transfer")

    old_trajectory_channels = int(config["trajectory_channels"])
    new_trajectory_channels = int(trajectory_channels)
    if new_trajectory_channels < 1:
        raise ValueError("trajectory_channels must be positive")
    if int(auxiliary_channels) != int(config.get("auxiliary_channels", auxiliary_channels)):
        raise ValueError("auxiliary channel count must match the trained fixed feature bank")

    new_config = dict(config)
    new_config["trajectory_channels"] = new_trajectory_channels
    new_config["auxiliary_channels"] = int(auxiliary_channels)
    new_config["in_channels"] = new_trajectory_channels + int(auxiliary_channels)
    model = AlbedoReadout(**new_config)
    source_state = payload["model"]
    target_state = model.state_dict()
    copied: list[str] = []
    reinitialized: list[str] = []

    for name, target in target_state.items():
        source = source_state.get(name)
        if source is not None and tuple(source.shape) == tuple(target.shape):
            target.copy_(source)
            copied.append(name)
        else:
            reinitialized.append(name)

    # Zero input projection preserves the auxiliary-only function exactly. Bias,
    # normalization, and all subsequent decoder weights were copied above.
    with torch.no_grad():
        model.trajectory_proj[0].weight.zero_()
    if "trajectory_proj.0.weight" not in reinitialized:
        reinitialized.append("trajectory_proj.0.weight")

    if freeze_shared:
        model.requires_grad_(False)
        model.trajectory_proj.requires_grad_(True)
    model = model.to(device)
    report = {
        "architecture": "intrinsic_v3",
        "old_trajectory_channels": old_trajectory_channels,
        "new_trajectory_channels": new_trajectory_channels,
        "auxiliary_channels": int(auxiliary_channels),
        "copied_tensors": copied,
        "reinitialized_tensors": sorted(set(reinitialized)),
        "trajectory_initialization": "zero; auxiliary-only function preserved",
        "freeze_shared": bool(freeze_shared),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, report


def save_transferred_readout(
    output: str | Path,
    model: AlbedoReadout,
    transfer_report: dict[str, Any],
    source_checkpoint: str | Path | None = None,
) -> None:
    payload = model.checkpoint()
    payload["transfer"] = transfer_report
    if source_checkpoint is not None:
        payload["source_checkpoint"] = str(source_checkpoint)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
