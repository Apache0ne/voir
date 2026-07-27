from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import CachedAlbedoDataset
from .device import resolve_device
from .losses import albedo_loss
from .metrics import albedo_metrics
from .readout import AlbedoReadout


@torch.no_grad()
def _evaluate(model: AlbedoReadout, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    samples = 0
    for features, target in loader:
        features = features.to(device)
        target = target.to(device)
        prediction = model(features, target.shape[-2:])
        metrics = albedo_metrics(prediction, target)
        batch = int(features.shape[0])
        samples += batch
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch
    return {name: value / max(samples, 1) for name, value in totals.items()}


def train_albedo(
    manifest: str,
    output: str,
    device: str = "auto",
    epochs: int = 20,
    batch_size: int = 2,
    learning_rate: float = 1.5e-3,
    width: int = 32,
    depth: int = 7,
    num_workers: int = 0,
    architecture: str = "dilated_v2",
    validation_manifest: str | None = None,
    seed: int = 42,
) -> dict:
    """Train only the albedo readout and retain the best held-out checkpoint."""
    dev = resolve_device(device)
    torch.manual_seed(int(seed))

    dataset = CachedAlbedoDataset(manifest)
    first_x, _ = dataset[0]
    model = AlbedoReadout(
        first_x.shape[0],
        width=width,
        depth=depth,
        architecture=architecture,
    ).to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    total_updates = max(1, int(epochs) * max(len(loader), 1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_updates,
        eta_min=max(float(learning_rate) * 0.02, 1e-6),
    )

    validation_loader = None
    if validation_manifest:
        validation_dataset = CachedAlbedoDataset(validation_manifest)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    history = []
    best_score = float("inf")
    best_state = None
    best_row = None

    for epoch in range(int(epochs)):
        model.train()
        running = 0.0
        batches = 0
        last_parts: dict[str, float] = {}
        for features, target in loader:
            features = features.to(dev)
            target = target.to(dev)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features, target.shape[-2:])
            loss, last_parts = albedo_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach())
            batches += 1

        row: dict[str, object] = {
            "epoch": epoch + 1,
            "loss": running / max(batches, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **last_parts,
        }
        if validation_loader is not None:
            validation = _evaluate(model, validation_loader, dev)
            row["validation"] = validation
            score = validation["mae"]
        else:
            score = float(row["loss"])

        if score < best_score:
            best_score = score
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            best_row = row
        history.append(row)
        print(json.dumps(row))

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = model.checkpoint()
    payload["history"] = history
    payload["best"] = best_row
    payload["seed"] = int(seed)
    payload["training_manifest"] = str(Path(manifest))
    payload["validation_manifest"] = validation_manifest
    torch.save(payload, out)
    return {
        "checkpoint": str(out),
        "history": history,
        "best": best_row,
        "device": str(dev),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
