from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import CachedAlbedoDataset
from .device import resolve_device
from .losses import albedo_loss
from .readout import AlbedoReadout


def train_albedo(
    manifest: str,
    output: str,
    device: str = "auto",
    epochs: int = 10,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    width: int = 96,
    depth: int = 4,
    num_workers: int = 0,
) -> dict:
    dev = resolve_device(device)
    dataset = CachedAlbedoDataset(manifest)
    first_x, _ = dataset[0]
    model = AlbedoReadout(first_x.shape[0], width=width, depth=depth).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    history = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        parts = {}
        for features, target in loader:
            features, target = features.to(dev), target.to(dev)
            optimizer.zero_grad(set_to_none=True)
            pred = model(features, target.shape[-2:])
            loss, parts = albedo_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
        row = {"epoch": epoch + 1, "loss": running / len(loader), **parts}
        history.append(row)
        print(json.dumps(row))
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = model.checkpoint()
    payload["history"] = history
    torch.save(payload, out)
    return {"checkpoint": str(out), "history": history, "device": str(dev)}
