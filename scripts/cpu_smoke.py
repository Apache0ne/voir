"""End-to-end CPU proof: fixed nonlinear reservoir, trainable albedo readout only."""
from pathlib import Path

import torch
from PIL import Image

from voir.losses import albedo_loss
from voir.readout import AlbedoReadout
from voir.reservoir import ToyImageReservoir
from voir.synthetic import synthetic_albedo_batch


def main():
    torch.manual_seed(0)
    observed, target = synthetic_albedo_batch(batch=8, size=64, seed=5)
    reservoir = ToyImageReservoir(channels=24, steps=4)
    states = [reservoir.capture(observed[i]).flattened().squeeze(0) for i in range(len(observed))]
    features = torch.stack(states)
    readout = AlbedoReadout(features.shape[1], width=48, depth=2)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=2e-3)
    initial = None
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        pred = readout(features, target.shape[-2:])
        loss, _ = albedo_loss(pred, target)
        initial = float(loss.detach()) if initial is None else initial
        loss.backward()
        optimizer.step()
    assert all(p.grad is None for p in reservoir.parameters())
    assert float(loss.detach()) < initial
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    image = (pred[0].detach().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
    Image.fromarray(image).save(out_dir / "cpu_albedo_smoke.png")
    torch.save(readout.checkpoint(), out_dir / "cpu_albedo_readout.pt")
    print({
        "initial_loss": initial,
        "final_loss": float(loss.detach()),
        "features": tuple(features.shape),
    })


if __name__ == "__main__":
    main()
