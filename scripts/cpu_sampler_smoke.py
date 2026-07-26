"""CPU proof for the standalone Beta + DPM++ SDE sampler and frozen readout flow."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from voir.losses import albedo_loss
from voir.readout import AlbedoReadout
from voir.reservoir import ToyImageReservoir
from voir.sampling import sample_dpmpp_sde_gpu
from voir.schedule import beta_flow_sigmas
from voir.synthetic import synthetic_albedo_batch


def main():
    torch.manual_seed(0)
    sigmas = beta_flow_sigmas(steps=4, alpha=0.60, beta=0.80)
    latent = torch.randn(1, 8, 16, 16)
    fixed_x0 = torch.tanh(torch.randn_like(latent))

    def frozen_flow_x0(x, sigma):
        s = sigma.reshape(-1, 1, 1, 1)
        return fixed_x0 + 0.025 * s * torch.tanh(x)

    sampled, trace = sample_dpmpp_sde_gpu(
        frozen_flow_x0,
        latent,
        sigmas,
        seed=123,
        prefer_torchsde=False,
    )
    assert torch.isfinite(sampled).all()

    observed, target = synthetic_albedo_batch(batch=8, size=48, seed=5)
    reservoir = ToyImageReservoir(channels=16, steps=4)
    states = [reservoir.capture(observed[i]).flattened().squeeze(0) for i in range(len(observed))]
    features = torch.stack(states)
    readout = AlbedoReadout(features.shape[1], width=32, depth=1)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=2e-3)
    initial = None
    for _ in range(20):
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
    report = {
        "device": "cpu",
        "sampler": trace.sampler,
        "beta_sigmas": trace.schedule_sigmas.tolist(),
        "model_evaluations": len(trace.model_eval_sigmas),
        "model_eval_sigmas": trace.model_eval_sigmas.tolist(),
        "noise_backend": trace.noise_backend,
        "sampler_output_mean": float(sampled.mean()),
        "initial_albedo_loss": initial,
        "final_albedo_loss": float(loss.detach()),
    }
    (out_dir / "cpu_sampler_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
