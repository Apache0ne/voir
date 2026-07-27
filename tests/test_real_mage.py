from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from voir.real_mage import RealMageAlbedoNet, RealMageCacheDataset, compute_branch_normalization


def _write_rgb(path: Path, height: int, width: int, value: int) -> None:
    array = np.full((height, width, 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def test_real_cache_dataset_and_model(tmp_path: Path):
    root = tmp_path / "mage_cache" / "real16"
    for name in ("states", "inputs", "albedo", "masks"):
        (root / name).mkdir(parents=True, exist_ok=True)
    _write_rgb(root / "inputs/000.png", 64, 80, 100)
    _write_rgb(root / "albedo/000.png", 64, 80, 140)
    Image.fromarray(np.full((64, 80), 255, dtype=np.uint8), mode="L").save(root / "masks/000.png")

    steps, layers, hidden_channels = 2, 2, 3
    grid = (4, 5)
    token_channels = 4
    trajectory = torch.randn(steps, 1, grid[0] * grid[1], token_channels)
    row = {
        "index": 0,
        "input": "inputs/000.png",
        "albedo": "albedo/000.png",
        "mask": "masks/000.png",
        "cache": "states/000.pt",
    }
    payload = {
        "format": "voir_mage_real16_cache_v1",
        "reservoir_state": {
            "features": torch.randn(steps, layers, hidden_channels, *grid),
            "output_size": (64, 80),
            "sigmas": torch.tensor([1.0, 0.5]),
            "layer_indices": (0, 1),
            "source": "test",
            "metadata": {},
            "aux_features": torch.randn(3, *grid),
        },
        "sampler_cache": {
            "target_grid": grid,
            "eval_latents": trajectory,
            "eval_denoised": trajectory + 0.1,
            "eval_velocity": trajectory - 0.1,
        },
        "dataset_record": {"index": 0, "subset": "train"},
        "manifest_record": row,
    }
    torch.save(payload, root / "states/000.pt")
    (root / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (root / "run.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")

    dataset = RealMageCacheDataset(root, split="train", patch_grid=None, augment=False, repeats=1)
    sample = dataset[0]
    assert sample["hidden"].shape == (steps * layers * hidden_channels, *grid)
    assert sample["sampler"].shape == (steps * token_channels * 3, *grid)
    assert sample["auxiliary"].shape == (3, *grid)
    assert sample["source"].shape == (3, 64, 80)

    stats = compute_branch_normalization(dataset)
    model = RealMageAlbedoNet(
        hidden_channels=sample["hidden"].shape[0],
        sampler_channels=sample["sampler"].shape[0],
        auxiliary_channels=sample["auxiliary"].shape[0],
        width=24,
        depth=2,
        upsample_stages=4,
    )
    model.set_normalization(stats)
    prediction = model(
        sample["hidden"].unsqueeze(0),
        sample["sampler"].unsqueeze(0),
        sample["auxiliary"].unsqueeze(0),
        sample["source"].unsqueeze(0),
    )
    assert prediction.shape == (1, 3, 64, 80)
    assert torch.isfinite(prediction).all()
    assert prediction.min() >= 0
    assert prediction.max() <= 1

    restored = RealMageAlbedoNet.from_checkpoint(model.checkpoint())
    restored_prediction = restored(
        sample["hidden"].unsqueeze(0),
        sample["sampler"].unsqueeze(0),
        sample["auxiliary"].unsqueeze(0),
        sample["source"].unsqueeze(0),
    )
    torch.testing.assert_close(restored_prediction, prediction)
