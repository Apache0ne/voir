"""Focused long CPU refinement using the architecture selected by the real30 search."""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_real_olbedo_cpu import (  # noqa: E402
    _evaluate,
    _identity_baseline,
    _load_pairs,
    _train_candidate,
    _write_sheet,
)
from voir.reservoir import ToyImageReservoir  # noqa: E402


def main() -> None:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started = time.time()
    dataset_dir = ROOT / "datasets/hf_olbedo_real30"
    output_dir = ROOT / "outputs/real_olbedo_cpu_v3_focused"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep validation at a higher resolution than the quick architecture search.
    train_pairs = _load_pairs(dataset_dir / "train_pairs.jsonl", max_side=384)
    validation_pairs = _load_pairs(dataset_dir / "validation_pairs.jsonl", max_side=384)
    reservoir = ToyImageReservoir(channels=16, steps=4, seed=1234)
    assert all(not parameter.requires_grad for parameter in reservoir.parameters())

    config = {"architecture": "intrinsic_v3", "width": 24, "depth": 6}
    phases = []
    model, phase = _train_candidate(
        config,
        reservoir,
        train_pairs,
        validation_pairs,
        steps=450,
        batch_size=3,
        crop_sizes=(112, 128, 144),
        learning_rate=8e-4,
        seed=5001,
        eval_every=75,
        augmentation_strength=0.80,
    )
    phases.append({"name": "illumination_invariance", **phase})

    model, phase = _train_candidate(
        config,
        reservoir,
        train_pairs,
        validation_pairs,
        steps=800,
        batch_size=3,
        crop_sizes=(128, 160, 192),
        learning_rate=2.5e-4,
        seed=6001,
        eval_every=100,
        augmentation_strength=0.45,
        initial_state=copy.deepcopy(model.state_dict()),
    )
    phases.append({"name": "high_resolution_refinement", **phase})

    model, phase = _train_candidate(
        config,
        reservoir,
        train_pairs,
        validation_pairs,
        steps=500,
        batch_size=2,
        crop_sizes=(160, 192, 224),
        learning_rate=8e-5,
        seed=7001,
        eval_every=100,
        augmentation_strength=0.20,
        initial_state=copy.deepcopy(model.state_dict()),
    )
    phases.append({"name": "low_rate_polish", **phase})

    identity = _identity_baseline(validation_pairs)
    final_metrics = _evaluate(model, reservoir, validation_pairs)
    checkpoint = model.checkpoint()
    checkpoint.update(
        {
            "dataset": "GDAOSU/Olbedo real30",
            "training": "focused three-phase real-photo CPU refinement",
            "phases": phases,
            "identity_metrics": identity,
            "validation_metrics": final_metrics,
            "frozen_reservoir": {
                "type": "ToyImageReservoir CPU surrogate",
                "channels": reservoir.channels,
                "steps": reservoir.steps,
                "trainable": False,
            },
            "mage_transfer": {
                "reusable_modules": [
                    "auxiliary_proj",
                    "fuse",
                    "blocks",
                    "global_fc",
                    "head_norm",
                    "correction_head",
                    "direct_head",
                    "gate_head",
                ],
                "replace_for_mage": "trajectory_proj",
            },
        }
    )
    checkpoint_path = output_dir / "voir_real_olbedo_v3_focused.pt"
    torch.save(checkpoint, checkpoint_path)
    comparison_path = output_dir / "comparison.png"
    _write_sheet(comparison_path, model, reservoir, validation_pairs)

    report = {
        "dataset": "GDAOSU/Olbedo",
        "real_train_pairs": len(train_pairs),
        "real_validation_pairs": len(validation_pairs),
        "max_side": 384,
        "architecture": config,
        "phases": phases,
        "identity_baseline": identity,
        "final_metrics": final_metrics,
        "improvement_over_rgb": {
            "mae_percent": 100.0 * (identity["mae"] - final_metrics["mae"]) / max(identity["mae"], 1e-9),
            "mse_percent": 100.0 * (identity["mse"] - final_metrics["mse"]) / max(identity["mse"], 1e-9),
            "psnr_db": final_metrics["psnr"] - identity["psnr"],
            "ssim_7x7": final_metrics["ssim_7x7"] - identity["ssim_7x7"],
            "global_ssim": final_metrics["global_ssim"] - identity["global_ssim"],
        },
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "reservoir_parameters_trained": 0,
        "seconds": time.time() - started,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "comparison": str(comparison_path.relative_to(ROOT)),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
