"""Five-fold group-held-out training on the 30 real Olbedo pairs.

This provides an honest out-of-fold measurement for the tiny real-photo set and
packages five diverse fold readouts plus one all-data readout for deployment.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_real_olbedo_cpu import (  # noqa: E402
    RealPair,
    _evaluate,
    _frozen_features,
    _identity_baseline,
    _load_pairs,
    _train_candidate,
)
from voir.albedo_features import AUXILIARY_ALBEDO_CHANNELS  # noqa: E402
from voir.ensemble import AlbedoReadoutEnsemble  # noqa: E402
from voir.metrics import albedo_metrics  # noqa: E402
from voir.readout import AlbedoReadout  # noqa: E402
from voir.reservoir import ToyImageReservoir  # noqa: E402


def _all_pairs(dataset_dir: Path, max_side: int) -> list[RealPair]:
    return _load_pairs(dataset_dir / "pairs.jsonl", max_side=max_side)


def _group_key(pair: RealPair) -> tuple[str, str, str]:
    item = pair.metadata
    return str(item.get("scene", "")), str(item.get("date", "")), str(item.get("lighting", ""))


def _group_folds(pairs: list[RealPair], folds: int = 5) -> list[list[int]]:
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        grouped[_group_key(pair)].append(index)
    fold_indices: list[list[int]] = [[] for _ in range(folds)]
    fold_sizes = [0] * folds
    # Largest groups first, assigned to the currently smallest fold. This keeps
    # every scene/date/lighting group wholly held out.
    for key, indices in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        destination = min(range(folds), key=lambda fold: (fold_sizes[fold], fold))
        fold_indices[destination].extend(indices)
        fold_sizes[destination] += len(indices)
    return [sorted(indices) for indices in fold_indices]


def _mean_metric_dict(values: list[dict[str, float]]) -> dict[str, float]:
    names = values[0].keys()
    return {name: sum(value[name] for value in values) / len(values) for name in names}


@torch.no_grad()
def _predict(model: AlbedoReadout, reservoir: ToyImageReservoir, pair: RealPair) -> torch.Tensor:
    return model(_frozen_features(reservoir, pair.image), pair.albedo.shape[-2:])[0]


@torch.no_grad()
def _trajectory_ablation(
    model: AlbedoReadout,
    reservoir: ToyImageReservoir,
    pairs: list[RealPair],
) -> dict[str, dict[str, float]]:
    full_metrics = []
    zero_metrics = []
    trajectory_channels = reservoir.channels * reservoir.steps
    for pair in pairs:
        features = _frozen_features(reservoir, pair.image)
        full = model(features, pair.albedo.shape[-2:])
        zero_features = features.clone()
        zero_features[:, :trajectory_channels] = 0
        zero = model(zero_features, pair.albedo.shape[-2:])
        target = pair.albedo.unsqueeze(0)
        mask = pair.mask.unsqueeze(0)
        full_metrics.append(albedo_metrics(full, target, mask))
        zero_metrics.append(albedo_metrics(zero, target, mask))
    return {
        "full": _mean_metric_dict(full_metrics),
        "zero_trajectory": _mean_metric_dict(zero_metrics),
    }


def _save_model(path: Path, model: AlbedoReadout, extra: dict) -> None:
    payload = model.checkpoint()
    payload.update(extra)
    torch.save(payload, path)


def _write_oof_sheet(
    path: Path,
    pairs: list[RealPair],
    predictions: list[torch.Tensor],
    sample_count: int = 15,
) -> None:
    order = np.linspace(0, len(pairs) - 1, min(sample_count, len(pairs))).round().astype(int).tolist()
    width, row_height, header = 240, 160, 32
    canvas = Image.new("RGB", (width * 6, header + row_height * len(order)), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ("Real RGB", "Olbedo target", "RGB error x4", "OOF VOIR v3", "OOF error x4", "Valid mask")
    for column, label in enumerate(labels):
        draw.text((column * width + 6, 9), label, fill="black")
    for row, pair_index in enumerate(order):
        pair = pairs[pair_index]
        prediction = predictions[pair_index]
        tensors = (
            pair.image,
            pair.albedo,
            (pair.image - pair.albedo).abs().mul(4).clamp(0, 1),
            prediction,
            (prediction - pair.albedo).abs().mul(4).clamp(0, 1),
            pair.mask.expand(3, -1, -1),
        )
        for column, tensor in enumerate(tensors):
            array = tensor.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
            image = Image.fromarray(array).resize((width, row_height), Image.Resampling.BICUBIC)
            canvas.paste(image, (column * width, header + row * row_height))
    canvas.save(path)


def main() -> None:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started = time.time()
    dataset_dir = ROOT / "datasets/hf_olbedo_real30"
    output_dir = ROOT / "outputs/real_olbedo_cpu_v3_crossval"
    model_dir = output_dir / "fold_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    pairs = _all_pairs(dataset_dir, max_side=320)
    folds = _group_folds(pairs, folds=5)
    reservoir = ToyImageReservoir(channels=16, steps=4, seed=1234)
    assert all(not parameter.requires_grad for parameter in reservoir.parameters())
    config = {"architecture": "intrinsic_v3", "width": 24, "depth": 6}

    fold_reports = []
    fold_models: list[AlbedoReadout] = []
    oof_predictions: list[torch.Tensor | None] = [None] * len(pairs)
    oof_metrics = []

    for fold, validation_indices in enumerate(folds):
        validation_set = set(validation_indices)
        train_pairs = [pair for index, pair in enumerate(pairs) if index not in validation_set]
        validation_pairs = [pairs[index] for index in validation_indices]

        model, phase1 = _train_candidate(
            config,
            reservoir,
            train_pairs,
            validation_pairs,
            steps=300,
            batch_size=3,
            crop_sizes=(112, 128, 144),
            learning_rate=8e-4,
            seed=10_000 + fold,
            eval_every=75,
            augmentation_strength=0.75,
        )
        model, phase2 = _train_candidate(
            config,
            reservoir,
            train_pairs,
            validation_pairs,
            steps=450,
            batch_size=3,
            crop_sizes=(128, 160, 192),
            learning_rate=2.5e-4,
            seed=20_000 + fold,
            eval_every=90,
            augmentation_strength=0.40,
            initial_state=copy.deepcopy(model.state_dict()),
        )
        fold_metric_rows = []
        for pair_index in validation_indices:
            prediction = _predict(model, reservoir, pairs[pair_index])
            oof_predictions[pair_index] = prediction.cpu()
            metric = albedo_metrics(
                prediction.unsqueeze(0),
                pairs[pair_index].albedo.unsqueeze(0),
                pairs[pair_index].mask.unsqueeze(0),
            )
            fold_metric_rows.append(metric)
            oof_metrics.append(metric)
        fold_metrics = _mean_metric_dict(fold_metric_rows)
        model_path = model_dir / f"fold_{fold}.pt"
        _save_model(
            model_path,
            model,
            {
                "fold": fold,
                "held_out_indices": validation_indices,
                "held_out_groups": sorted({"|".join(_group_key(pairs[index])) for index in validation_indices}),
                "phase1": phase1,
                "phase2": phase2,
                "held_out_metrics": fold_metrics,
                "frozen_reservoir": True,
            },
        )
        fold_models.append(model)
        fold_reports.append(
            {
                "fold": fold,
                "train_count": len(train_pairs),
                "validation_count": len(validation_pairs),
                "validation_indices": validation_indices,
                "validation_groups": sorted({"|".join(_group_key(pairs[index])) for index in validation_indices}),
                "phase1": phase1,
                "phase2": phase2,
                "metrics": fold_metrics,
                "checkpoint": str(model_path.relative_to(ROOT)),
            }
        )

    if any(prediction is None for prediction in oof_predictions):
        raise RuntimeError("not every real pair received an out-of-fold prediction")
    oof_predictions_final = [prediction for prediction in oof_predictions if prediction is not None]
    oof_average = _mean_metric_dict(oof_metrics)
    identity = _identity_baseline(pairs)
    _write_oof_sheet(output_dir / "oof_comparison.png", pairs, oof_predictions_final)

    # Train one final model on all thirty pairs using the fold-selected schedule.
    # It is included alongside the diverse fold models for deployment, but is not
    # used in the honest out-of-fold metric above.
    final_model, final_phase1 = _train_candidate(
        config,
        reservoir,
        pairs,
        pairs,
        steps=300,
        batch_size=3,
        crop_sizes=(112, 128, 144),
        learning_rate=8e-4,
        seed=31_415,
        eval_every=75,
        augmentation_strength=0.75,
    )
    final_model, final_phase2 = _train_candidate(
        config,
        reservoir,
        pairs,
        pairs,
        steps=450,
        batch_size=3,
        crop_sizes=(128, 160, 192),
        learning_rate=2.5e-4,
        seed=27_182,
        eval_every=90,
        augmentation_strength=0.40,
        initial_state=copy.deepcopy(final_model.state_dict()),
    )
    final_path = model_dir / "all30.pt"
    _save_model(
        final_path,
        final_model,
        {
            "training": "all 30 real pairs after group-held-out hyperparameter selection",
            "phase1": final_phase1,
            "phase2": final_phase2,
            "frozen_reservoir": True,
        },
    )

    # Six readouts total only about 250k trainable parameters. New-image
    # inference averages them after one shared frozen feature extraction.
    deployment_models = fold_models + [final_model]
    deployment_ensemble = AlbedoReadoutEnsemble(deployment_models)
    ensemble_manifest = {
        "architecture": "mean_readout_ensemble",
        "members": [str((model_dir / f"fold_{fold}.pt").relative_to(output_dir)) for fold in range(5)]
        + [str(final_path.relative_to(output_dir))],
        "weights": [1.0 / 6.0] * 6,
        "shared_feature_extraction": True,
        "member_parameters": [sum(parameter.numel() for parameter in model.parameters()) for model in deployment_models],
        "total_readout_parameters": sum(
            sum(parameter.numel() for parameter in model.parameters()) for model in deployment_models
        ),
    }
    (output_dir / "ensemble.json").write_text(json.dumps(ensemble_manifest, indent=2), encoding="utf-8")

    ablation = _trajectory_ablation(final_model, reservoir, pairs)
    report = {
        "dataset": "GDAOSU/Olbedo",
        "pairs": len(pairs),
        "split_method": "five group-held-out folds; scene/date/lighting group never crosses a fold",
        "fold_sizes": [len(indices) for indices in folds],
        "architecture": config,
        "folds": fold_reports,
        "identity_baseline_all30": identity,
        "out_of_fold_metrics_all30": oof_average,
        "out_of_fold_improvement_over_rgb": {
            "mae_percent": 100.0 * (identity["mae"] - oof_average["mae"]) / max(identity["mae"], 1e-9),
            "mse_percent": 100.0 * (identity["mse"] - oof_average["mse"]) / max(identity["mse"], 1e-9),
            "psnr_db": oof_average["psnr"] - identity["psnr"],
            "ssim_7x7": oof_average["ssim_7x7"] - identity["ssim_7x7"],
            "global_ssim": oof_average["global_ssim"] - identity["global_ssim"],
        },
        "all30_training": {"phase1": final_phase1, "phase2": final_phase2},
        "trajectory_ablation_all30_model": ablation,
        "deployment_ensemble": ensemble_manifest,
        "reservoir_parameters_trained": 0,
        "seconds": time.time() - started,
        "oof_comparison": str((output_dir / "oof_comparison.png").relative_to(ROOT)),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
