"""Train and select the VOIR readout on real paired Olbedo photographs.

The Mage backbone is not executed on CPU. This benchmark freezes the same
trajectory/readout interface using ToyImageReservoir, trains only the readout,
and keeps the 63-channel auxiliary branch directly transferable to Mage.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from voir.albedo_features import AUXILIARY_ALBEDO_CHANNELS, fixed_albedo_features
from voir.losses import masked_albedo_loss, masked_mean
from voir.metrics import albedo_metrics
from voir.readout import AlbedoReadout
from voir.reservoir import ToyImageReservoir


@dataclass
class RealPair:
    image: torch.Tensor
    albedo: torch.Tensor
    mask: torch.Tensor
    metadata: dict


def _pil_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _pil_mask(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    mask = torch.from_numpy(array).unsqueeze(0).contiguous()
    # The Olbedo mask is a validity/segmentation preview. Preserve soft edges but
    # ensure a corrupted all-zero mask cannot erase a complete sample.
    return torch.ones_like(mask) if float(mask.mean()) < 1e-4 else mask


def _resize_triplet(
    image: torch.Tensor,
    albedo: torch.Tensor,
    mask: torch.Tensor,
    max_side: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = image.shape[-2:]
    scale = min(1.0, float(max_side) / max(height, width))
    out_h = max(32, round(height * scale / 8) * 8)
    out_w = max(32, round(width * scale / 8) * 8)
    if (out_h, out_w) == (height, width):
        return image, albedo, mask
    image = F.interpolate(image.unsqueeze(0), (out_h, out_w), mode="bicubic", align_corners=False, antialias=True)[0]
    albedo = F.interpolate(albedo.unsqueeze(0), (out_h, out_w), mode="bicubic", align_corners=False, antialias=True)[0]
    mask = F.interpolate(mask.unsqueeze(0), (out_h, out_w), mode="bilinear", align_corners=False)[0]
    return image.clamp(0, 1), albedo.clamp(0, 1), mask.clamp(0, 1)


def _load_pairs(manifest: Path, max_side: int) -> list[RealPair]:
    base = manifest.parent
    pairs: list[RealPair] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        image = _pil_rgb(base / record["image"])
        albedo = _pil_rgb(base / record["albedo"])
        mask = _pil_mask(base / record["mask"])
        image, albedo, mask = _resize_triplet(image, albedo, mask, max_side)
        pairs.append(RealPair(image, albedo, mask, record))
    if not pairs:
        raise ValueError(f"manifest contains no pairs: {manifest}")
    return pairs


@torch.no_grad()
def _frozen_features(reservoir: ToyImageReservoir, image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    base = reservoir.input_conv(image)
    state = torch.zeros_like(base)
    trajectory = []
    for leak in torch.linspace(0.35, 0.80, reservoir.steps).tolist():
        state = torch.tanh(
            base
            + float(leak) * reservoir.recurrent_conv(state)
            + 0.15 * reservoir.mix_conv(state)
        )
        trajectory.append(state)
    trajectory_flat = torch.cat(trajectory, dim=1)
    auxiliary = fixed_albedo_features(image)
    return torch.cat([trajectory_flat, auxiliary], dim=1)


def _crop_with_mask(
    pair: RealPair,
    crop_size: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image, albedo, mask = pair.image, pair.albedo, pair.mask
    height, width = image.shape[-2:]
    size = min(crop_size, height, width)
    top = left = 0
    for _ in range(12):
        top = rng.randint(0, height - size) if height > size else 0
        left = rng.randint(0, width - size) if width > size else 0
        candidate = mask[:, top : top + size, left : left + size]
        if float(candidate.mean()) >= 0.35:
            break
    image = image[:, top : top + size, left : left + size]
    albedo = albedo[:, top : top + size, left : left + size]
    mask = mask[:, top : top + size, left : left + size]
    if rng.random() < 0.5:
        image = image.flip(-1)
        albedo = albedo.flip(-1)
        mask = mask.flip(-1)
    if rng.random() < 0.15:
        image = image.flip(-2)
        albedo = albedo.flip(-2)
        mask = mask.flip(-2)
    return image, albedo, mask


def _photometric_augmentation(
    image: torch.Tensor,
    generator: torch.Generator,
    strength: float,
) -> torch.Tensor:
    batch = image.shape[0]
    exposure = torch.pow(
        torch.tensor(2.0),
        (torch.rand(batch, 1, 1, 1, generator=generator) * 2.0 - 1.0) * 0.35 * strength,
    )
    gamma = 1.0 + (torch.rand(batch, 1, 1, 1, generator=generator) * 2.0 - 1.0) * 0.18 * strength
    cast = 1.0 + (torch.rand(batch, 3, 1, 1, generator=generator) * 2.0 - 1.0) * 0.14 * strength
    coarse = torch.rand(batch, 1, 5, 5, generator=generator)
    field = F.interpolate(coarse, size=image.shape[-2:], mode="bicubic", align_corners=False)
    field = 1.0 + (field - field.mean(dim=(-2, -1), keepdim=True)) * 0.65 * strength
    augmented = image.clamp_min(1e-4).pow(gamma) * exposure * cast * field

    # Smooth colored highlight, useful for teaching the readout not to bake
    # illumination into albedo while remaining fully paired with the real target.
    if strength > 0:
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, image.shape[-2]),
            torch.linspace(-1, 1, image.shape[-1]),
            indexing="ij",
        )
        center = torch.rand(batch, 2, generator=generator) * 1.5 - 0.75
        radius = 0.20 + torch.rand(batch, 1, generator=generator) * 0.45
        distance = (
            (xx[None] - center[:, 0, None, None]).square()
            + (yy[None] - center[:, 1, None, None]).square()
        )
        blob = torch.exp(-distance / radius[:, :, None].square()).unsqueeze(1)
        color = torch.rand(batch, 3, 1, 1, generator=generator)
        augmented = augmented + blob * color * (0.10 * strength)
    return augmented.clamp(0.0, 1.0)


def _shading_smoothness(
    prediction: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    log_shading = torch.log(source.clamp_min(2e-3)) - torch.log(prediction.clamp_min(2e-3))
    sx = log_shading[..., :, 1:] - log_shading[..., :, :-1]
    sy = log_shading[..., 1:, :] - log_shading[..., :-1, :]
    tx = (target[..., :, 1:] - target[..., :, :-1]).abs().mean(dim=1, keepdim=True)
    ty = (target[..., 1:, :] - target[..., :-1, :]).abs().mean(dim=1, keepdim=True)
    mx = torch.minimum(mask[..., :, 1:], mask[..., :, :-1])
    my = torch.minimum(mask[..., 1:, :], mask[..., :-1, :])
    return masked_mean(sx.abs() * torch.exp(-8.0 * tx), mx) + masked_mean(
        sy.abs() * torch.exp(-8.0 * ty), my
    )


def _training_batch(
    pairs: list[RealPair],
    batch_size: int,
    crop_sizes: tuple[int, ...],
    step_seed: int,
    augmentation_strength: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = random.Random(step_seed)
    crop_size = crop_sizes[rng.randrange(len(crop_sizes))]
    images, targets, masks = [], [], []
    for _ in range(batch_size):
        pair = pairs[rng.randrange(len(pairs))]
        image, target, mask = _crop_with_mask(pair, crop_size, rng)
        images.append(image)
        targets.append(target)
        masks.append(mask)
    image_batch = torch.stack(images)
    target_batch = torch.stack(targets)
    mask_batch = torch.stack(masks)
    generator = torch.Generator().manual_seed(step_seed + 100_003)
    image_batch = _photometric_augmentation(image_batch, generator, augmentation_strength)
    return image_batch, target_batch, mask_batch


@torch.no_grad()
def _evaluate(
    model: AlbedoReadout,
    reservoir: ToyImageReservoir,
    pairs: list[RealPair],
) -> dict[str, float]:
    model.eval()
    per_image = []
    for pair in pairs:
        features = _frozen_features(reservoir, pair.image)
        prediction = model(features, pair.albedo.shape[-2:])
        per_image.append(
            albedo_metrics(
                prediction,
                pair.albedo.unsqueeze(0),
                pair.mask.unsqueeze(0),
            )
        )
    names = per_image[0].keys()
    return {name: sum(item[name] for item in per_image) / len(per_image) for name in names}


@torch.no_grad()
def _identity_baseline(pairs: list[RealPair]) -> dict[str, float]:
    per_image = [
        albedo_metrics(
            pair.image.unsqueeze(0),
            pair.albedo.unsqueeze(0),
            pair.mask.unsqueeze(0),
        )
        for pair in pairs
    ]
    names = per_image[0].keys()
    return {name: sum(item[name] for item in per_image) / len(per_image) for name in names}


def _score(metrics: dict[str, float]) -> float:
    return metrics["mae"] + 0.025 * (1.0 - metrics["ssim_7x7"])


def _ema_update(ema: dict[str, torch.Tensor], model: AlbedoReadout, decay: float) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                ema[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema[name].copy_(value)


def _train_candidate(
    config: dict,
    reservoir: ToyImageReservoir,
    train_pairs: list[RealPair],
    validation_pairs: list[RealPair],
    *,
    steps: int,
    batch_size: int,
    crop_sizes: tuple[int, ...],
    learning_rate: float,
    seed: int,
    eval_every: int,
    augmentation_strength: float,
    initial_state: dict[str, torch.Tensor] | None = None,
) -> tuple[AlbedoReadout, dict]:
    torch.manual_seed(seed)
    trajectory_channels = reservoir.channels * reservoir.steps
    in_channels = trajectory_channels + AUXILIARY_ALBEDO_CHANNELS
    model = AlbedoReadout(
        in_channels=in_channels,
        width=config["width"],
        depth=config["depth"],
        architecture=config["architecture"],
        trajectory_channels=trajectory_channels,
        auxiliary_channels=AUXILIARY_ALBEDO_CHANNELS,
    )
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(steps, 1),
        eta_min=learning_rate * 0.05,
    )
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = _evaluate(model, reservoir, validation_pairs)
    best_score = _score(best_metrics)
    best_step = 0
    history = []

    for step in range(1, steps + 1):
        source, target, mask = _training_batch(
            train_pairs,
            batch_size,
            crop_sizes,
            seed * 1_000_003 + step,
            augmentation_strength,
        )
        features = _frozen_features(reservoir, source)
        model.train()
        prediction = model(features, target.shape[-2:])
        loss, parts = masked_albedo_loss(prediction, target, mask)
        smooth = _shading_smoothness(prediction, source, target, mask)
        total = loss + 0.025 * smooth
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        _ema_update(ema, model, decay=0.995)

        if step % eval_every == 0 or step == steps:
            raw_state = copy.deepcopy(model.state_dict())
            raw_metrics = _evaluate(model, reservoir, validation_pairs)
            model.load_state_dict(ema)
            ema_metrics = _evaluate(model, reservoir, validation_pairs)
            if _score(ema_metrics) <= _score(raw_metrics):
                candidate_metrics = ema_metrics
                candidate_state = copy.deepcopy(ema)
                selected = "ema"
            else:
                candidate_metrics = raw_metrics
                candidate_state = raw_state
                selected = "raw"
            model.load_state_dict(raw_state)
            row = {
                "step": step,
                "loss": float(total.detach()),
                "smooth": float(smooth.detach()),
                "selected": selected,
                "metrics": candidate_metrics,
                "parts": parts,
            }
            history.append(row)
            if _score(candidate_metrics) < best_score:
                best_score = _score(candidate_metrics)
                best_metrics = candidate_metrics
                best_state = candidate_state
                best_step = step

    model.load_state_dict(best_state)
    return model, {
        "config": config,
        "steps": steps,
        "best_step": best_step,
        "best_metrics": best_metrics,
        "score": best_score,
        "history": history,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


@torch.no_grad()
def _write_sheet(
    path: Path,
    model: AlbedoReadout,
    reservoir: ToyImageReservoir,
    pairs: list[RealPair],
) -> None:
    rows = []
    for pair in pairs:
        prediction = model(_frozen_features(reservoir, pair.image), pair.albedo.shape[-2:])[0]
        error = (prediction - pair.albedo).abs().mul(4).clamp(0, 1)
        identity_error = (pair.image - pair.albedo).abs().mul(4).clamp(0, 1)
        rows.append((pair.image, pair.albedo, identity_error, prediction, error, pair.mask.expand(3, -1, -1)))

    width = 240
    row_height = 160
    header = 32
    canvas = Image.new("RGB", (width * 6, header + row_height * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    headers = ("Real RGB", "Olbedo target", "RGB error x4", "VOIR v3", "VOIR error x4", "Valid mask")
    for column, label in enumerate(headers):
        draw.text((column * width + 6, 9), label, fill="black")
    for row_index, tensors in enumerate(rows):
        for column, tensor in enumerate(tensors):
            array = tensor.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
            image = Image.fromarray(array).resize((width, row_height), Image.Resampling.BICUBIC)
            canvas.paste(image, (column * width, header + row_index * row_height))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/hf_olbedo_real30")
    parser.add_argument("--output-dir", default="outputs/real_olbedo_cpu_v3")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(1)
    started = time.time()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        max_side = 256
        candidate_steps = 100
        refine_steps = 220
        batch_size = 3
        crop_sizes = (96, 128)
        eval_every = 50
        candidates = [
            {"architecture": "intrinsic_v3", "width": 24, "depth": 6},
            {"architecture": "intrinsic_v3", "width": 32, "depth": 8},
        ]
    else:
        max_side = 384
        candidate_steps = 450
        refine_steps = 1400
        batch_size = 4
        crop_sizes = (128, 160, 192, 224)
        eval_every = 75
        candidates = [
            {"architecture": "intrinsic_v3", "width": 24, "depth": 6},
            {"architecture": "intrinsic_v3", "width": 32, "depth": 8},
            {"architecture": "intrinsic_v3", "width": 40, "depth": 8},
            {"architecture": "dilated_v2", "width": 32, "depth": 7},
        ]

    train_pairs = _load_pairs(dataset_dir / "train_pairs.jsonl", max_side)
    validation_pairs = _load_pairs(dataset_dir / "validation_pairs.jsonl", max_side)
    reservoir = ToyImageReservoir(channels=16, steps=4, seed=1234)
    assert all(not parameter.requires_grad for parameter in reservoir.parameters())
    identity = _identity_baseline(validation_pairs)

    trials = []
    trained_models = []
    for index, config in enumerate(candidates):
        model, result = _train_candidate(
            config,
            reservoir,
            train_pairs,
            validation_pairs,
            steps=candidate_steps,
            batch_size=batch_size,
            crop_sizes=crop_sizes,
            learning_rate=8e-4,
            seed=100 + index,
            eval_every=eval_every,
            augmentation_strength=0.75,
        )
        trials.append(result)
        trained_models.append(model)

    winner_index = min(range(len(trials)), key=lambda index: trials[index]["score"])
    winner = trained_models[winner_index]
    winner_config = candidates[winner_index]
    refined, refinement = _train_candidate(
        winner_config,
        reservoir,
        train_pairs,
        validation_pairs,
        steps=refine_steps,
        batch_size=batch_size,
        crop_sizes=crop_sizes,
        learning_rate=2.5e-4,
        seed=9001,
        eval_every=eval_every,
        augmentation_strength=0.45,
        initial_state=copy.deepcopy(winner.state_dict()),
    )
    final_metrics = _evaluate(refined, reservoir, validation_pairs)

    checkpoint = refined.checkpoint()
    checkpoint.update(
        {
            "dataset": "GDAOSU/Olbedo real30",
            "identity_metrics": identity,
            "validation_metrics": final_metrics,
            "selection_trials": trials,
            "refinement": refinement,
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
    torch.save(checkpoint, output_dir / "voir_real_olbedo_v3.pt")
    _write_sheet(output_dir / "comparison.png", refined, reservoir, validation_pairs)
    report = {
        "dataset": "GDAOSU/Olbedo",
        "real_train_pairs": len(train_pairs),
        "real_validation_pairs": len(validation_pairs),
        "max_side": max_side,
        "crop_sizes": list(crop_sizes),
        "identity_baseline": identity,
        "trials": trials,
        "winner_index": winner_index,
        "winner_config": winner_config,
        "refinement": refinement,
        "final_metrics": final_metrics,
        "improvement_over_rgb": {
            "mae_percent": 100.0 * (identity["mae"] - final_metrics["mae"]) / max(identity["mae"], 1e-9),
            "mse_percent": 100.0 * (identity["mse"] - final_metrics["mse"]) / max(identity["mse"], 1e-9),
            "psnr_db": final_metrics["psnr"] - identity["psnr"],
            "ssim_7x7": final_metrics["ssim_7x7"] - identity["ssim_7x7"],
        },
        "trainable_parameters": sum(parameter.numel() for parameter in refined.parameters()),
        "reservoir_parameters_trained": 0,
        "seconds": time.time() - started,
        "checkpoint": str(output_dir / "voir_real_olbedo_v3.pt"),
        "comparison": str(output_dir / "comparison.png"),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
