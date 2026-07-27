"""Train and evaluate the frozen-reservoir albedo readout entirely on CPU."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from voir.albedo_features import fixed_albedo_features
from voir.losses import albedo_loss
from voir.metrics import albedo_metrics
from voir.readout import AlbedoReadout
from voir.reservoir import ToyImageReservoir
from voir.synthetic import synthetic_albedo_benchmark_batch


@torch.no_grad()
def _capture_batch(
    reservoir: ToyImageReservoir,
    images: torch.Tensor,
    chunk_size: int = 32,
) -> torch.Tensor:
    chunks = []
    leak_rates = torch.linspace(0.35, 0.80, reservoir.steps).tolist()
    for start in range(0, len(images), chunk_size):
        image = images[start:start + chunk_size]
        base = reservoir.input_conv(image)
        state = torch.zeros_like(base)
        trajectory = []
        for leak in leak_rates:
            state = torch.tanh(
                base
                + float(leak) * reservoir.recurrent_conv(state)
                + 0.15 * reservoir.mix_conv(state)
            )
            trajectory.append(state)
        trajectory_flat = torch.cat(trajectory, dim=1)
        auxiliary = fixed_albedo_features(image)
        chunks.append(torch.cat([auxiliary, trajectory_flat], dim=1).cpu())
    return torch.cat(chunks, dim=0)


def _train_phase(
    model: AlbedoReadout,
    features: torch.Tensor,
    target: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    eta_min: float,
) -> float:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(steps, 1),
        eta_min=eta_min,
    )
    last_loss = 0.0
    model.train()
    for _ in range(steps):
        indices = torch.randint(0, len(features), (batch_size,))
        prediction = model(features[indices], target.shape[-2:])
        loss, _ = albedo_loss(prediction, target[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach())
    return last_loss


@torch.no_grad()
def _predict(
    model: AlbedoReadout,
    features: torch.Tensor,
    batch_size: int = 8,
) -> torch.Tensor:
    model.eval()
    return torch.cat(
        [model(features[index:index + batch_size]) for index in range(0, len(features), batch_size)],
        dim=0,
    )


@torch.no_grad()
def _evaluate_seed(
    model: AlbedoReadout,
    reservoir: ToyImageReservoir,
    seed: int,
    samples: int,
) -> dict[str, float]:
    observed, target = synthetic_albedo_benchmark_batch(samples, size=48, seed=seed)
    features = _capture_batch(reservoir, observed)
    prediction = _predict(model, features)
    return albedo_metrics(prediction, target)


def _mean_metrics(per_seed: dict[str, dict[str, float]]) -> dict[str, float]:
    names = next(iter(per_seed.values())).keys()
    return {
        name: sum(metrics[name] for metrics in per_seed.values()) / len(per_seed)
        for name in names
    }


def _online_generalization_phase(
    model: AlbedoReadout,
    reservoir: ToyImageReservoir,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    eta_min: float,
    seed_base: int,
    validation_seeds: tuple[int, ...],
    validation_samples: int,
    eval_every: int,
) -> dict[str, object]:
    """Fine-tune on newly generated samples and keep the best held-out MAE state."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(steps, 1),
        eta_min=eta_min,
    )

    initial = {
        str(seed): _evaluate_seed(model, reservoir, seed, validation_samples)
        for seed in validation_seeds
    }
    best_score = _mean_metrics(initial)["mae"]
    best_step = 0
    best_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    last_loss = 0.0

    for step in range(1, steps + 1):
        observed, target = synthetic_albedo_benchmark_batch(
            batch_size,
            size=48,
            seed=seed_base + step,
        )
        features = _capture_batch(reservoir, observed)
        model.train()
        prediction = model(features)
        loss, _ = albedo_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach())

        if step % eval_every == 0 or step == steps:
            per_seed = {
                str(seed): _evaluate_seed(model, reservoir, seed, validation_samples)
                for seed in validation_seeds
            }
            score = _mean_metrics(per_seed)["mae"]
            if score < best_score:
                best_score = score
                best_step = step
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }

    model.load_state_dict(best_state)
    final = {
        str(seed): _evaluate_seed(model, reservoir, seed, validation_samples)
        for seed in validation_seeds
    }
    return {
        "steps": steps,
        "best_step": best_step,
        "last_loss": last_loss,
        "learning_rate": learning_rate,
        "eta_min": eta_min,
        "seed_base": seed_base,
        "validation": final,
        "validation_average": _mean_metrics(final),
    }


def _write_sheet(
    path: Path,
    observed: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    samples: int = 16,
) -> None:
    samples = min(samples, len(observed))
    height, width = observed.shape[-2:]
    scale = 3
    cell_w, cell_h = width * scale, height * scale
    header = 28
    sheet = Image.new("RGB", (cell_w * 4, header + cell_h * samples), "white")
    draw = ImageDraw.Draw(sheet)
    for column, name in enumerate(("Observed", "Exact albedo", "VOIR v2", "Error x4")):
        draw.text((column * cell_w + 6, 7), name, fill="black")
    for row in range(samples):
        error = (prediction[row] - target[row]).abs().mul(4.0).clamp(0.0, 1.0)
        for column, tensor in enumerate((observed[row], target[row], prediction[row], error)):
            array = tensor.permute(1, 2, 0).clamp(0.0, 1.0).mul(255).byte().numpy()
            image = Image.fromarray(array).resize((cell_w, cell_h), Image.Resampling.NEAREST)
            sheet.paste(image, (column * cell_w, header + row * cell_h))
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/cpu_albedo_benchmark")
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.threads)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if args.quick:
        train_small, train_large = 64, 96
        offline_steps = (30, 30, 30)
        online_steps = (15, 15, 15, 15)
        selection_samples = 12
        final_samples = 24
        eval_every = 5
    else:
        train_small, train_large = 256, 512
        offline_steps = (401, 401, 401)
        online_steps = (240, 240, 240, 240)
        selection_samples = 48
        final_samples = 96
        eval_every = 30

    observed_small, target_small = synthetic_albedo_benchmark_batch(
        train_small, size=48, seed=10
    )
    observed_large, target_large = synthetic_albedo_benchmark_batch(
        train_large, size=48, seed=10
    )

    reservoir = ToyImageReservoir(channels=16, steps=4, seed=1234)
    features_small = _capture_batch(reservoir, observed_small)
    features_large = _capture_batch(reservoir, observed_large)

    torch.manual_seed(42)
    model = AlbedoReadout(
        features_small.shape[1],
        width=32,
        depth=7,
        architecture="dilated_v2",
    )
    phase1_loss = _train_phase(
        model,
        features_small,
        target_small,
        steps=offline_steps[0],
        batch_size=8,
        learning_rate=1.5e-3,
        weight_decay=1e-4,
        eta_min=3e-5,
    )

    torch.manual_seed(31415)
    phase2_loss = _train_phase(
        model,
        features_large,
        target_large,
        steps=offline_steps[1],
        batch_size=8,
        learning_rate=3e-4,
        weight_decay=5e-5,
        eta_min=5e-5,
    )

    torch.manual_seed(27182)
    phase3_loss = _train_phase(
        model,
        features_large,
        target_large,
        steps=offline_steps[2],
        batch_size=8,
        learning_rate=1.2e-4,
        weight_decay=2e-5,
        eta_min=1e-5,
    )

    validation_seeds = (999, 2027)
    online_configs = (
        (7e-5, 5e-6, 100000),
        (3e-5, 2e-6, 200000),
        (1.5e-5, 1e-6, 300000),
        (8e-6, 5e-7, 400000),
    )
    online_history = []
    for steps, (learning_rate, eta_min, seed_base) in zip(online_steps, online_configs):
        online_history.append(
            _online_generalization_phase(
                model,
                reservoir,
                steps=steps,
                batch_size=8,
                learning_rate=learning_rate,
                eta_min=eta_min,
                seed_base=seed_base,
                validation_seeds=validation_seeds,
                validation_samples=selection_samples,
                eval_every=eval_every,
            )
        )

    evaluation_seeds = (999, 2027, 4040)
    per_seed = {
        str(seed): _evaluate_seed(model, reservoir, seed, final_samples)
        for seed in evaluation_seeds
    }
    average_metrics = _mean_metrics(per_seed)

    observed_sheet, target_sheet = synthetic_albedo_benchmark_batch(
        final_samples, size=48, seed=4040
    )
    features_sheet = _capture_batch(reservoir, observed_sheet)
    prediction_sheet = _predict(model, features_sheet)

    payload = model.checkpoint()
    payload.update(
        {
            "metrics": {
                "per_seed": per_seed,
                "three_seed_average": average_metrics,
            },
            "benchmark": {
                "train_samples_phase1": train_small,
                "train_samples_phase2_3": train_large,
                "selection_samples_per_seed": selection_samples,
                "final_samples_per_seed": final_samples,
                "image_size": 48,
                "offline_steps": list(offline_steps),
                "online_steps": list(online_steps),
                "frozen_reservoir": True,
                "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
            },
            "online_history": online_history,
        }
    )
    checkpoint_path = output_dir / "albedo_readout_v2_generalized.pt"
    comparison_path = output_dir / "comparison_seed4040.png"
    torch.save(payload, checkpoint_path)
    _write_sheet(
        comparison_path,
        observed_sheet,
        target_sheet,
        prediction_sheet,
    )

    report = {
        **payload["benchmark"],
        "readout_channels": int(features_large.shape[1]),
        "trajectory_channels": 64,
        "auxiliary_channels": 63,
        "offline_phase_losses": [phase1_loss, phase2_loss, phase3_loss],
        "online_history": online_history,
        "metrics": payload["metrics"],
        "seconds": time.time() - started,
        "checkpoint": str(checkpoint_path),
        "comparison": str(comparison_path),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
