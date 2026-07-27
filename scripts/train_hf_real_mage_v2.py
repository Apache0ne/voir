from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from voir.losses import masked_albedo_loss
from voir.metrics import albedo_metrics
from voir.real_mage import RealMageCacheDataset, compute_branch_normalization, cosine_warmup
from voir.real_mage_v2 import RealMageAlbedoNetV2


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_cache_root(dataset_dir: Path) -> Path:
    candidates = [dataset_dir, dataset_dir / "mage_cache" / "real16", dataset_dir / "real16"]
    for candidate in candidates:
        if (candidate / "run.json").exists() and (candidate / "manifest.jsonl").exists():
            return candidate
    raise FileNotFoundError(f"could not locate a Mage cache under {dataset_dir}")


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device, non_blocking=True)
            if value.ndim == 4 and name != "index":
                value = value.contiguous(memory_format=torch.channels_last)
            moved[name] = value
        else:
            moved[name] = value
    return moved


def _rotate_batch(batch: dict[str, torch.Tensor], enabled: bool) -> dict[str, torch.Tensor]:
    if not enabled:
        return batch
    k = int(torch.randint(0, 4, (1,)).item())
    if k == 0:
        return batch
    for name in ("hidden", "sampler", "auxiliary", "source", "target", "mask"):
        value = batch[name]
        if value.shape[-2] != value.shape[-1]:
            raise ValueError("90-degree augmentation requires square training crops")
        batch[name] = torch.rot90(value, k=k, dims=(-2, -1)).contiguous()
    return batch


def _model_call(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(batch["hidden"], batch["sampler"], batch["auxiliary"], batch["source"])


@torch.no_grad()
def _update_ema(ema: RealMageAlbedoNetV2, model: RealMageAlbedoNetV2, decay: float) -> None:
    model_parameters = dict(model.named_parameters())
    for name, parameter in ema.named_parameters():
        parameter.mul_(decay).add_(model_parameters[name].detach(), alpha=1.0 - decay)
    model_buffers = dict(model.named_buffers())
    for name, value in ema.named_buffers():
        value.copy_(model_buffers[name].detach())


@torch.no_grad()
def _evaluate(
    model: RealMageAlbedoNetV2,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, float], list[dict[str, torch.Tensor]]]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    outputs: list[dict[str, torch.Tensor]] = []
    for batch in loader:
        batch = _move(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            prediction = _model_call(model, batch)
            loss, parts = masked_albedo_loss(prediction, batch["target"], batch["mask"])
        metrics = albedo_metrics(prediction.float(), batch["target"].float(), batch["mask"].float())
        row = {"loss": float(loss), **parts, **metrics}
        batch_size = int(prediction.shape[0])
        count += batch_size
        for name, value in row.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch_size
        for item in range(batch_size):
            outputs.append(
                {
                    "index": batch["index"][item].detach().cpu(),
                    "source": batch["source"][item].detach().float().cpu(),
                    "target": batch["target"][item].detach().float().cpu(),
                    "mask": batch["mask"][item].detach().float().cpu(),
                    "prediction": prediction[item].detach().float().cpu(),
                }
            )
    return {name: value / max(count, 1) for name, value in totals.items()}, outputs


def _to_image(value: torch.Tensor) -> Image.Image:
    array = (
        value.detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _save_comparisons(outputs: list[dict[str, torch.Tensor]], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in outputs:
        source = _to_image(item["source"])
        target = _to_image(item["target"])
        prediction = _to_image(item["prediction"])
        canvas = Image.new("RGB", (source.width * 3, source.height))
        canvas.paste(source, (0, 0))
        canvas.paste(target, (source.width, 0))
        canvas.paste(prediction, (source.width * 2, 0))
        index = int(item["index"].item())
        canvas.save(destination / f"validation_{index:03d}_source_target_prediction.png", optimize=True)


def _write_model_card(output_dir: Path, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    card = f"""---
license: mit
pipeline_tag: image-to-image
tags:
- albedo
- mage-flow
- reservoir-computing
- intrinsic-images
- pytorch
---

# VOIR Real Mage Albedo v2

Multi-scale source-fusion albedo decoder using frozen internal trajectories from
`microsoft/Mage-Flow-Edit-Turbo`.

- Cache dataset: `{args.hf_dataset}`
- Train/validation: {summary['train_images']} / {summary['validation_images']}
- Trainable parameters: {summary['trainable_parameters']:,}
- Epochs completed: {summary['epochs_completed']}
- Warm start: `{args.init_checkpoint or 'none'}`

## Best held-out validation

```json
{json.dumps(summary['best'], indent=2)}
```

The validation set is small. Treat these metrics as pilot evidence, not proof of
universal generalization.
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multi-scale Real Mage albedo v2 decoder.")
    parser.add_argument("--hf-dataset", default="ApacheOne/voir-mage-pilot48-cache-v1")
    parser.add_argument("--dataset-dir", default="/content/voir_pilot48_real/cache")
    parser.add_argument("--output-dir", default="/content/voir_pilot48_v2")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--patch-grid", default="24,24")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=2.0e-4)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--validation-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=2.0e-5)
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--rotate90", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--upload-repo", default=None)
    args = parser.parse_args()

    _seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        cache_root = _resolve_cache_root(dataset_dir)
    except FileNotFoundError:
        from huggingface_hub import snapshot_download

        print(f"Downloading public cache dataset {args.hf_dataset}...")
        snapshot_download(repo_id=args.hf_dataset, repo_type="dataset", local_dir=dataset_dir)
        cache_root = _resolve_cache_root(dataset_dir)
    print("Cache root:", cache_root)

    patch_grid = tuple(int(value.strip()) for value in args.patch_grid.split(","))
    if len(patch_grid) != 2:
        raise ValueError("--patch-grid must be HEIGHT,WIDTH")
    if args.rotate90 and patch_grid[0] != patch_grid[1]:
        raise ValueError("--rotate90 requires a square --patch-grid")

    train_dataset = RealMageCacheDataset(
        cache_root,
        split="train",
        patch_grid=patch_grid,
        augment=True,
        repeats=args.repeats,
    )
    validation_dataset = RealMageCacheDataset(
        cache_root,
        split="validation",
        patch_grid=None,
        augment=False,
        repeats=1,
    )
    print(
        json.dumps(
            {
                "train_images": train_dataset.base_length,
                "train_samples_per_epoch": len(train_dataset),
                "validation_images": validation_dataset.base_length,
                "channels": train_dataset.channels,
                "full_output_size": list(validation_dataset.samples[0]["output_size"]),
                "grid": list(validation_dataset.samples[0]["grid"]),
            },
            indent=2,
        )
    )

    statistics = compute_branch_normalization(train_dataset)
    grid_h, grid_w = validation_dataset.samples[0]["grid"]
    out_h, out_w = validation_dataset.samples[0]["output_size"]
    scale_h = out_h / grid_h
    scale_w = out_w / grid_w
    if scale_h != 16 or scale_w != 16:
        raise ValueError(f"v2 expects a 16x cache-to-output scale, got {scale_h} x {scale_w}")

    init_payload: dict[str, Any] | None = None
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        init_payload = torch.load(init_path, map_location="cpu", weights_only=False)
        model = RealMageAlbedoNetV2.from_checkpoint(init_payload)
        expected_channels = train_dataset.channels
        config = model.config()
        if config["hidden_channels"] != expected_channels["hidden"]:
            raise ValueError("warm-start hidden-channel mismatch")
        if config["sampler_channels"] != expected_channels["sampler"]:
            raise ValueError("warm-start sampler-channel mismatch")
        if config["auxiliary_channels"] != expected_channels["auxiliary"]:
            raise ValueError("warm-start auxiliary-channel mismatch")
        print("Warm start:", init_path)
        print("Warm-start config:", json.dumps(config, indent=2))
    else:
        model = RealMageAlbedoNetV2(
            hidden_channels=train_dataset.channels["hidden"],
            sampler_channels=train_dataset.channels["sampler"],
            auxiliary_channels=train_dataset.channels["auxiliary"],
            width=args.width,
            depth=args.depth,
            upsample_stages=4,
        )
    model.set_normalization(statistics)
    model = model.to(device).to(memory_format=torch.channels_last)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print("Trainable parameters:", f"{trainable_parameters:,}")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch:", torch.__version__)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = max(1, int(total_steps * max(0.0, args.warmup_fraction)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_warmup(step, total_steps, warmup_steps),
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    compiled_model: torch.nn.Module = model
    if args.compile:
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile: enabled")
        except Exception as exc:
            print("torch.compile unavailable; continuing eagerly:", exc)
            compiled_model = model

    history: list[dict[str, Any]] = []
    initial_validation, initial_outputs = _evaluate(ema_model, validation_loader, device, amp_dtype)
    best: dict[str, Any] = {
        "epoch": 0,
        "train_loss": None,
        "lr": 0.0,
        "seconds": 0.0,
        "validation": initial_validation,
    }
    best_state = {name: value.detach().cpu().clone() for name, value in ema_model.state_dict().items()}
    _save_comparisons(initial_outputs, output_dir / "comparisons_initial")
    print("INITIAL_VALIDATION=", json.dumps(initial_validation))

    stale_validations = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for batch in train_loader:
            batch = _rotate_batch(_move(batch, device), args.rotate90)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                prediction = _model_call(compiled_model, batch)
                loss, _ = masked_albedo_loss(prediction, batch["target"], batch["mask"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            _update_ema(ema_model, model, args.ema_decay)
            running += float(loss.detach())
            batches += 1

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": running / max(batches, 1),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - started,
        }
        should_validate = epoch == 1 or epoch == args.epochs or epoch % args.validation_every == 0
        if should_validate:
            validation, outputs = _evaluate(ema_model, validation_loader, device, amp_dtype)
            row["validation"] = validation
            improved = validation["mae"] < best["validation"]["mae"] - args.min_delta
            if improved:
                best = dict(row)
                best_state = {name: value.detach().cpu().clone() for name, value in ema_model.state_dict().items()}
                stale_validations = 0
                _save_comparisons(outputs, output_dir / "comparisons")
            else:
                stale_validations += 1
        history.append(row)
        print(json.dumps(row))
        (output_dir / "history.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in history),
            encoding="utf-8",
        )
        if should_validate and stale_validations >= args.patience:
            print(f"Early stopping after {stale_validations} validation checks without improvement")
            break

    ema_model.load_state_dict(best_state)
    checkpoint = ema_model.checkpoint(
        best=best,
        history=history,
        initial_validation=initial_validation,
        dataset=args.hf_dataset,
        cache_root=str(cache_root),
        seed=args.seed,
        train_images=train_dataset.base_length,
        validation_images=validation_dataset.base_length,
        trainable_parameters=trainable_parameters,
        torch_version=torch.__version__,
        warm_start=args.init_checkpoint,
        training_args=vars(args),
    )
    checkpoint_path = output_dir / "voir_real_mage_albedo_v2.pt"
    torch.save(checkpoint, checkpoint_path)

    final_validation, final_outputs = _evaluate(ema_model, validation_loader, device, amp_dtype)
    _save_comparisons(final_outputs, output_dir / "comparisons")
    summary = {
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "dataset": args.hf_dataset,
        "cache_root": str(cache_root),
        "train_images": train_dataset.base_length,
        "validation_images": validation_dataset.base_length,
        "channels": train_dataset.channels,
        "model_config": ema_model.config(),
        "trainable_parameters": trainable_parameters,
        "epochs_completed": len(history),
        "initial_validation": initial_validation,
        "best": best,
        "final_validation": final_validation,
        "total_seconds": time.time() - started,
        "warm_start": args.init_checkpoint,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_model_card(output_dir, args, summary)
    print(json.dumps(summary, indent=2))

    if args.upload_repo:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError("--upload-repo requires HF_TOKEN")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(args.upload_repo, repo_type="model", private=False, exist_ok=True)
        commit = api.upload_folder(
            repo_id=args.upload_repo,
            repo_type="model",
            folder_path=output_dir,
            commit_message="Train multi-scale real Mage albedo v2 model",
        )
        print("HF_MODEL_REPO=", args.upload_repo)
        print("HF_MODEL_COMMIT=", commit.oid)
        print("HF_MODEL_URL=https://huggingface.co/" + args.upload_repo)


if __name__ == "__main__":
    main()
