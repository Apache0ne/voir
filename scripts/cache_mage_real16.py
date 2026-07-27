"""Cache actual Mage-Flow-Edit-Turbo trajectories for sixteen paired real images.

The output is intentionally GitHub-friendly: one portable mixed-precision ``.pt``
file per image, plus PNG inputs/targets/masks/previews and JSON manifests. No raw
full-width transformer activations are stored; selected layers are reduced by the
fixed deterministic hash projection configured in ``CaptureConfig``.

Projected transformer states stay float32 because their valid dynamic range can
exceed float16 even when every source value is finite. Large sampler/conditioning
tensors remain float16.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from voir.reservoir import CaptureConfig, MageEditReservoir
from voir.state import ReservoirState


FORMAT = "voir_mage_real16_cache_v1"
DEFAULT_PROMPT = "remove illumination, shadows, highlights, and reflections; output diffuse albedo only"


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def _state_payload(state: ReservoirState) -> dict[str, Any]:
    state.validate()
    # Do not downcast projected hidden states to float16. The fixed hash projection
    # can produce finite values outside float16's +/-65504 range. A float16 cast
    # would silently create infinities and corrupt an otherwise valid capture.
    features = state.features.detach().cpu().float().contiguous()
    if not torch.isfinite(features).all():
        raise ValueError("projected hidden states are non-finite before serialization")
    return {
        "features": features,
        "output_size": tuple(int(value) for value in state.output_size),
        "sigmas": state.sigmas.detach().cpu().float().contiguous(),
        "layer_indices": tuple(int(value) for value in state.layer_indices),
        "source": state.source,
        "metadata": {
            **state.metadata,
            "feature_storage_dtype": "float32",
            "feature_max_abs": float(features.abs().max()),
        },
        "aux_features": None
        if state.aux_features is None
        else state.aux_features.detach().cpu().to(torch.float16).contiguous(),
    }


def _validate_cache(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError(f"unexpected cache format in {path}")
    state = payload["reservoir_state"]
    sampler = payload["sampler_cache"]
    features = state["features"]
    evaluations = sampler["model_eval_sigmas"].numel()
    if features.ndim != 5 or features.shape[0] != evaluations:
        raise ValueError("hidden-state and sampler evaluation counts differ")
    for key in ("eval_latents", "eval_denoised", "eval_velocity"):
        tensor = sampler[key]
        if tensor.shape[0] != evaluations or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid {key}")
    if not torch.isfinite(features).all():
        raise ValueError("projected hidden states contain non-finite values")
    return payload


def _gpu_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "compute_capability": f"{props.major}.{props.minor}",
        "torch_cuda": torch.version.cuda,
    }


def _copy_asset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/hf_olbedo_mage16")
    parser.add_argument("--output-dir", default="mage_cache/real16")
    parser.add_argument("--model", default="microsoft/Mage-Flow-Edit-Turbo")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--layers", default="0,2,5,8,11")
    parser.add_argument("--projection-channels", type=int, default=64)
    parser.add_argument("--projection-seed", type=int, default=1337)
    parser.add_argument("--seed-base", type=int, default=42000)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--attn-backend", choices=("sdpa", "flash2", "flash4"), default="sdpa")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-file-mib", type=float, default=95.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Actual Mage cache capture requires a CUDA GPU")
    if args.count < 1:
        raise ValueError("count must be positive")
    layers = tuple(int(item.strip()) for item in args.layers.split(",") if item.strip())
    if not layers:
        raise ValueError("at least one transformer layer is required")

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = dataset_dir / "pairs.jsonl"
    records = _read_jsonl(manifest_path)
    if len(records) < args.count:
        raise RuntimeError(f"requested {args.count} images but manifest contains {len(records)}")
    records = records[: args.count]

    for subdir in ("states", "inputs", "albedo", "masks", "previews"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    try:
        from mage_flow.models.modules._attn_backend import set_attn_backend
    except ImportError as exc:
        raise RuntimeError("Install the pinned Microsoft Mage source before running this script") from exc
    set_attn_backend(args.attn_backend)

    config = CaptureConfig(
        layers=layers,
        projection_channels=args.projection_channels,
        projection_seed=args.projection_seed,
        steps=4,
        alpha=0.60,
        beta=0.80,
        start=1.0,
        end=0.0,
        shift=6.0,
        train_timesteps=1000,
        eta=1.0,
        s_noise=1.0,
        r=0.5,
        prefer_torchsde=True,
    )

    started = time.time()
    print(f"Loading {args.model} on CUDA...")
    reservoir = MageEditReservoir.from_pretrained(
        args.model,
        device="cuda",
        config=config,
        attn_backend=args.attn_backend,
    )
    set_attn_backend(args.attn_backend)
    reservoir.pipeline.model.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in reservoir.pipeline.model.parameters()):
        raise RuntimeError("Mage must remain completely frozen during cache capture")

    run_info = {
        "format": FORMAT,
        "status": "running",
        "started_unix": started,
        "model": args.model,
        "prompt": args.prompt,
        "count": args.count,
        "max_size": args.max_size,
        "layers": list(layers),
        "projection_channels": args.projection_channels,
        "projection_seed": args.projection_seed,
        "seed_base": args.seed_base,
        "attention_backend": args.attn_backend,
        "feature_storage_dtype": "float32",
        "sampler": {
            "name": "dpmpp_sde_gpu",
            "steps": 4,
            "beta_alpha": 0.60,
            "beta_beta": 0.80,
            "sigma_start": 1.0,
            "sigma_end": 0.0,
            "shift": 6.0,
            "eta": 1.0,
            "s_noise": 1.0,
            "r": 0.5,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "gpu": _gpu_info(),
            "voir_commit": _git_rev(Path(__file__).resolve().parents[1]),
            "mage_commit": _git_rev(Path(os.environ.get("MAGE_SOURCE_DIR", "/content/Mage"))),
        },
        "items": [],
    }
    _atomic_json(output_dir / "run.json", run_info)

    completed_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        stem = f"{index:03d}"
        state_path = output_dir / "states" / f"{stem}.pt"
        input_source = dataset_dir / record["image"]
        albedo_source = dataset_dir / record["albedo"]
        mask_source = dataset_dir / record["mask"]
        input_path = output_dir / "inputs" / f"{stem}.png"
        albedo_path = output_dir / "albedo" / f"{stem}.png"
        mask_path = output_dir / "masks" / f"{stem}.png"
        preview_path = output_dir / "previews" / f"{stem}.png"
        seed = args.seed_base + index

        if state_path.exists() and not args.overwrite:
            try:
                existing = _validate_cache(state_path)
                row = dict(existing["manifest_record"])
                row["resumed"] = True
                completed_rows.append(row)
                _atomic_jsonl(output_dir / "manifest.jsonl", completed_rows)
                print(f"[{index + 1:02d}/{args.count}] valid cache exists: {state_path.name}")
                continue
            except Exception as exc:
                print(f"Existing cache is invalid and will be replaced: {exc}")

        _copy_asset(input_source, input_path)
        _copy_asset(albedo_source, albedo_path)
        _copy_asset(mask_source, mask_path)
        torch.cuda.reset_peak_memory_stats()
        item_started = time.time()
        print(f"[{index + 1:02d}/{args.count}] Mage capture seed={seed} source_row={record.get('source_row')}")

        state, preview, sampler_cache = reservoir.capture_detailed(
            input_source,
            prompt=args.prompt,
            seed=seed,
            max_size=args.max_size,
        )
        preview.save(preview_path, optimize=True)

        feature_max_abs = float(state.features.abs().max())
        row = {
            "index": index,
            "source_row": record.get("source_row"),
            "scene": record.get("scene"),
            "date": record.get("date"),
            "lighting": record.get("lighting"),
            "seed": seed,
            "input": str(input_path.relative_to(output_dir)),
            "albedo": str(albedo_path.relative_to(output_dir)),
            "mask": str(mask_path.relative_to(output_dir)),
            "preview": str(preview_path.relative_to(output_dir)),
            "cache": str(state_path.relative_to(output_dir)),
            "state_shape": list(state.features.shape),
            "state_dtype": "float32",
            "state_max_abs": feature_max_abs,
            "auxiliary_shape": None if state.aux_features is None else list(state.aux_features.shape),
            "output_size": list(state.output_size),
            "target_grid": list(sampler_cache.target_grid),
            "target_token_shape": list(sampler_cache.initial_target_tokens.shape),
            "reference_token_shape": list(sampler_cache.reference_tokens.shape),
            "text_token_shape": list(sampler_cache.text_tokens.shape),
            "model_evaluations": int(sampler_cache.model_eval_sigmas.numel()),
            "schedule_sigmas": sampler_cache.schedule_sigmas.tolist(),
            "model_eval_sigmas": sampler_cache.model_eval_sigmas.tolist(),
            "seconds": time.time() - item_started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "input_sha256": _sha256(input_path),
            "albedo_sha256": _sha256(albedo_path),
            "mask_sha256": _sha256(mask_path),
            "preview_sha256": _sha256(preview_path),
        }
        payload = {
            "format": FORMAT,
            "reservoir_state": _state_payload(state),
            "sampler_cache": sampler_cache.to_payload(),
            "dataset_record": record,
            "manifest_record": row,
        }
        temporary = state_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(state_path)
        row["cache_bytes"] = int(state_path.stat().st_size)
        row["cache_sha256"] = _sha256(state_path)
        maximum_bytes = int(args.max_file_mib * 1024 * 1024)
        if row["cache_bytes"] > maximum_bytes:
            state_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"{stem}.pt is {row['cache_bytes'] / 2**20:.1f} MiB, above the configured "
                f"GitHub-safe limit of {args.max_file_mib:.1f} MiB"
            )

        # Reload from disk so a successful row proves the serialized cache itself is valid.
        _validate_cache(state_path)

        completed_rows.append(row)
        run_info["items"] = completed_rows
        _atomic_jsonl(output_dir / "manifest.jsonl", completed_rows)
        _atomic_json(output_dir / "run.json", run_info)
        print(
            f"    saved {state_path.name}: {row['cache_bytes'] / 2**20:.2f} MiB, "
            f"state={row['state_shape']}, max_abs={feature_max_abs:.3f}, "
            f"evals={row['model_evaluations']}"
        )
        del state, sampler_cache, payload
        torch.cuda.empty_cache()

    run_info["status"] = "complete"
    run_info["completed_unix"] = time.time()
    run_info["total_seconds"] = run_info["completed_unix"] - started
    run_info["items"] = completed_rows
    run_info["total_cache_bytes"] = sum(int(row["cache_bytes"]) for row in completed_rows)
    run_info["manifest_sha256"] = _sha256(output_dir / "manifest.jsonl")
    _atomic_json(output_dir / "run.json", run_info)
    print(json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()
