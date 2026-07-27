"""Resumable Colab pipeline for a larger multi-domain actual-Mage albedo model.

This script builds paired data, captures frozen Mage trajectories in restart-safe
chunks, mirrors progress to Google Drive, trains a larger decoder, compares it to
the source-RGB baseline, and optionally uploads cache/model artifacts to HF.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


def _run(command: list[Any], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    command = [str(value) for value in command]
    print("$", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"command failed with exit code {code}: {' '.join(command)}")


def _mirror_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        needs_copy = not target.exists()
        if not needs_copy:
            source_stat = path.stat()
            target_stat = target.stat()
            needs_copy = source_stat.st_size != target_stat.st_size or source_stat.st_mtime_ns > target_stat.st_mtime_ns
        if needs_copy:
            shutil.copy2(path, target)
            copied += 1
    print(f"Mirror sync: {source} -> {destination}; files copied={copied}")


def _count_states(cache_dir: Path) -> int:
    return len(list((cache_dir / "states").glob("*.pt")))


def _validate_cache_count(cache_dir: Path, expected: int) -> None:
    run_path = cache_dir / "run.json"
    manifest_path = cache_dir / "manifest.jsonl"
    if not run_path.exists() or not manifest_path.exists():
        raise RuntimeError(f"cache metadata missing under {cache_dir}")
    run_info = json.loads(run_path.read_text(encoding="utf-8"))
    rows = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    states = _count_states(cache_dir)
    if run_info.get("status") != "complete" or len(rows) != expected or states != expected:
        raise RuntimeError(
            f"cache validation failed: status={run_info.get('status')} manifest={len(rows)} states={states} expected={expected}"
        )


def _source_baseline(cache_dir: Path) -> dict[str, float]:
    from voir.metrics import albedo_metrics
    from voir.real_mage import RealMageCacheDataset

    dataset = RealMageCacheDataset(cache_dir, split="validation", patch_grid=None, augment=False, repeats=1)
    totals: dict[str, float] = {}
    count = 0
    for sample in dataset.samples:
        source = sample["source"].unsqueeze(0)
        target = sample["target"].unsqueeze(0)
        mask = sample["mask"].unsqueeze(0)
        metrics = albedo_metrics(source, target, mask)
        count += 1
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value)
    return {name: value / max(count, 1) for name, value in totals.items()}


def _write_general_model_card(
    model_dir: Path,
    *,
    cache_name: str,
    model_repo: str,
    total: int,
    domains: dict[str, int],
    summary: dict[str, Any],
    acceptance: dict[str, Any],
) -> None:
    card = f"""---
license: cc-by-nc-4.0
pipeline_tag: image-to-image
tags:
- albedo
- inverse-rendering
- mage-flow
- reservoir-computing
- intrinsic-images
- pytorch
---

# VOIR General Albedo v2

Full-resolution albedo decoder trained on cached internal trajectories from the
frozen `microsoft/Mage-Flow-Edit-Turbo` model.

## Training data

- Cache: `{cache_name}`
- Total paired Mage caches: {total}
- Olbedo real outdoor/aerial pairs: {domains['olbedo']}
- MatPredict rendered-object pairs: {domains['matpredict']}
- PBR-Rooms rendered-indoor pairs: {domains['pbrrooms']}
- Train images: {summary['train_images']}
- Validation images: {summary['validation_images']}

The PBR-Rooms component is CC BY-NC 4.0, so this mixed-data checkpoint is marked
non-commercial. Remove that source and retrain for a differently licensed model.

## Held-out validation

```json
{json.dumps(summary['final_validation'], indent=2)}
```

## Source-RGB baseline and acceptance

```json
{json.dumps(acceptance, indent=2)}
```

This is a multi-domain research checkpoint, not a guarantee of exact albedo for
all possible images. Single-image intrinsic decomposition is underdetermined, and
performance must be evaluated on each intended deployment domain.

Model repository: `{model_repo}`
"""
    (model_dir / "README.md").write_text(card, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the resumable VOIR general albedo v2 pipeline.")
    parser.add_argument("--repo-dir", default="/content/voir-general")
    parser.add_argument("--mage-dir", default="/content/Mage")
    parser.add_argument("--work-dir", default="/content/voir_general_v2")
    parser.add_argument("--mirror-dir", default="/content/drive/MyDrive/VOIR_general_v2")
    parser.add_argument("--olbedo-count", type=int, default=256)
    parser.add_argument("--matpredict-count", type=int, default=256)
    parser.add_argument("--pbrrooms-count", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--cache-repo", default=None)
    parser.add_argument("--model-repo", default="ApacheOne/voir-general-albedo-v2")
    parser.add_argument("--skip-cache-upload", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_properties(0).total_memory < 20 * 2**30:
        raise RuntimeError("Use an L4/A100-class GPU with at least 20 GiB VRAM")
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required")

    repo_dir = Path(args.repo_dir).resolve()
    mage_dir = Path(args.mage_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    mirror_dir = Path(args.mirror_dir).resolve()
    pairs_dir = work_dir / "pairs"
    cache_dir = work_dir / "cache"
    model_dir = work_dir / "model"
    training_dataset_dir = work_dir / "training_dataset"
    mirror_pairs = mirror_dir / "pairs"
    mirror_cache = mirror_dir / "cache"
    mirror_model = mirror_dir / "model"
    total = int(args.olbedo_count + args.matpredict_count + args.pbrrooms_count)
    if total < 30:
        raise ValueError("general v2 requires at least 30 total pairs")
    if not repo_dir.exists() or not (repo_dir / "scripts").exists():
        raise FileNotFoundError(f"VOIR repository not found: {repo_dir}")
    if not mage_dir.exists():
        raise FileNotFoundError(f"Microsoft Mage source not found: {mage_dir}")

    work_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    _mirror_tree(mirror_pairs, pairs_dir)
    _mirror_tree(mirror_cache, cache_dir)
    _mirror_tree(mirror_model, model_dir)

    environment = os.environ.copy()
    environment["MAGE_SOURCE_DIR"] = str(mage_dir)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repo_dir), str(mage_dir), environment.get("PYTHONPATH", "")]
    )
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["HF_TOKEN"] = token
    environment["HUGGING_FACE_HUB_TOKEN"] = token

    _run(
        [
            sys.executable,
            repo_dir / "scripts/build_general_albedo_v2.py",
            "--output",
            pairs_dir,
            "--size",
            args.image_size,
            "--olbedo-count",
            args.olbedo_count,
            "--matpredict-count",
            args.matpredict_count,
            "--pbrrooms-count",
            args.pbrrooms_count,
            "--validation-fraction",
            "0.15",
            "--seed",
            "20260727",
        ],
        cwd=repo_dir,
        env=environment,
    )
    pair_rows = [line for line in (pairs_dir / "pairs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(pair_rows) != total:
        raise RuntimeError(f"paired dataset has {len(pair_rows)} rows; expected {total}")
    _mirror_tree(pairs_dir, mirror_pairs)

    chunk = max(1, int(args.chunk_size))
    for end in range(chunk, total + chunk, chunk):
        end = min(end, total)
        existing = _count_states(cache_dir)
        if existing >= end:
            print(f"Cache chunk {end}/{total}: already present ({existing} states)")
            continue
        print("=" * 96)
        print(f"CAPTURE CHUNK: states through {end}/{total}")
        print("=" * 96)
        _run(
            [
                sys.executable,
                repo_dir / "scripts/cache_mage_real16.py",
                "--dataset-dir",
                pairs_dir,
                "--output-dir",
                cache_dir,
                "--model",
                "microsoft/Mage-Flow-Edit-Turbo",
                "--count",
                end,
                "--max-size",
                args.image_size,
                "--layers",
                "0,2,5,8,11",
                "--projection-channels",
                "64",
                "--projection-seed",
                "1337",
                "--seed-base",
                "520000",
                "--prompt",
                "remove illumination, shadows, highlights, and reflections; output diffuse albedo only",
                "--attn-backend",
                "sdpa",
            ],
            cwd=repo_dir,
            env=environment,
        )
        _mirror_tree(cache_dir, mirror_cache)
        print(f"Chunk checkpointed to Drive: {end}/{total}")

    _validate_cache_count(cache_dir, total)
    _mirror_tree(cache_dir, mirror_cache)

    cache_name = args.cache_repo or "local/voir-general-albedo-v2-cache"
    if args.cache_repo and not args.skip_cache_upload:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(args.cache_repo, repo_type="dataset", private=False, exist_ok=True)
        commit = api.upload_folder(
            repo_id=args.cache_repo,
            repo_type="dataset",
            folder_path=cache_dir,
            commit_message=f"Upload {total} multi-domain actual-Mage albedo caches",
        )
        print("HF_CACHE_REPO=", args.cache_repo)
        print("HF_CACHE_COMMIT=", commit.oid)
        print("HF_CACHE_URL=https://huggingface.co/datasets/" + args.cache_repo)

    if args.skip_training:
        print(json.dumps({"status": "CACHE_COMPLETE", "pairs": total, "cache_dir": str(cache_dir)}, indent=2))
        return

    # The legacy trainer checks for dataset_dir/mage_cache/real16 before resolving
    # a local cache root. Create a lightweight symlink instead of copying the cache.
    if training_dataset_dir.exists() or training_dataset_dir.is_symlink():
        shutil.rmtree(training_dataset_dir, ignore_errors=True)
    nested = training_dataset_dir / "mage_cache"
    nested.mkdir(parents=True, exist_ok=True)
    os.symlink(cache_dir, nested / "real16", target_is_directory=True)

    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    _run(
        [
            sys.executable,
            repo_dir / "scripts/train_hf_real_mage.py",
            "--hf-dataset",
            cache_name,
            "--dataset-dir",
            training_dataset_dir,
            "--output-dir",
            model_dir,
            "--epochs",
            "60",
            "--batch-size",
            "4",
            "--repeats",
            "2",
            "--patch-grid",
            "24,24",
            "--width",
            "96",
            "--depth",
            "8",
            "--lr",
            "0.00035",
            "--weight-decay",
            "0.0002",
            "--ema-decay",
            "0.99",
            "--validation-every",
            "1",
            "--patience",
            "10",
            "--seed",
            "20260727",
            "--num-workers",
            "0",
        ],
        cwd=repo_dir,
        env=environment,
    )

    summary_path = model_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError("training did not produce summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_baseline = _source_baseline(cache_dir)
    model_metrics = summary["final_validation"]
    relative_mae_gain = (source_baseline["mae"] - model_metrics["mae"]) / max(source_baseline["mae"], 1e-12)
    gate = {
        "mae_below_0_07": model_metrics["mae"] < 0.07,
        "psnr_above_21": model_metrics["psnr"] > 21.0,
        "ssim_above_0_75": model_metrics["ssim_7x7"] > 0.75,
        "beats_source_mae_by_10_percent": relative_mae_gain >= 0.10,
    }
    domains = {
        "olbedo": args.olbedo_count,
        "matpredict": args.matpredict_count,
        "pbrrooms": args.pbrrooms_count,
    }
    final = {
        "status": "PASS" if all(gate.values()) else "TRAINED_NEEDS_MORE_DATA",
        "pairs": total,
        "domains": domains,
        "model_repo": args.model_repo,
        "source_rgb_baseline": source_baseline,
        "model_validation": model_metrics,
        "relative_mae_gain_over_source": relative_mae_gain,
        "acceptance_gate": gate,
        "work_dir": str(work_dir),
        "mirror_dir": str(mirror_dir),
        "completed_unix": time.time(),
    }
    (model_dir / "general_v2_acceptance.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    _write_general_model_card(
        model_dir,
        cache_name=cache_name,
        model_repo=args.model_repo,
        total=total,
        domains=domains,
        summary=summary,
        acceptance=final,
    )
    _mirror_tree(model_dir, mirror_model)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.model_repo, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.model_repo,
        repo_type="model",
        folder_path=model_dir,
        commit_message="Train multi-domain actual-Mage general albedo v2",
    )
    print("HF_MODEL_REPO=", args.model_repo)
    print("HF_MODEL_COMMIT=", commit.oid)
    print("HF_MODEL_URL=https://huggingface.co/" + args.model_repo)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
