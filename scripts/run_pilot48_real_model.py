"""Capture the 48-pair pilot with frozen Mage and train the real albedo decoder.

The pipeline is designed for one L4 Colab session:
- restore completed cache/model files from Google Drive;
- load Mage once and resume per-image caches;
- mirror every four newly completed caches to Drive;
- train a held-out multi-domain decoder;
- compare the decoder to source-RGB and raw Mage-preview baselines;
- upload the cache dataset and model artifacts to Hugging Face.
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
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image


def _run(
    command: list[Any],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
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


def _run_capture(
    command: list[Any],
    *,
    cwd: Path,
    env: dict[str, str],
    cache_dir: Path,
    mirror_cache: Path,
    sync_every: int,
) -> None:
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
    completed_since_sync = 0
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if "    saved " in line:
                completed_since_sync += 1
                if completed_since_sync >= sync_every:
                    _mirror_tree(cache_dir, mirror_cache)
                    completed_since_sync = 0
        code = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        _mirror_tree(cache_dir, mirror_cache)
        raise
    _mirror_tree(cache_dir, mirror_cache)
    if code != 0:
        raise RuntimeError(f"Mage capture failed with exit code {code}")


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
            needs_copy = (
                source_stat.st_size != target_stat.st_size
                or source_stat.st_mtime_ns > target_stat.st_mtime_ns
            )
        if needs_copy:
            shutil.copy2(path, target)
            copied += 1
    print(f"Mirror sync: {source} -> {destination}; files copied={copied}", flush=True)


def _count_states(cache_dir: Path) -> int:
    return len(list((cache_dir / "states").glob("*.pt")))


def _validate_cache(cache_dir: Path, expected: int) -> dict[str, Any]:
    run_path = cache_dir / "run.json"
    manifest_path = cache_dir / "manifest.jsonl"
    if not run_path.exists() or not manifest_path.exists():
        raise RuntimeError(f"cache metadata is missing under {cache_dir}")
    run_info = json.loads(run_path.read_text(encoding="utf-8"))
    rows = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    states = _count_states(cache_dir)
    if run_info.get("status") != "complete" or len(rows) != expected or states != expected:
        raise RuntimeError(
            f"cache validation failed: status={run_info.get('status')} "
            f"manifest={len(rows)} states={states} expected={expected}"
        )
    return run_info


def _tensor_rgb(path: Path, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _tensor_mask(path: Path, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.NEAREST)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def _image_baseline(cache_dir: Path, prediction_key: str) -> dict[str, Any]:
    from voir.metrics import albedo_metrics

    rows = [
        json.loads(line)
        for line in (cache_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    totals: dict[str, float] = {}
    by_domain: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    count = 0
    for row in rows:
        payload = torch.load(cache_dir / row["cache"], map_location="cpu", weights_only=False)
        record = dict(payload.get("dataset_record") or {})
        if record.get("subset", "train") != "validation":
            continue
        state = payload["reservoir_state"]
        size = tuple(int(value) for value in state["output_size"])
        prediction = _tensor_rgb(cache_dir / row[prediction_key], size)
        target = _tensor_rgb(cache_dir / row["albedo"], size)
        mask = _tensor_mask(cache_dir / row["mask"], size)
        metrics = albedo_metrics(prediction, target, mask)
        domain = str(record.get("domain") or "unknown")
        count += 1
        counts[domain] = counts.get(domain, 0) + 1
        bucket = by_domain.setdefault(domain, {})
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value)
            bucket[name] = bucket.get(name, 0.0) + float(value)
    aggregate = {name: value / max(count, 1) for name, value in totals.items()}
    domain_metrics = {
        domain: {name: value / max(counts[domain], 1) for name, value in values.items()}
        for domain, values in by_domain.items()
    }
    return {"images": count, "aggregate": aggregate, "by_domain": domain_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real Mage pilot48 capture and training pipeline.")
    parser.add_argument("--repo-dir", default="/content/voir-pilot")
    parser.add_argument("--mage-dir", default="/content/Mage")
    parser.add_argument("--pilot-dir", default="/content/voir_pilot48")
    parser.add_argument("--work-dir", default="/content/voir_pilot48_real")
    parser.add_argument("--mirror-dir", default="/content/drive/MyDrive/VOIR_pilot48_real")
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--sync-every", type=int, default=4)
    parser.add_argument("--cache-repo", default="ApacheOne/voir-mage-pilot48-cache-v1")
    parser.add_argument("--model-repo", default="ApacheOne/voir-mage-pilot48-albedo-v1")
    parser.add_argument("--skip-cache-upload", action="store_true")
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
    pilot_dir = Path(args.pilot_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    mirror_dir = Path(args.mirror_dir).resolve()
    cache_dir = work_dir / "cache"
    model_dir = work_dir / "model"
    training_dataset_dir = work_dir / "training_dataset"
    mirror_cache = mirror_dir / "cache"
    mirror_model = mirror_dir / "model"

    if not (pilot_dir / "pairs.jsonl").exists():
        raise FileNotFoundError(f"pilot manifest is missing: {pilot_dir / 'pairs.jsonl'}")
    rows = [line for line in (pilot_dir / "pairs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != args.count:
        raise RuntimeError(f"pilot contains {len(rows)} pairs; expected {args.count}")

    work_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    _mirror_tree(mirror_cache, cache_dir)
    _mirror_tree(mirror_model, model_dir)

    environment = os.environ.copy()
    environment["HF_TOKEN"] = token
    environment["HUGGING_FACE_HUB_TOKEN"] = token
    environment["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
    environment["HF_HUB_ETAG_TIMEOUT"] = "120"
    environment["HF_XET_HIGH_PERFORMANCE"] = "1"
    environment["MAGE_SOURCE_DIR"] = str(mage_dir)
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repo_dir), str(mage_dir), environment.get("PYTHONPATH", "")]
    )

    print("=" * 96)
    print("ACTUAL MAGE PILOT48 CAPTURE")
    print("=" * 96)
    print("Existing valid-looking state files:", _count_states(cache_dir))
    capture_command = [
        sys.executable,
        repo_dir / "scripts/cache_mage_real16.py",
        "--dataset-dir",
        pilot_dir,
        "--output-dir",
        cache_dir,
        "--model",
        "microsoft/Mage-Flow-Edit-Turbo",
        "--count",
        args.count,
        "--max-size",
        "512",
        "--layers",
        "0,2,5,8,11",
        "--projection-channels",
        "64",
        "--projection-seed",
        "1337",
        "--seed-base",
        "620000",
        "--prompt",
        "remove illumination, shadows, highlights, and reflections; output diffuse albedo only",
        "--attn-backend",
        "sdpa",
    ]
    _run_capture(
        capture_command,
        cwd=repo_dir,
        env=environment,
        cache_dir=cache_dir,
        mirror_cache=mirror_cache,
        sync_every=max(1, args.sync_every),
    )
    run_info = _validate_cache(cache_dir, args.count)

    source_baseline = _image_baseline(cache_dir, "input")
    mage_preview_baseline = _image_baseline(cache_dir, "preview")
    baseline_report = {
        "source_rgb": source_baseline,
        "mage_preview": mage_preview_baseline,
    }
    (work_dir / "baselines.json").write_text(json.dumps(baseline_report, indent=2), encoding="utf-8")
    print("BASELINES=", json.dumps(baseline_report, indent=2))

    if training_dataset_dir.exists() or training_dataset_dir.is_symlink():
        shutil.rmtree(training_dataset_dir, ignore_errors=True)
    nested = training_dataset_dir / "mage_cache"
    nested.mkdir(parents=True, exist_ok=True)
    os.symlink(cache_dir, nested / "real16", target_is_directory=True)

    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)

    print("=" * 96)
    print("TRAINING REAL MULTI-DOMAIN PILOT48 MODEL")
    print("=" * 96)
    _run(
        [
            sys.executable,
            repo_dir / "scripts/train_hf_real_mage.py",
            "--hf-dataset",
            args.cache_repo,
            "--dataset-dir",
            training_dataset_dir,
            "--output-dir",
            model_dir,
            "--epochs",
            "80",
            "--batch-size",
            "2",
            "--repeats",
            "4",
            "--patch-grid",
            "16,16",
            "--width",
            "80",
            "--depth",
            "7",
            "--lr",
            "0.0005",
            "--weight-decay",
            "0.0002",
            "--ema-decay",
            "0.99",
            "--validation-every",
            "1",
            "--patience",
            "12",
            "--seed",
            "20260727",
            "--num-workers",
            "0",
            "--upload-repo",
            args.model_repo,
        ],
        cwd=repo_dir,
        env=environment,
    )
    _mirror_tree(model_dir, mirror_model)

    summary_path = model_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError("training did not produce summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_metrics = summary["final_validation"]
    source_metrics = source_baseline["aggregate"]
    preview_metrics = mage_preview_baseline["aggregate"]
    report = {
        "status": "PASS" if model_metrics["mae"] < min(source_metrics["mae"], preview_metrics["mae"]) else "TRAINED_BASELINE_NOT_BEATEN",
        "pairs": args.count,
        "train_images": summary["train_images"],
        "validation_images": summary["validation_images"],
        "cache_total_gib": run_info.get("total_cache_bytes", 0) / 2**30,
        "source_rgb_baseline": source_baseline,
        "mage_preview_baseline": mage_preview_baseline,
        "model_validation": model_metrics,
        "mae_gain_over_source": (source_metrics["mae"] - model_metrics["mae"]) / max(source_metrics["mae"], 1e-12),
        "mae_gain_over_mage_preview": (preview_metrics["mae"] - model_metrics["mae"]) / max(preview_metrics["mae"], 1e-12),
        "cache_repo": args.cache_repo,
        "model_repo": args.model_repo,
        "completed_unix": time.time(),
    }
    report_path = model_dir / "pilot48_acceptance.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _mirror_tree(model_dir, mirror_model)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    if not args.skip_cache_upload:
        api.create_repo(args.cache_repo, repo_type="dataset", private=False, exist_ok=True)
        cache_commit = api.upload_folder(
            repo_id=args.cache_repo,
            repo_type="dataset",
            folder_path=cache_dir,
            commit_message="Upload 48-pair multi-domain actual-Mage cache",
        )
        print("HF_CACHE_COMMIT=", cache_commit.oid)
        print("HF_CACHE_URL=https://huggingface.co/datasets/" + args.cache_repo)

    api.create_repo(args.model_repo, repo_type="model", private=False, exist_ok=True)
    model_commit = api.upload_folder(
        repo_id=args.model_repo,
        repo_type="model",
        folder_path=model_dir,
        commit_message="Add pilot48 metrics and acceptance report",
    )
    print("HF_MODEL_COMMIT=", model_commit.oid)
    print("HF_MODEL_URL=https://huggingface.co/" + args.model_repo)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
