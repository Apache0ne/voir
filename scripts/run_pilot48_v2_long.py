"""Run a longer architecture search and fine-tune pass on the existing Pilot48 cache.

No Mage inference is repeated. Two multi-scale v2 candidates are trained from
scratch, the lower held-out-MAE checkpoint is selected, and that checkpoint receives
a low-learning-rate large-crop fine-tune. The fine-tune trainer evaluates epoch zero,
so the final result cannot silently replace the selected candidate with a worse state.
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


def _run(command: list[Any], *, cwd: Path, env: dict[str, str]) -> None:
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


def _summary(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _complete(path: Path) -> bool:
    try:
        summary = _summary(path)
    except Exception:
        return False
    checkpoint = path / "voir_real_mage_albedo_v2.pt"
    return summary.get("status") == "PASS" and checkpoint.exists()


def _mirror(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    print(f"Drive mirror updated: {destination}", flush=True)


def _train_command(
    repo_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    cache_repo: str,
    *,
    epochs: int,
    batch_size: int,
    repeats: int,
    patch_grid: str,
    width: int,
    depth: int,
    lr: float,
    weight_decay: float,
    ema_decay: float,
    patience: int,
    min_delta: float,
    seed: int,
    init_checkpoint: Path | None = None,
) -> list[Any]:
    command: list[Any] = [
        sys.executable,
        repo_dir / "scripts/train_hf_real_mage_v2.py",
        "--hf-dataset",
        cache_repo,
        "--dataset-dir",
        cache_dir,
        "--output-dir",
        output_dir,
        "--epochs",
        epochs,
        "--batch-size",
        batch_size,
        "--repeats",
        repeats,
        "--patch-grid",
        patch_grid,
        "--width",
        width,
        "--depth",
        depth,
        "--lr",
        lr,
        "--weight-decay",
        weight_decay,
        "--ema-decay",
        ema_decay,
        "--validation-every",
        1,
        "--patience",
        patience,
        "--min-delta",
        min_delta,
        "--warmup-fraction",
        0.03 if init_checkpoint is None else 0.01,
        "--seed",
        seed,
        "--num-workers",
        0,
        "--rotate90",
    ]
    if init_checkpoint is not None:
        command.extend(["--init-checkpoint", init_checkpoint])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Long Pilot48 v2 architecture search and fine-tune.")
    parser.add_argument("--repo-dir", default="/content/voir-pilot")
    parser.add_argument("--cache-dir", default="/content/voir_pilot48_real/cache")
    parser.add_argument("--work-dir", default="/content/voir_pilot48_v2_long")
    parser.add_argument("--mirror-dir", default="/content/drive/MyDrive/VOIR_pilot48_v2_long")
    parser.add_argument("--cache-repo", default="ApacheOne/voir-mage-pilot48-cache-v1")
    parser.add_argument("--v1-model-repo", default="ApacheOne/voir-mage-pilot48-albedo-v1")
    parser.add_argument("--model-repo", default="ApacheOne/voir-mage-pilot48-albedo-v2")
    args = parser.parse_args()

    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required")

    repo_dir = Path(args.repo_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    mirror_dir = Path(args.mirror_dir).resolve()
    candidate_a = work_dir / "candidate_a_w96_d10"
    candidate_b = work_dir / "candidate_b_w112_d12"
    finetune_dir = work_dir / "finetune"
    final_dir = work_dir / "final"

    work_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    if not (cache_dir / "run.json").exists():
        drive_cache = Path("/content/drive/MyDrive/VOIR_pilot48_real/cache")
        if (drive_cache / "run.json").exists():
            shutil.copytree(drive_cache, cache_dir, dirs_exist_ok=True)
        else:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=args.cache_repo,
                repo_type="dataset",
                local_dir=cache_dir,
                token=token,
            )
    run_info = json.loads((cache_dir / "run.json").read_text(encoding="utf-8"))
    states = len(list((cache_dir / "states").glob("*.pt")))
    if run_info.get("status") != "complete" or states != 48:
        raise RuntimeError(f"Pilot48 cache is incomplete: status={run_info.get('status')} states={states}")

    environment = os.environ.copy()
    environment["HF_TOKEN"] = token
    environment["HUGGING_FACE_HUB_TOKEN"] = token
    environment["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
    environment["HF_HUB_ETAG_TIMEOUT"] = "120"
    environment["HF_XET_HIGH_PERFORMANCE"] = "1"
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["PYTHONPATH"] = os.pathsep.join([str(repo_dir), environment.get("PYTHONPATH", "")])

    trials = [
        (
            "candidate_a_w96_d10",
            candidate_a,
            dict(
                epochs=140,
                batch_size=4,
                repeats=5,
                patch_grid="24,24",
                width=96,
                depth=10,
                lr=2.5e-4,
                weight_decay=2.0e-4,
                ema_decay=0.995,
                patience=28,
                min_delta=1.0e-5,
                seed=20260727,
            ),
        ),
        (
            "candidate_b_w112_d12",
            candidate_b,
            dict(
                epochs=140,
                batch_size=3,
                repeats=5,
                patch_grid="24,24",
                width=112,
                depth=12,
                lr=2.0e-4,
                weight_decay=2.0e-4,
                ema_decay=0.995,
                patience=28,
                min_delta=1.0e-5,
                seed=20260728,
            ),
        ),
    ]

    results: list[dict[str, Any]] = []
    for name, output_dir, config in trials:
        print("=" * 96)
        print("TRAINING", name)
        print("=" * 96)
        if not _complete(output_dir):
            if output_dir.exists():
                shutil.rmtree(output_dir)
            _run(
                _train_command(
                    repo_dir,
                    cache_dir,
                    output_dir,
                    args.cache_repo,
                    **config,
                ),
                cwd=repo_dir,
                env=environment,
            )
        else:
            print("Reusing completed candidate:", output_dir)
        _mirror(output_dir, mirror_dir / name)
        summary = _summary(output_dir)
        results.append(
            {
                "name": name,
                "output_dir": str(output_dir),
                "checkpoint": str(output_dir / "voir_real_mage_albedo_v2.pt"),
                "summary": summary,
                "mae": float(summary["final_validation"]["mae"]),
            }
        )

    results.sort(key=lambda item: item["mae"])
    winner = results[0]
    winner_checkpoint = Path(winner["checkpoint"])
    winner_config = winner["summary"]["model_config"]
    print("CANDIDATE_WINNER=", json.dumps({"name": winner["name"], "mae": winner["mae"], "config": winner_config}, indent=2))

    print("=" * 96)
    print("LOW-LR LARGE-CROP FINE-TUNE")
    print("=" * 96)
    if not _complete(finetune_dir):
        if finetune_dir.exists():
            shutil.rmtree(finetune_dir)
        _run(
            _train_command(
                repo_dir,
                cache_dir,
                finetune_dir,
                args.cache_repo,
                epochs=120,
                batch_size=3,
                repeats=8,
                patch_grid="28,28",
                width=int(winner_config["width"]),
                depth=int(winner_config["depth"]),
                lr=5.0e-5,
                weight_decay=1.0e-4,
                ema_decay=0.997,
                patience=30,
                min_delta=5.0e-6,
                seed=20260729,
                init_checkpoint=winner_checkpoint,
            ),
            cwd=repo_dir,
            env=environment,
        )
    else:
        print("Reusing completed fine-tune:", finetune_dir)
    _mirror(finetune_dir, mirror_dir / "finetune")
    finetune_summary = _summary(finetune_dir)

    v1_summary: dict[str, Any] | None = None
    local_v1 = Path("/content/voir_pilot48_real/model/summary.json")
    if local_v1.exists():
        v1_summary = json.loads(local_v1.read_text(encoding="utf-8"))
    else:
        try:
            from huggingface_hub import hf_hub_download

            v1_path = hf_hub_download(
                repo_id=args.v1_model_repo,
                filename="summary.json",
                repo_type="model",
                token=token,
            )
            v1_summary = json.loads(Path(v1_path).read_text(encoding="utf-8"))
        except Exception as exc:
            print("Could not load v1 summary:", exc)

    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(finetune_dir, final_dir)
    experiments_dir = final_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    for item in results:
        source = Path(item["output_dir"])
        destination = experiments_dir / item["name"]
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("summary.json", "history.jsonl"):
            if (source / filename).exists():
                shutil.copy2(source / filename, destination / filename)

    final_metrics = finetune_summary["final_validation"]
    v1_metrics = None if v1_summary is None else v1_summary.get("final_validation")
    selection = {
        "status": "PASS",
        "cache_repo": args.cache_repo,
        "model_repo": args.model_repo,
        "candidates": [
            {
                "name": item["name"],
                "mae": item["mae"],
                "psnr": item["summary"]["final_validation"]["psnr"],
                "ssim_7x7": item["summary"]["final_validation"]["ssim_7x7"],
                "global_ssim": item["summary"]["final_validation"]["global_ssim"],
                "model_config": item["summary"]["model_config"],
                "epochs_completed": item["summary"]["epochs_completed"],
            }
            for item in results
        ],
        "selected_candidate": winner["name"],
        "selected_candidate_mae": winner["mae"],
        "finetuned_metrics": final_metrics,
        "v1_metrics": v1_metrics,
        "mae_improvement_over_v1": None
        if v1_metrics is None
        else (float(v1_metrics["mae"]) - float(final_metrics["mae"])) / max(float(v1_metrics["mae"]), 1e-12),
        "completed_unix": time.time(),
    }
    (final_dir / "v2_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    for source_path, name in (
        (repo_dir / "voir/real_mage_v2.py", "real_mage_v2.py"),
        (repo_dir / "scripts/train_hf_real_mage_v2.py", "train_hf_real_mage_v2.py"),
    ):
        shutil.copy2(source_path, final_dir / name)
    _mirror(final_dir, mirror_dir / "final")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.model_repo, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.model_repo,
        repo_type="model",
        folder_path=final_dir,
        commit_message="Train longer multi-scale Pilot48 albedo v2 model",
    )
    print("HF_MODEL_COMMIT=", commit.oid)
    print("HF_MODEL_URL=https://huggingface.co/" + args.model_repo)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
