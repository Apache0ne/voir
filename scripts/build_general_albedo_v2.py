"""Build a scene/object-disjoint paired RGB-to-albedo dataset from public HF sources.

Sources:
- GDAOSU/Olbedo: real outdoor/aerial scenes with paired albedo.
- UMTRI/MatPredict: synthetic rendered objects with paired base-color maps.
- pbr-rooms-benchmark/PBR-Rooms: synthetic indoor scenes with paired albedo.

The output schema is compatible with scripts/cache_mage_real16.py. All pairs are
center-cropped to a common square size so every Mage cache has one fixed token grid.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps


def _as_pil(value: Any, mode: str = "RGB") -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert(mode)
    if isinstance(value, dict):
        if value.get("path"):
            return Image.open(value["path"]).convert(mode)
        if value.get("bytes"):
            import io

            return Image.open(io.BytesIO(value["bytes"])).convert(mode)
    if isinstance(value, (str, os.PathLike)):
        return Image.open(value).convert(mode)
    raise TypeError(f"cannot convert {type(value)!r} to PIL image")


def _linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return np.where(value <= 0.0031308, 12.92 * value, 1.055 * np.power(value, 1.0 / 2.4) - 0.055)


def _read_numeric_image(path: str | Path, *, linear_to_srgb: bool = False) -> Image.Image:
    path = Path(path)
    if path.suffix.lower() not in {".exr", ".hdr", ".npy"}:
        return Image.open(path).convert("RGB")
    if path.suffix.lower() == ".npy":
        value = np.load(path)
    else:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2

        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None:
            raise RuntimeError(f"OpenCV could not read {path}")
        if value.ndim == 3:
            value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    if value.shape[-1] > 3:
        value = value[..., :3]
    finite = np.isfinite(value)
    if not finite.any():
        raise ValueError(f"numeric image has no finite pixels: {path}")
    value = np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
    if value.max() > 1.0 or value.min() < 0.0:
        lo, hi = np.percentile(value[finite], [0.1, 99.9])
        value = (value - float(lo)) / max(float(hi - lo), 1e-6)
    value = np.clip(value, 0.0, 1.0)
    if linear_to_srgb:
        value = _linear_to_srgb(value)
    return Image.fromarray(np.round(value * 255.0).astype(np.uint8), mode="RGB")


def _fit_pair(
    image: Image.Image,
    albedo: Image.Image,
    mask: Image.Image | None,
    size: int,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    image = image.convert("RGB")
    albedo = albedo.convert("RGB")
    if albedo.size != image.size:
        albedo = albedo.resize(image.size, Image.Resampling.BICUBIC)
    if mask is None:
        mask = Image.new("L", image.size, 255)
    else:
        mask = mask.convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
    target = (int(size), int(size))
    image = ImageOps.fit(image, target, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    albedo = ImageOps.fit(albedo, target, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    mask = ImageOps.fit(mask, target, method=Image.Resampling.NEAREST, centering=(0.5, 0.5))
    return image, albedo, mask


def _split_for_group(domain: str, group: str, seed: int, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}|{domain}|{group}".encode("utf-8")).digest()
    number = int.from_bytes(digest[:8], "big") / float(2**64)
    return "validation" if number < float(validation_fraction) else "train"


def _write_records(output: Path, records: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in {
        "pairs.jsonl": records,
        "train_pairs.jsonl": [row for row in records if row["subset"] == "train"],
        "validation_pairs.jsonl": [row for row in records if row["subset"] == "validation"],
    }.items():
        (output / name).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )


def _save_pair(
    output: Path,
    index: int,
    image: Image.Image,
    albedo: Image.Image,
    mask: Image.Image | None,
    *,
    size: int,
    domain: str,
    group: str,
    source_id: str,
    seed: int,
    validation_fraction: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image, albedo, mask = _fit_pair(image, albedo, mask, size)
    stem = f"{index:06d}"
    for subdir in ("images", "albedo", "masks"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    image_path = output / "images" / f"{stem}.png"
    albedo_path = output / "albedo" / f"{stem}.png"
    mask_path = output / "masks" / f"{stem}.png"
    image.save(image_path, optimize=True)
    albedo.save(albedo_path, optimize=True)
    mask.save(mask_path, optimize=True)
    return {
        "index": int(index),
        "subset": _split_for_group(domain, group, seed, validation_fraction),
        "image": str(image_path.relative_to(output)),
        "albedo": str(albedo_path.relative_to(output)),
        "mask": str(mask_path.relative_to(output)),
        "domain": domain,
        "group": group,
        "source_id": source_id,
        "width": int(size),
        "height": int(size),
        "target_type": "paired_albedo",
        **(metadata or {}),
    }


def _balanced_take(items: list[dict[str, Any]], count: int, group_key: str, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item[group_key])].append(item)
    for values in groups.values():
        rng.shuffle(values)
    keys = list(groups)
    rng.shuffle(keys)
    chosen: list[dict[str, Any]] = []
    cursor = 0
    while len(chosen) < count and keys:
        key = keys[cursor % len(keys)]
        if groups[key]:
            chosen.append(groups[key].pop())
        if not groups[key]:
            keys.remove(key)
            cursor = 0
        else:
            cursor += 1
    if len(chosen) < count:
        raise RuntimeError(f"requested {count} samples but selected only {len(chosen)}")
    return chosen


def _olbedo_pairs(count: int, seed: int) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    stream = load_dataset("GDAOSU/Olbedo", split="train_selected", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=max(2048, count * 4))
    candidates: list[dict[str, Any]] = []
    per_group: Counter[str] = Counter()
    cap = max(8, count // 8)
    for row_index, row in enumerate(stream):
        scene = str(row.get("scene") or row.get("model_name") or "unknown")
        group = f"{scene}|{row.get('date', 'unknown')}"
        if per_group[group] >= cap:
            continue
        image = row.get("image")
        albedo = row.get("albedo")
        if image is None or albedo is None:
            continue
        candidates.append(
            {
                "image": _as_pil(image, "RGB"),
                "albedo": _as_pil(albedo, "RGB"),
                "mask": None if row.get("mask") is None else _as_pil(row.get("mask"), "L"),
                "group": group,
                "source_id": f"GDAOSU/Olbedo:{row.get('frame_id', row_index)}",
                "metadata": {
                    "source_dataset": "GDAOSU/Olbedo",
                    "source_split": "train_selected",
                    "source_row": int(row_index),
                    "scene": scene,
                    "date": str(row.get("date", "unknown")),
                    "lighting": str(row.get("lighting", "unknown")),
                },
            }
        )
        per_group[group] += 1
        if len(candidates) >= count:
            break
    if len(candidates) < count:
        raise RuntimeError(f"Olbedo yielded {len(candidates)} usable pairs, requested {count}")
    return candidates[:count]


def _matpredict_pairs(count: int, seed: int, token: str | None) -> Iterable[dict[str, Any]]:
    from huggingface_hub import HfApi, hf_hub_download

    repo = "UMTRI/MatPredict"
    api = HfApi(token=token)
    files = api.list_repo_files(repo, repo_type="dataset")
    file_set = set(files)
    candidates = []
    for image_path in files:
        lower = image_path.lower()
        if "/images/" not in lower or not lower.endswith((".png", ".jpg", ".jpeg")):
            continue
        marker = lower.index("/images/")
        albedo_path = image_path[:marker] + "/albedo/" + image_path[marker + len("/images/") :]
        if albedo_path not in file_set:
            continue
        prefix = image_path[:marker]
        parts = prefix.split("/")
        group = "/".join(parts[:-1]) if len(parts) > 1 else prefix
        candidates.append(
            {
                "image_path": image_path,
                "albedo_path": albedo_path,
                "group": group,
                "source_id": f"{repo}:{prefix}/{Path(image_path).stem}",
            }
        )
    selected = _balanced_take(candidates, count, "group", seed)
    for item in selected:
        image_file = hf_hub_download(repo, item["image_path"], repo_type="dataset", token=token)
        albedo_file = hf_hub_download(repo, item["albedo_path"], repo_type="dataset", token=token)
        yield {
            "image": Image.open(image_file).convert("RGB"),
            "albedo": Image.open(albedo_file).convert("RGB"),
            "mask": None,
            "group": item["group"],
            "source_id": item["source_id"],
            "metadata": {
                "source_dataset": repo,
                "source_split": "object_disjoint_hash",
                "scene": item["group"],
            },
        }


def _choose_manifest_column(columns: list[str], kind: str) -> str:
    scored = []
    for column in columns:
        lower = column.lower()
        score = 0
        if kind == "albedo":
            if "albedo" in lower or "basecolor" in lower or "base_color" in lower:
                score += 100
        elif kind == "image":
            if "albedo" in lower:
                continue
            if lower in {"image", "rgb", "image_path", "rgb_path", "input", "input_path"}:
                score += 100
            if "image" in lower or "rgb" in lower:
                score += 20
        elif kind == "scene":
            if lower == "scene":
                score += 100
            if "scene" in lower:
                score += 20
        if "path" in lower:
            score += 5
        scored.append((score, column))
    score, column = max(scored, default=(0, ""))
    if score <= 0:
        raise KeyError(f"could not find {kind} column in {columns}")
    return column


def _pbrrooms_manifest_candidates(token: str | None) -> tuple[str, list[dict[str, Any]]]:
    from huggingface_hub import hf_hub_download

    repo = "pbr-rooms-benchmark/PBR-Rooms"
    manifest = hf_hub_download(repo, "benchmark_manifest.csv", repo_type="dataset", token=token)
    with open(manifest, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("PBR-Rooms benchmark_manifest.csv is empty")
    columns = list(rows[0])
    image_column = _choose_manifest_column(columns, "image")
    albedo_column = _choose_manifest_column(columns, "albedo")
    try:
        scene_column = _choose_manifest_column(columns, "scene")
    except KeyError:
        scene_column = image_column
    candidates = []
    for row in rows:
        image_path = str(row.get(image_column) or "").strip().lstrip("./")
        albedo_path = str(row.get(albedo_column) or "").strip().lstrip("./")
        if not image_path or not albedo_path:
            continue
        scene = str(row.get(scene_column) or Path(image_path).parts[0])
        candidates.append(
            {
                "image_path": image_path,
                "albedo_path": albedo_path,
                "group": scene,
                "source_id": f"{repo}:{image_path}",
                "metadata": row,
            }
        )
    return repo, candidates


def _pbrrooms_file_candidates(token: str | None) -> tuple[str, list[dict[str, Any]]]:
    from huggingface_hub import HfApi

    repo = "pbr-rooms-benchmark/PBR-Rooms"
    files = HfApi(token=token).list_repo_files(repo, repo_type="dataset")
    file_set = set(files)
    candidates = []
    for image_path in files:
        lower = image_path.lower()
        if "/image/" not in lower or not lower.endswith((".png", ".jpg", ".jpeg", ".exr")):
            continue
        marker = lower.index("/image/")
        suffix = image_path[marker + len("/image/") :]
        albedo_prefix = image_path[:marker] + "/Albedo/"
        possibilities = [albedo_prefix + suffix]
        stem = str(Path(suffix).with_suffix(""))
        possibilities.extend([albedo_prefix + stem + ext for ext in (".exr", ".png", ".npy")])
        albedo_path = next((value for value in possibilities if value in file_set), None)
        if albedo_path is None:
            continue
        prefix = image_path[:marker]
        group = "/".join(prefix.split("/")[:-1]) or prefix
        candidates.append(
            {
                "image_path": image_path,
                "albedo_path": albedo_path,
                "group": group,
                "source_id": f"{repo}:{image_path}",
                "metadata": {},
            }
        )
    return repo, candidates


def _pbrrooms_pairs(count: int, seed: int, token: str | None) -> Iterable[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    repo, candidates = _pbrrooms_manifest_candidates(token)
    selected = _balanced_take(candidates, min(count, len(candidates)), "group", seed)
    delivered = 0
    failures = []
    for item in selected:
        try:
            image_file = hf_hub_download(repo, item["image_path"], repo_type="dataset", token=token)
            albedo_file = hf_hub_download(repo, item["albedo_path"], repo_type="dataset", token=token)
            yield {
                "image": _read_numeric_image(image_file, linear_to_srgb=Path(image_file).suffix.lower() in {".exr", ".hdr"}),
                "albedo": _read_numeric_image(albedo_file, linear_to_srgb=True),
                "mask": None,
                "group": item["group"],
                "source_id": item["source_id"],
                "metadata": {
                    "source_dataset": repo,
                    "source_split": "benchmark_scene_hash",
                    "scene": item["group"],
                },
            }
            delivered += 1
            if delivered >= count:
                return
        except Exception as exc:
            failures.append(f"{item['image_path']}: {exc}")
    if delivered < count:
        repo, fallback = _pbrrooms_file_candidates(token)
        fallback = [item for item in fallback if item["source_id"] not in {x["source_id"] for x in selected}]
        for item in _balanced_take(fallback, min(count - delivered, len(fallback)), "group", seed + 1):
            image_file = hf_hub_download(repo, item["image_path"], repo_type="dataset", token=token)
            albedo_file = hf_hub_download(repo, item["albedo_path"], repo_type="dataset", token=token)
            yield {
                "image": _read_numeric_image(image_file, linear_to_srgb=Path(image_file).suffix.lower() in {".exr", ".hdr"}),
                "albedo": _read_numeric_image(albedo_file, linear_to_srgb=True),
                "mask": None,
                "group": item["group"],
                "source_id": item["source_id"],
                "metadata": {
                    "source_dataset": repo,
                    "source_split": "released_files_scene_hash",
                    "scene": item["group"],
                },
            }
            delivered += 1
            if delivered >= count:
                return
    raise RuntimeError(f"PBR-Rooms delivered {delivered}/{count} pairs. First failures: {failures[:3]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the VOIR multi-domain general albedo v2 dataset.")
    parser.add_argument("--output", default="/content/voir_general_v2/pairs")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--olbedo-count", type=int, default=256)
    parser.add_argument("--matpredict-count", type=int, default=256)
    parser.add_argument("--pbrrooms-count", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    expected = args.olbedo_count + args.matpredict_count + args.pbrrooms_count
    metadata_path = output / "metadata.json"
    if not args.overwrite and metadata_path.exists() and (output / "pairs.jsonl").exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        records = [line for line in (output / "pairs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if metadata.get("expected_pairs") == expected and len(records) == expected:
            print(json.dumps({"status": "REUSED", **metadata}, indent=2))
            return
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    records: list[dict[str, Any]] = []
    sources = [
        ("olbedo", args.olbedo_count, _olbedo_pairs(args.olbedo_count, args.seed)),
        ("matpredict", args.matpredict_count, _matpredict_pairs(args.matpredict_count, args.seed + 1, token)),
        ("pbrrooms", args.pbrrooms_count, _pbrrooms_pairs(args.pbrrooms_count, args.seed + 2, token)),
    ]
    for domain, requested, iterator in sources:
        print(f"Collecting {requested} pairs from {domain}...")
        before = len(records)
        for item in iterator:
            record = _save_pair(
                output,
                len(records),
                item["image"],
                item["albedo"],
                item.get("mask"),
                size=args.size,
                domain=domain,
                group=str(item["group"]),
                source_id=str(item["source_id"]),
                seed=args.seed,
                validation_fraction=args.validation_fraction,
                metadata=item.get("metadata"),
            )
            records.append(record)
            if len(records) - before >= requested:
                break
        if len(records) - before != requested:
            raise RuntimeError(f"{domain} produced {len(records) - before}/{requested} pairs")
        _write_records(output, records)
        print(f"  {domain}: {len(records) - before} saved")

    # Interleave domains so partial cache runs are not single-domain.
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        buckets[row["domain"]].append(row)
    interleaved: list[dict[str, Any]] = []
    while any(buckets.values()):
        for domain in sorted(buckets):
            if buckets[domain]:
                interleaved.append(buckets[domain].pop(0))
    # Rename files to match interleaved order and rewrite indices atomically.
    staging = output / ".reorder"
    if staging.exists():
        shutil.rmtree(staging)
    for subdir in ("images", "albedo", "masks"):
        (staging / subdir).mkdir(parents=True, exist_ok=True)
    reordered = []
    for new_index, row in enumerate(interleaved):
        stem = f"{new_index:06d}"
        for key, subdir in (("image", "images"), ("albedo", "albedo"), ("mask", "masks")):
            source = output / row[key]
            destination = staging / subdir / f"{stem}.png"
            shutil.copy2(source, destination)
            row[key] = f"{subdir}/{stem}.png"
        row["index"] = new_index
        reordered.append(row)
    for subdir in ("images", "albedo", "masks"):
        shutil.rmtree(output / subdir)
        shutil.move(str(staging / subdir), str(output / subdir))
    shutil.rmtree(staging)
    _write_records(output, reordered)

    counts = Counter(row["domain"] for row in reordered)
    split_counts = Counter(row["subset"] for row in reordered)
    domain_splits = Counter(f"{row['domain']}:{row['subset']}" for row in reordered)
    if split_counts["train"] == 0 or split_counts["validation"] == 0:
        raise RuntimeError(f"invalid split distribution: {dict(split_counts)}")
    metadata = {
        "status": "PASS",
        "expected_pairs": expected,
        "size": args.size,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "domain_counts": dict(counts),
        "split_counts": dict(split_counts),
        "domain_split_counts": dict(domain_splits),
        "sources": {
            "GDAOSU/Olbedo": "real outdoor/aerial paired albedo",
            "UMTRI/MatPredict": "synthetic rendered object paired albedo",
            "pbr-rooms-benchmark/PBR-Rooms": "synthetic indoor paired albedo; CC BY-NC 4.0",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
