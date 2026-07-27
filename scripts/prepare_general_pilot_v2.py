"""Prepare a small validated RGB-to-albedo pilot from individually selected files.

This avoids downloading full multi-gigabyte dataset archives. Sources:
- GDAOSU/Olbedo through the Hugging Face Dataset Viewer rows API.
- yGaoJiany/SynCom through individually selected images/albedo files.
- UMTRI/MatPredict through individually selected rendered/Basecolor files.

The output schema is compatible with ``scripts/cache_mage_real16.py`` and includes
scene/object-disjoint train/validation manifests plus a visual contact sheet.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageOps

ROWS_API = "https://datasets-server.huggingface.co/rows"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".exr", ".hdr", ".npy"}
DISPLAY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
USER_AGENT = "voir-fast-pilot-v2/1.0"


def _request_json(url: str, token: str | None = None, retries: int = 10) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=(30, 240))
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last = exc
            delay = min(2**attempt, 30)
            print(f"JSON retry {attempt + 1}/{retries} in {delay}s: {exc}", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"failed to fetch JSON: {url}") from last


def _request_bytes(url: str, token: str | None = None, retries: int = 10) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=(30, 600))
            response.raise_for_status()
            return response.content
        except Exception as exc:
            last = exc
            delay = min(2**attempt, 30)
            print(f"asset retry {attempt + 1}/{retries} in {delay}s: {exc}", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"failed to download asset: {url}") from last


def _asset_url(cell: Any) -> str | None:
    if isinstance(cell, str) and cell.startswith(("http://", "https://")):
        return cell
    if isinstance(cell, dict):
        for key in ("src", "url"):
            value = cell.get(key)
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                return value
    return None


def _rows_url(dataset: str, split: str, offset: int, length: int) -> str:
    return ROWS_API + "?" + urlencode(
        {
            "dataset": dataset,
            "config": "default",
            "split": split,
            "offset": int(offset),
            "length": int(length),
        }
    )


def _even_offsets(total: int, windows: int, length: int) -> list[int]:
    if total <= length:
        return [0]
    maximum = total - length
    return sorted({round(index * maximum / max(windows - 1, 1)) for index in range(windows)})


def _balanced_take(items: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item["group"])].append(item)
    for values in groups.values():
        rng.shuffle(values)
    keys = list(groups)
    rng.shuffle(keys)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while keys and len(selected) < count:
        key = keys[cursor % len(keys)]
        selected.append(groups[key].pop())
        if not groups[key]:
            keys.remove(key)
            cursor = 0
        else:
            cursor += 1
    if len(selected) < count:
        raise RuntimeError(f"balanced selector returned {len(selected)}/{count}")
    return selected


def _olbedo_descriptors(count: int, token: str | None, seed: int) -> list[dict[str, Any]]:
    dataset = "GDAOSU/Olbedo"
    split = "train_selected"
    first = _request_json(_rows_url(dataset, split, 0, 1), token)
    total = int(first["num_rows_total"])
    candidates: list[dict[str, Any]] = []
    seen: set[int] = set()
    # Small row slices avoid fetching the underlying 22 GB Parquet split.
    windows = max(12, count)
    length = 4
    for offset in _even_offsets(total, windows, length):
        payload = _request_json(_rows_url(dataset, split, offset, length), token)
        for wrapper in payload.get("rows", []):
            row_index = int(wrapper.get("row_idx", -1))
            row = wrapper.get("row") or {}
            image_url = _asset_url(row.get("image"))
            albedo_url = _asset_url(row.get("albedo"))
            if row_index < 0 or row_index in seen or not image_url or not albedo_url:
                continue
            scene = str(row.get("scene") or row.get("model_name") or "unknown")
            date = str(row.get("date") or "unknown")
            candidates.append(
                {
                    "kind": "url",
                    "domain": "olbedo",
                    "group": f"{scene}|{date}",
                    "source_id": f"{dataset}:{split}:{row_index}",
                    "image_url": image_url,
                    "albedo_url": albedo_url,
                    "mask_url": _asset_url(row.get("mask")),
                    "metadata": {
                        "source_dataset": dataset,
                        "source_split": split,
                        "source_row": row_index,
                        "scene": scene,
                        "date": date,
                        "lighting": str(row.get("lighting") or "unknown"),
                    },
                }
            )
            seen.add(row_index)
    if len(candidates) < count:
        raise RuntimeError(f"Olbedo row API produced {len(candidates)} candidates; requested {count}")
    return _balanced_take(candidates, count, seed)


def _path_key_before(path: str, marker: str) -> tuple[str, str] | None:
    lower = path.lower()
    marker_lower = marker.lower()
    if marker_lower not in lower:
        return None
    position = lower.index(marker_lower)
    root = path[:position].rstrip("/")
    suffix = path[position + len(marker) :].lstrip("/")
    return root, str(Path(suffix).with_suffix(""))


def _syncom_descriptors(count: int, token: str | None, seed: int) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi

    repo = "yGaoJiany/SynCom"
    print(f"Listing {repo} files...", flush=True)
    files = list(HfApi(token=token).list_repo_files(repo, repo_type="dataset"))
    albedo_map: dict[tuple[str, str], str] = {}
    for path in files:
        if Path(path).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = _path_key_before(path, "/albedo/")
        if key is not None:
            current = albedo_map.get(key)
            if current is None or (Path(path).suffix.lower() == ".png" and Path(current).suffix.lower() != ".png"):
                albedo_map[key] = path
    candidates: list[dict[str, Any]] = []
    for path in files:
        if Path(path).suffix.lower() not in DISPLAY_EXTENSIONS:
            continue
        key = _path_key_before(path, "/images/")
        if key is None or key not in albedo_map:
            continue
        root, stem = key
        parts = [part for part in root.split("/") if part]
        group = "/".join(parts[:3]) if len(parts) >= 3 else root
        candidates.append(
            {
                "kind": "hub",
                "domain": "syncom",
                "group": group,
                "source_id": f"{repo}:{root}:{stem}",
                "repo": repo,
                "image_path": path,
                "albedo_path": albedo_map[key],
                "metadata": {
                    "source_dataset": repo,
                    "source_split": "object_render_pair",
                    "scene": root,
                },
            }
        )
    print(f"  SynCom matched pairs: {len(candidates)}", flush=True)
    if len(candidates) < count:
        raise RuntimeError(f"SynCom produced {len(candidates)} matched RGB/albedo files; requested {count}")
    return _balanced_take(candidates, count, seed)


def _relative_key(path: str, root_name: str) -> str | None:
    normalized = path.replace("\\", "/")
    lower = normalized.lower()
    marker = root_name.lower().rstrip("/") + "/"
    if not lower.startswith(marker):
        return None
    relative = normalized[len(marker) :]
    return str(Path(relative).with_suffix("")).lower()


def _matpredict_descriptors(count: int, token: str | None, seed: int) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi

    repo = "UMTRI/MatPredict"
    print(f"Listing {repo} files...", flush=True)
    files = list(HfApi(token=token).list_repo_files(repo, repo_type="dataset"))
    file_set = set(files)
    candidates: list[dict[str, Any]] = []

    # Current repository layout: rendered[_cropped]/<object>/<variant>/<frame>.png
    # paired with Basecolor/<object>/<variant>/<frame>.<image extension>.
    basecolor_by_key: dict[str, str] = {}
    for path in files:
        if Path(path).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = _relative_key(path, "Basecolor")
        if key is not None:
            current = basecolor_by_key.get(key)
            if current is None or (Path(path).suffix.lower() == ".png" and Path(current).suffix.lower() != ".png"):
                basecolor_by_key[key] = path

    preferred_rgb_roots = ("rendered_cropped", "rendered")
    seen_keys: set[str] = set()
    for rgb_root in preferred_rgb_roots:
        for path in files:
            if Path(path).suffix.lower() not in DISPLAY_EXTENSIONS:
                continue
            key = _relative_key(path, rgb_root)
            if key is None or key in seen_keys or key not in basecolor_by_key:
                continue
            relative_parts = Path(key).parts
            group = "/".join(relative_parts[:2]) if len(relative_parts) >= 2 else str(Path(key).parent)
            candidates.append(
                {
                    "kind": "hub",
                    "domain": "matpredict",
                    "group": group,
                    "source_id": f"{repo}:{rgb_root}:{key}",
                    "repo": repo,
                    "image_path": path,
                    "albedo_path": basecolor_by_key[key],
                    "metadata": {
                        "source_dataset": repo,
                        "source_split": "object_variant_group",
                        "scene": group,
                        "rgb_root": rgb_root,
                    },
                }
            )
            seen_keys.add(key)

    # Also support the documented nested layout <object>/<variant>/images and /albedo.
    nested_albedo: dict[tuple[str, str], str] = {}
    for path in files:
        if Path(path).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = _path_key_before(path, "/albedo/")
        if key is not None:
            nested_albedo[key] = path
    for path in files:
        if Path(path).suffix.lower() not in DISPLAY_EXTENSIONS:
            continue
        key = _path_key_before(path, "/images/")
        if key is None or key not in nested_albedo:
            continue
        root, stem = key
        identity = f"nested:{root}:{stem}".lower()
        if identity in seen_keys:
            continue
        parts = [part for part in root.split("/") if part]
        group = "/".join(parts[:2]) if len(parts) >= 2 else root
        candidates.append(
            {
                "kind": "hub",
                "domain": "matpredict",
                "group": group,
                "source_id": f"{repo}:{identity}",
                "repo": repo,
                "image_path": path,
                "albedo_path": nested_albedo[key],
                "metadata": {
                    "source_dataset": repo,
                    "source_split": "object_variant_group",
                    "scene": group,
                },
            }
        )
        seen_keys.add(identity)

    print(f"  MatPredict files: {len(files)}", flush=True)
    print(f"  MatPredict basecolor keys: {len(basecolor_by_key)}", flush=True)
    print(f"  MatPredict matched pairs: {len(candidates)}", flush=True)
    if len(candidates) < count:
        sample_rgb = [path for path in files if path.lower().startswith(("rendered/", "rendered_cropped/"))][:8]
        sample_albedo = [path for path in files if path.lower().startswith("basecolor/")][:8]
        raise RuntimeError(
            f"MatPredict produced {len(candidates)} matched RGB/basecolor files; requested {count}. "
            f"RGB examples={sample_rgb}; Basecolor examples={sample_albedo}"
        )
    return _balanced_take(candidates, count, seed)


def _decode_numeric(path: str | Path, *, linear_to_srgb: bool) -> Image.Image:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".exr", ".hdr", ".npy"}:
        return Image.open(path).convert("RGB")
    if suffix == ".npy":
        value = np.load(path)
    else:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2

        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None:
            raise RuntimeError(f"OpenCV could not decode {path}")
        if value.ndim == 3:
            value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    if value.shape[-1] > 3:
        value = value[..., :3]
    finite = np.isfinite(value)
    if not finite.any():
        raise ValueError(f"numeric image has no finite samples: {path}")
    value = np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
    if value.max() > 1.0 or value.min() < 0.0:
        low, high = np.percentile(value[finite], [0.1, 99.9])
        value = (value - float(low)) / max(float(high - low), 1e-6)
    value = np.clip(value, 0.0, 1.0)
    if linear_to_srgb:
        value = np.where(value <= 0.0031308, 12.92 * value, 1.055 * np.power(value, 1.0 / 2.4) - 0.055)
    return Image.fromarray(np.round(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")


def _decode_bytes(data: bytes, mode: str = "RGB") -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        return image.convert(mode)


def _load_descriptor(item: dict[str, Any], token: str | None) -> tuple[Image.Image, Image.Image, Image.Image | None]:
    if item["kind"] == "url":
        image = _decode_bytes(_request_bytes(item["image_url"], token), "RGB")
        albedo = _decode_bytes(_request_bytes(item["albedo_url"], token), "RGB")
        mask = None
        if item.get("mask_url"):
            mask = _decode_bytes(_request_bytes(item["mask_url"], token), "L")
        return image, albedo, mask
    from huggingface_hub import hf_hub_download

    image_path = hf_hub_download(
        repo_id=item["repo"], filename=item["image_path"], repo_type="dataset", token=token
    )
    albedo_path = hf_hub_download(
        repo_id=item["repo"], filename=item["albedo_path"], repo_type="dataset", token=token
    )
    image = _decode_numeric(image_path, linear_to_srgb=Path(image_path).suffix.lower() in {".exr", ".hdr"})
    albedo = _decode_numeric(
        albedo_path, linear_to_srgb=Path(albedo_path).suffix.lower() in {".exr", ".hdr", ".npy"}
    )
    return image, albedo, None


def _fit_pair(image: Image.Image, albedo: Image.Image, mask: Image.Image | None, size: int) -> tuple[Image.Image, Image.Image, Image.Image]:
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
    return (
        ImageOps.fit(image, target, method=Image.Resampling.LANCZOS),
        ImageOps.fit(albedo, target, method=Image.Resampling.LANCZOS),
        ImageOps.fit(mask, target, method=Image.Resampling.NEAREST),
    )


def _split_groups(items: list[dict[str, Any]], validation_fraction: float, seed: int) -> set[tuple[str, str]]:
    validation: set[tuple[str, str]] = set()
    by_domain: dict[str, list[str]] = defaultdict(list)
    for item in items:
        if item["group"] not in by_domain[item["domain"]]:
            by_domain[item["domain"]].append(item["group"])
    for domain, groups in by_domain.items():
        ranked = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}|{domain}|{group}".encode()).digest())
        count = max(1, round(len(ranked) * validation_fraction))
        for group in ranked[:count]:
            validation.add((domain, group))
    return validation


def _interleave(domains: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    queues = {name: list(values) for name, values in domains.items()}
    result: list[dict[str, Any]] = []
    while any(queues.values()):
        for name in sorted(queues):
            if queues[name]:
                result.append(queues[name].pop(0))
    return result


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _contact_sheet(output: Path, rows: list[dict[str, Any]], limit: int = 18) -> Path:
    shown = rows[: min(limit, len(rows))]
    thumb = 192
    label = 28
    canvas = Image.new("RGB", (thumb * 2, (thumb + label) * len(shown)), "white")
    draw = ImageDraw.Draw(canvas)
    for line, row in enumerate(shown):
        image = Image.open(output / row["image"]).convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
        albedo = Image.open(output / row["albedo"]).convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
        y = line * (thumb + label)
        canvas.paste(image, (0, y))
        canvas.paste(albedo, (thumb, y))
        draw.text((4, y + thumb + 5), f"{row['index']:03d} {row['domain']} RGB", fill="black")
        draw.text((thumb + 4, y + thumb + 5), "ALBEDO", fill="black")
    path = output / "pilot_contact_sheet.png"
    canvas.save(path, optimize=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a fast Olbedo/SynCom/MatPredict RGB-albedo pilot.")
    parser.add_argument("--output", default="/content/voir_pilot48")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--olbedo-count", type=int, default=16)
    parser.add_argument("--syncom-count", type=int, default=16)
    parser.add_argument("--matpredict-count", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    expected = args.olbedo_count + args.syncom_count + args.matpredict_count
    metadata_path = output / "metadata.json"
    if not args.overwrite and metadata_path.exists() and (output / "pairs.jsonl").exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows = [line for line in (output / "pairs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if metadata.get("expected_pairs") == expected and len(rows) == expected:
            print(json.dumps({"status": "REUSED", **metadata}, indent=2))
            return
    if output.exists():
        shutil.rmtree(output)
    for subdir in ("images", "albedo", "masks"):
        (output / subdir).mkdir(parents=True, exist_ok=True)

    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None
    print("Collecting lightweight descriptors; no full dataset snapshots or zip archives will be downloaded.", flush=True)
    domains = {
        "olbedo": _olbedo_descriptors(args.olbedo_count, token, args.seed),
        "syncom": _syncom_descriptors(args.syncom_count, token, args.seed + 1),
        "matpredict": _matpredict_descriptors(args.matpredict_count, token, args.seed + 2),
    }
    selected = _interleave(domains)
    validation_groups = _split_groups(selected, args.validation_fraction, args.seed)

    records: list[dict[str, Any]] = []
    for index, item in enumerate(selected):
        print(f"[{index + 1:03d}/{expected}] {item['domain']} {item['source_id']}", flush=True)
        image, albedo, mask = _load_descriptor(item, token)
        image, albedo, mask = _fit_pair(image, albedo, mask, args.size)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        albedo_array = np.asarray(albedo, dtype=np.float32) / 255.0
        if image_array.std() < 0.005 or albedo_array.std() < 0.005:
            raise ValueError(f"near-constant pair rejected: {item['source_id']}")
        stem = f"{index:06d}"
        image_path = output / "images" / f"{stem}.png"
        albedo_path = output / "albedo" / f"{stem}.png"
        mask_path = output / "masks" / f"{stem}.png"
        image.save(image_path, optimize=True)
        albedo.save(albedo_path, optimize=True)
        mask.save(mask_path, optimize=True)
        subset = "validation" if (item["domain"], item["group"]) in validation_groups else "train"
        records.append(
            {
                "index": index,
                "subset": subset,
                "domain": item["domain"],
                "group": item["group"],
                "source_id": item["source_id"],
                "image": str(image_path.relative_to(output)),
                "albedo": str(albedo_path.relative_to(output)),
                "mask": str(mask_path.relative_to(output)),
                "width": args.size,
                "height": args.size,
                "target_type": "paired_albedo",
                "rgb_mean": float(image_array.mean()),
                "rgb_std": float(image_array.std()),
                "albedo_mean": float(albedo_array.mean()),
                "albedo_std": float(albedo_array.std()),
                **item.get("metadata", {}),
            }
        )
        _write_jsonl(output / "pairs.partial.jsonl", records)

    _write_jsonl(output / "pairs.jsonl", records)
    _write_jsonl(output / "train_pairs.jsonl", [row for row in records if row["subset"] == "train"])
    _write_jsonl(output / "validation_pairs.jsonl", [row for row in records if row["subset"] == "validation"])
    contact_sheet = _contact_sheet(output, records)
    domain_counts = Counter(row["domain"] for row in records)
    split_counts = Counter(row["subset"] for row in records)
    domain_splits = Counter(f"{row['domain']}:{row['subset']}" for row in records)
    metadata = {
        "status": "PASS",
        "expected_pairs": expected,
        "size": args.size,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "domain_counts": dict(domain_counts),
        "split_counts": dict(split_counts),
        "domain_split_counts": dict(domain_splits),
        "contact_sheet": str(contact_sheet.relative_to(output)),
        "sources": {
            "GDAOSU/Olbedo": "real outdoor/aerial paired albedo, selected through rows API",
            "yGaoJiany/SynCom": "synthetic object paired albedo, selected individual files",
            "UMTRI/MatPredict": "synthetic material/object paired basecolor, selected individual files",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
