from __future__ import annotations

import argparse
import io
import json
import shutil
import time
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

from PIL import Image


ROWS_API = "https://datasets-server.huggingface.co/rows"
USER_AGENT = "voir-real-albedo/0.4"


def _json_get(url: str, retries: int = 5) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = exc
            time.sleep(min(2**attempt, 16))
    raise RuntimeError(f"failed to fetch {url}") from last_error


def _bytes_get(url: str, retries: int = 5) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = exc
            time.sleep(min(2**attempt, 16))
    raise RuntimeError(f"failed to download asset {url}") from last_error


def _rows_url(dataset: str, split: str, offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": "default",
            "split": split,
            "offset": int(offset),
            "length": int(length),
        }
    )
    return f"{ROWS_API}?{query}"


def _asset_url(cell) -> str | None:
    if isinstance(cell, str) and cell.startswith(("http://", "https://")):
        return cell
    if not isinstance(cell, dict):
        return None
    for key in ("src", "url"):
        value = cell.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return None


def _as_rgb(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def _as_mask(data: bytes | None, size: tuple[int, int]) -> Image.Image:
    if data is None:
        return Image.new("L", size, 255)
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("L")


def _candidate_windows(total_rows: int, windows: int, window_size: int) -> list[int]:
    if total_rows <= window_size:
        return [0]
    maximum = total_rows - window_size
    if windows <= 1:
        return [0]
    return sorted({round(index * maximum / (windows - 1)) for index in range(windows)})


def _collect_candidates(dataset: str, split: str, windows: int = 32, window_size: int = 8) -> tuple[list[dict], int]:
    first = _json_get(_rows_url(dataset, split, 0, 1))
    total_rows = int(first["num_rows_total"])
    candidates: list[dict] = []
    seen_rows: set[int] = set()
    for offset in _candidate_windows(total_rows, windows, window_size):
        payload = _json_get(_rows_url(dataset, split, offset, window_size))
        for wrapper in payload.get("rows", []):
            row_index = int(wrapper.get("row_idx", -1))
            row = wrapper.get("row") or {}
            if row_index < 0 or row_index in seen_rows:
                continue
            image_url = _asset_url(row.get("image"))
            albedo_url = _asset_url(row.get("albedo"))
            if image_url is None or albedo_url is None:
                continue
            seen_rows.add(row_index)
            candidates.append(
                {
                    "row_index": row_index,
                    "frame_id": int(row.get("frame_id", row_index)),
                    "scene": str(row.get("scene", "unknown")),
                    "date": str(row.get("date", "unknown")),
                    "lighting": str(row.get("lighting", "unknown")),
                    "image_url": image_url,
                    "albedo_url": albedo_url,
                    "mask_url": _asset_url(row.get("mask")),
                }
            )
    return candidates, total_rows


def _balanced_select(candidates: list[dict], count: int) -> list[dict]:
    groups: dict[tuple[str, str, str], deque[dict]] = defaultdict(deque)
    for candidate in sorted(candidates, key=lambda item: item["row_index"]):
        key = (candidate["scene"], candidate["date"], candidate["lighting"])
        groups[key].append(candidate)
    if not groups:
        raise RuntimeError("Hugging Face row API returned no usable RGB/albedo pairs")

    selected: list[dict] = []
    ordered_keys = sorted(groups)
    while len(selected) < count and any(groups.values()):
        for key in ordered_keys:
            if groups[key]:
                selected.append(groups[key].popleft())
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise RuntimeError(f"requested {count} pairs but only found {len(selected)}")
    return selected


def _validation_indices(records: list[dict], validation_count: int) -> set[int]:
    by_group: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_group[(record["scene"], record["date"], record["lighting"])].append(index)
    validation: set[int] = set()
    groups = sorted(by_group, key=lambda key: (-len(by_group[key]), key))
    cursor = 0
    while len(validation) < validation_count:
        key = groups[cursor % len(groups)]
        available = [index for index in reversed(by_group[key]) if index not in validation]
        if available and len(by_group[key]) - sum(index in validation for index in by_group[key]) > 1:
            validation.add(available[0])
        cursor += 1
        if cursor > validation_count * len(groups) * 4:
            break
    if len(validation) < validation_count:
        for index in reversed(range(len(records))):
            validation.add(index)
            if len(validation) == validation_count:
                break
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a tiny paired real-photo/albedo subset from Hugging Face."
    )
    parser.add_argument("--dataset", default="GDAOSU/Olbedo")
    parser.add_argument("--split", default="train_selected")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--train-count", type=int, default=24)
    parser.add_argument("--output", default="datasets/hf_olbedo_real30")
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=8)
    args = parser.parse_args()

    if args.count < 2:
        raise ValueError("count must be at least 2")
    if not 1 <= args.train_count < args.count:
        raise ValueError("train-count must be between 1 and count-1")

    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    image_dir = output / "images"
    albedo_dir = output / "albedo"
    mask_dir = output / "masks"
    image_dir.mkdir(parents=True)
    albedo_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    candidates, total_rows = _collect_candidates(
        args.dataset,
        args.split,
        windows=args.windows,
        window_size=args.window_size,
    )
    selected = _balanced_select(candidates, args.count)
    validation = _validation_indices(selected, args.count - args.train_count)
    records: list[dict] = []

    for index, candidate in enumerate(selected):
        image = _as_rgb(_bytes_get(candidate["image_url"]))
        albedo = _as_rgb(_bytes_get(candidate["albedo_url"]))
        if albedo.size != image.size:
            albedo = albedo.resize(image.size, Image.Resampling.BICUBIC)
        mask_data = _bytes_get(candidate["mask_url"]) if candidate["mask_url"] else None
        mask = _as_mask(mask_data, image.size)
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)

        stem = f"{index:03d}"
        image.save(image_dir / f"{stem}.png", optimize=True)
        albedo.save(albedo_dir / f"{stem}.png", optimize=True)
        mask.save(mask_dir / f"{stem}.png", optimize=True)
        subset = "validation" if index in validation else "train"
        records.append(
            {
                "index": index,
                "subset": subset,
                "image": f"images/{stem}.png",
                "albedo": f"albedo/{stem}.png",
                "mask": f"masks/{stem}.png",
                "source_dataset": args.dataset,
                "source_split": args.split,
                "source_row": candidate["row_index"],
                "frame_id": candidate["frame_id"],
                "scene": candidate["scene"],
                "date": candidate["date"],
                "lighting": candidate["lighting"],
                "width": image.width,
                "height": image.height,
                "target_type": "paired_multiview_consistent_albedo",
            }
        )

    train_records = [record for record in records if record["subset"] == "train"]
    validation_records = [record for record in records if record["subset"] == "validation"]

    def write_manifest(name: str, values: list[dict]) -> None:
        (output / name).write_text(
            "".join(json.dumps(record) + "\n" for record in values),
            encoding="utf-8",
        )

    write_manifest("pairs.jsonl", records)
    write_manifest("train_pairs.jsonl", train_records)
    write_manifest("validation_pairs.jsonl", validation_records)
    group_counts: dict[str, int] = defaultdict(int)
    for record in records:
        group_counts[f"{record['scene']}|{record['date']}|{record['lighting']}"] += 1
    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "dataset_rows": total_rows,
        "candidate_rows_scanned": len(candidates),
        "count": len(records),
        "train_pairs": len(train_records),
        "validation_pairs": len(validation_records),
        "selection": "balanced round-robin across scene/date/lighting groups",
        "groups": dict(sorted(group_counts.items())),
        "license": "cc-by-4.0",
        "contents": ["real UAV RGB photo", "paired albedo", "validity mask"],
        "purpose": "real-photo supervised adaptation of the VOIR albedo readout",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
