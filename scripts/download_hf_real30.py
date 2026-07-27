from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def _as_rgb(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    return Image.open(value).convert("RGB")


def _as_mask(value, size: tuple[int, int]) -> Image.Image:
    if value is None:
        return Image.new("L", size, 255)
    if isinstance(value, Image.Image):
        return value.convert("L")
    return Image.open(value).convert("L")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a tiny paired real-photo/albedo subset from Hugging Face."
    )
    parser.add_argument("--dataset", default="GDAOSU/Olbedo")
    parser.add_argument("--split", default="train_selected")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--train-count", type=int, default=24)
    parser.add_argument("--stride", type=int, default=17)
    parser.add_argument("--output", default="datasets/hf_olbedo_real30")
    args = parser.parse_args()

    if args.count < 2:
        raise ValueError("count must be at least 2")
    if not 1 <= args.train_count < args.count:
        raise ValueError("train-count must be between 1 and count-1")
    if args.stride < 1:
        raise ValueError("stride must be >= 1")

    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    image_dir = output / "images"
    albedo_dir = output / "albedo"
    mask_dir = output / "masks"
    image_dir.mkdir(parents=True)
    albedo_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    stream = load_dataset(args.dataset, split=args.split, streaming=True)
    records: list[dict] = []

    for scanned, example in enumerate(stream):
        if scanned % args.stride != 0:
            continue
        image_value = example.get("image")
        albedo_value = example.get("albedo")
        if image_value is None or albedo_value is None:
            continue

        image = _as_rgb(image_value)
        albedo = _as_rgb(albedo_value)
        if albedo.size != image.size:
            albedo = albedo.resize(image.size, Image.Resampling.BICUBIC)
        mask = _as_mask(example.get("mask"), image.size)
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)

        index = len(records)
        stem = f"{index:03d}"
        image_path = image_dir / f"{stem}.png"
        albedo_path = albedo_dir / f"{stem}.png"
        mask_path = mask_dir / f"{stem}.png"
        image.save(image_path, optimize=True)
        albedo.save(albedo_path, optimize=True)
        mask.save(mask_path, optimize=True)

        subset = "train" if index < args.train_count else "validation"
        record = {
            "index": index,
            "subset": subset,
            "image": str(image_path.relative_to(output)),
            "albedo": str(albedo_path.relative_to(output)),
            "mask": str(mask_path.relative_to(output)),
            "source_dataset": args.dataset,
            "source_split": args.split,
            "source_row": scanned,
            "frame_id": int(example.get("frame_id", scanned)),
            "scene": str(example.get("scene", "")),
            "date": str(example.get("date", "")),
            "lighting": str(example.get("lighting", "")),
            "width": image.width,
            "height": image.height,
            "target_type": "paired_multiview_consistent_albedo",
        }
        records.append(record)
        if len(records) == args.count:
            break

    if len(records) != args.count:
        raise RuntimeError(f"requested {args.count} pairs but downloaded {len(records)}")

    def write_manifest(name: str, selected: list[dict]) -> None:
        (output / name).write_text(
            "".join(json.dumps(record) + "\n" for record in selected),
            encoding="utf-8",
        )

    write_manifest("pairs.jsonl", records)
    write_manifest("train_pairs.jsonl", records[: args.train_count])
    write_manifest("validation_pairs.jsonl", records[args.train_count :])

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "count": len(records),
        "train_pairs": args.train_count,
        "validation_pairs": args.count - args.train_count,
        "stride": args.stride,
        "license": "cc-by-4.0",
        "contents": ["real UAV RGB photo", "paired albedo", "confidence/validity mask"],
        "purpose": "first real-photo supervised adaptation of the VOIR albedo readout",
        "note": (
            "Olbedo contains real UAV images with albedo targets derived by a calibrated "
            "multi-view inverse-rendering pipeline. It is substantially more suitable than "
            "unpaired COCO photos for supervised albedo training."
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
