from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def _extract_labels(example: dict) -> list[int]:
    objects = example.get("objects") or {}
    categories = objects.get("category") or objects.get("categories") or []
    return [int(value) for value in categories]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="detection-datasets/coco")
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--output", default="datasets/hf_coco_real30")
    parser.add_argument("--min-side", type=int, default=480)
    parser.add_argument("--max-scan", type=int, default=2000)
    args = parser.parse_args()

    output = Path(args.output)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    stream = load_dataset(args.dataset, split=args.split, streaming=True)
    records: list[dict] = []
    seen_label_sets: set[tuple[int, ...]] = set()

    for scanned, example in enumerate(stream):
        if scanned >= args.max_scan or len(records) >= args.count:
            break
        image = example.get("image")
        if image is None:
            continue
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        width, height = image.size
        if min(width, height) < args.min_side:
            continue

        labels = _extract_labels(example)
        label_key = tuple(sorted(set(labels)))
        # Prefer diverse scenes for this tiny real-photo adaptation set.
        if label_key and label_key in seen_label_sets and len(records) < args.count - 5:
            continue
        if label_key:
            seen_label_sets.add(label_key)

        index = len(records)
        filename = f"{index:03d}.jpg"
        image.save(image_dir / filename, quality=95, subsampling=0)
        source_id = example.get("image_id", example.get("id", scanned))
        records.append(
            {
                "index": index,
                "image": f"images/{filename}",
                "source_dataset": args.dataset,
                "source_split": args.split,
                "source_id": int(source_id) if isinstance(source_id, (int, float)) else str(source_id),
                "width": width,
                "height": height,
                "object_category_ids": labels,
                "albedo_target": None,
                "target_type": "pending_real_photo_pseudo_label",
            }
        )

    if len(records) != args.count:
        raise RuntimeError(f"requested {args.count} images but selected {len(records)}")

    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "count": len(records),
        "selection": {
            "real_photos_only": True,
            "minimum_side": args.min_side,
            "diverse_object_sets": True,
            "deterministic_stream_order": True,
        },
        "purpose": "real-photo albedo adaptation and validation",
        "note": "These are real RGB photos. Albedo targets must be generated or manually supplied before supervised training.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
