from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from voir.metrics import albedo_metrics
from voir.real_mage import RealMageAlbedoNet, load_real_mage_cache


def _to_image(value: torch.Tensor) -> Image.Image:
    array = (
        value.detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict albedo from one complete actual-Mage cache.")
    parser.add_argument("cache")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--root", default=None, help="Cache root containing inputs/albedo/masks.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--comparison", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    sample = load_real_mage_cache(args.cache, root=args.root)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = RealMageAlbedoNet.from_checkpoint(payload, device=device).eval()
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16

    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        prediction = model(
            sample["hidden"].unsqueeze(0).to(device),
            sample["sampler"].unsqueeze(0).to(device),
            sample["auxiliary"].unsqueeze(0).to(device),
            sample["source"].unsqueeze(0).to(device),
        )[0].float().cpu()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _to_image(prediction).save(output, optimize=True)

    metrics = albedo_metrics(
        prediction.unsqueeze(0),
        sample["target"].unsqueeze(0),
        sample["mask"].unsqueeze(0),
    )
    print(json.dumps({"output": str(output), "metrics": metrics}, indent=2))

    if args.comparison:
        source = _to_image(sample["source"])
        target = _to_image(sample["target"])
        predicted = _to_image(prediction)
        canvas = Image.new("RGB", (source.width * 3, source.height))
        canvas.paste(source, (0, 0))
        canvas.paste(target, (source.width, 0))
        canvas.paste(predicted, (source.width * 2, 0))
        comparison = Path(args.comparison)
        comparison.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(comparison, optimize=True)
        print("comparison:", comparison)


if __name__ == "__main__":
    main()
