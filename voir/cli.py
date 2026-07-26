from __future__ import annotations

import argparse
import json

import torch
from PIL import Image

from .device import resolve_device
from .readout import AlbedoReadout
from .reservoir import CaptureConfig, MageEditReservoir
from .state import ReservoirState
from .train import train_albedo


def _layers(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def main():
    parser = argparse.ArgumentParser(prog="voir")
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture-mage", help="capture frozen Mage edit states")
    cap.add_argument("image")
    cap.add_argument("state_output")
    cap.add_argument("--preview-output")
    cap.add_argument("--model", default="microsoft/Mage-Flow-Edit-Turbo")
    cap.add_argument("--device", default="cuda")
    cap.add_argument("--prompt", default="remove illumination, shadows, highlights, and reflections; output diffuse albedo only")
    cap.add_argument("--max-size", type=int, default=512)
    cap.add_argument("--seed", type=int, default=42)
    cap.add_argument("--steps", type=int, default=4)
    cap.add_argument("--alpha", type=float, default=0.60)
    cap.add_argument("--beta", type=float, default=0.80)
    cap.add_argument("--layers", type=_layers, default=(0, 12, 23))
    cap.add_argument("--projection-channels", type=int, default=64)

    train = sub.add_parser("train-albedo", help="train only the albedo readout")
    train.add_argument("manifest")
    train.add_argument("output")
    train.add_argument("--device", default="auto")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--width", type=int, default=96)
    train.add_argument("--depth", type=int, default=4)

    pred = sub.add_parser("predict-albedo", help="run a trained readout on saved states")
    pred.add_argument("state")
    pred.add_argument("checkpoint")
    pred.add_argument("output")
    pred.add_argument("--device", default="auto")

    args = parser.parse_args()
    if args.command == "capture-mage":
        config = CaptureConfig(
            layers=args.layers,
            projection_channels=args.projection_channels,
            steps=args.steps,
            alpha=args.alpha,
            beta=args.beta,
        )
        reservoir = MageEditReservoir.from_pretrained(args.model, args.device, config)
        state, preview = reservoir.capture(args.image, args.prompt, args.seed, args.max_size)
        state.save(args.state_output)
        if args.preview_output:
            preview.save(args.preview_output)
        print(json.dumps({
            "state": args.state_output,
            "shape": list(state.features.shape),
            "readout_channels": state.readout_channels,
        }))
    elif args.command == "train-albedo":
        result = train_albedo(
            args.manifest, args.output, args.device, args.epochs,
            args.batch_size, args.lr, args.width, args.depth,
        )
        print(json.dumps(result, default=str))
    elif args.command == "predict-albedo":
        device = resolve_device(args.device)
        state = ReservoirState.load(args.state)
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = AlbedoReadout.from_checkpoint(payload, device)
        out = model.predict_state(state, device)[0].permute(1, 2, 0).cpu().numpy()
        Image.fromarray((out.clip(0, 1) * 255).astype("uint8")).save(args.output)
