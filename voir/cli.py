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

    cap = sub.add_parser(
        "capture-mage",
        help="capture frozen Mage states with native Beta + DPM++ SDE",
    )
    cap.add_argument("image")
    cap.add_argument("state_output")
    cap.add_argument("--preview-output")
    cap.add_argument("--model", default="microsoft/Mage-Flow-Edit-Turbo")
    cap.add_argument("--device", default="cuda")
    cap.add_argument(
        "--prompt",
        default="remove illumination, shadows, highlights, and reflections; output diffuse albedo only",
    )
    cap.add_argument("--max-size", type=int, default=512)
    cap.add_argument("--seed", type=int, default=42)
    cap.add_argument("--steps", type=int, default=4)
    cap.add_argument("--alpha", type=float, default=0.60)
    cap.add_argument("--beta", type=float, default=0.80)
    cap.add_argument("--sigma-start", type=float, default=1.0)
    cap.add_argument("--sigma-end", type=float, default=0.0)
    cap.add_argument("--shift", type=float, default=6.0)
    cap.add_argument("--train-timesteps", type=int, default=1000)
    cap.add_argument("--eta", type=float, default=1.0)
    cap.add_argument("--s-noise", type=float, default=1.0)
    cap.add_argument("--r", type=float, default=0.5)
    cap.add_argument(
        "--native-brownian",
        action="store_true",
        help="use the built-in Brownian bridge instead of torchsde",
    )
    cap.add_argument("--layers", type=_layers, default=(0, 12, 23))
    cap.add_argument("--projection-channels", type=int, default=64)

    train = sub.add_parser("train-albedo", help="train only the albedo readout")
    train.add_argument("manifest")
    train.add_argument("output")
    train.add_argument("--validation-manifest")
    train.add_argument("--device", default="auto")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--lr", type=float, default=1.5e-3)
    train.add_argument("--width", type=int, default=32)
    train.add_argument("--depth", type=int, default=7)
    train.add_argument(
        "--architecture",
        choices=["dilated_v2", "legacy"],
        default="dilated_v2",
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--num-workers", type=int, default=0)

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
            start=args.sigma_start,
            end=args.sigma_end,
            shift=args.shift,
            train_timesteps=args.train_timesteps,
            eta=args.eta,
            s_noise=args.s_noise,
            r=args.r,
            prefer_torchsde=not args.native_brownian,
        )
        reservoir = MageEditReservoir.from_pretrained(args.model, args.device, config)
        state, preview = reservoir.capture(args.image, args.prompt, args.seed, args.max_size)
        state.save(args.state_output)
        if args.preview_output:
            preview.save(args.preview_output)
        print(
            json.dumps(
                {
                    "state": args.state_output,
                    "shape": list(state.features.shape),
                    "readout_channels": state.readout_channels,
                    "trajectory_channels": state.trajectory_channels,
                    "auxiliary_channels": state.auxiliary_channels,
                    "sampler": state.metadata.get("sampler"),
                    "schedule_sigmas": state.metadata.get("schedule_sigmas"),
                    "model_evaluations": len(state.metadata.get("model_eval_sigmas", [])),
                    "noise_backend": state.metadata.get("noise_backend"),
                }
            )
        )
    elif args.command == "train-albedo":
        result = train_albedo(
            args.manifest,
            args.output,
            args.device,
            args.epochs,
            args.batch_size,
            args.lr,
            args.width,
            args.depth,
            args.num_workers,
            args.architecture,
            args.validation_manifest,
            args.seed,
        )
        print(json.dumps(result, default=str))
    elif args.command == "predict-albedo":
        device = resolve_device(args.device)
        state = ReservoirState.load(args.state)
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = AlbedoReadout.from_checkpoint(payload, device)
        out = model.predict_state(state, device)[0].permute(1, 2, 0).cpu().numpy()
        Image.fromarray((out.clip(0, 1) * 255).astype("uint8")).save(args.output)
