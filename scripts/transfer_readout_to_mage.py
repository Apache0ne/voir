from __future__ import annotations

import argparse
import json
from pathlib import Path

from voir.transfer import save_transferred_readout, transfer_intrinsic_readout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer a real-photo intrinsic_v3 readout to cached Mage trajectory width."
    )
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--trajectory-channels", type=int, required=True)
    parser.add_argument("--auxiliary-channels", type=int, default=63)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--unfreeze-all",
        action="store_true",
        help="make the complete readout trainable immediately instead of projection-only warm-up",
    )
    args = parser.parse_args()

    model, report = transfer_intrinsic_readout(
        args.checkpoint,
        trajectory_channels=args.trajectory_channels,
        auxiliary_channels=args.auxiliary_channels,
        device=args.device,
        freeze_shared=not args.unfreeze_all,
    )
    save_transferred_readout(args.output, model, report, source_checkpoint=args.checkpoint)
    report["output"] = str(Path(args.output))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
