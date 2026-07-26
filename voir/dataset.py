from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .state import ReservoirState


def _pil_to_tensor(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    h, w = size
    if image.size != (w, h):
        image = image.resize((w, h), Image.Resampling.BICUBIC)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class CachedAlbedoDataset(Dataset):
    """JSONL dataset with state and albedo paths per line."""

    def __init__(self, manifest: str | Path):
        self.manifest = Path(manifest)
        base = self.manifest.parent
        self.records = []
        for line in self.manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if not Path(item["state"]).is_absolute():
                item["state"] = str((base / item["state"]).resolve())
            if not Path(item["albedo"]).is_absolute():
                item["albedo"] = str((base / item["albedo"]).resolve())
            self.records.append(item)
        if not self.records:
            raise ValueError(f"manifest has no samples: {self.manifest}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        item = self.records[index]
        state = ReservoirState.load(item["state"])
        target = _pil_to_tensor(Image.open(item["albedo"]), state.output_size)
        return state.flattened().squeeze(0), target
