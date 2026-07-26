from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower().strip()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def inference_dtype(device: torch.device, requested: str = "auto") -> torch.dtype:
    requested = requested.lower().strip()
    if requested == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if requested not in mapping:
        raise ValueError(f"unsupported dtype: {requested}")
    dtype = mapping[requested]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("use fp32 or bf16 for CPU execution")
    return dtype
