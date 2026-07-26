from __future__ import annotations

import torch


def feature_hash_project(tokens: torch.Tensor, out_channels: int, seed: int = 1337) -> torch.Tensor:
    """Fixed deterministic feature hashing from [B,N,D] to [B,N,C]."""
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B,N,D]")
    if out_channels <= 0 or out_channels >= tokens.shape[-1]:
        return tokens
    d = tokens.shape[-1]
    idx = torch.arange(d, device=tokens.device, dtype=torch.int64)
    mixed = idx * 1103515245 + int(seed) * 12345 + 1013904223
    buckets = torch.remainder(mixed, out_channels)
    signs = torch.where(torch.bitwise_and(mixed, 1) == 0, 1.0, -1.0).to(tokens.dtype)
    out = tokens.new_zeros((*tokens.shape[:-1], out_channels))
    scatter_idx = buckets.view(1, 1, d).expand(tokens.shape[0], tokens.shape[1], d)
    out.scatter_add_(-1, scatter_idx, tokens * signs.view(1, 1, d))
    return out / (d / out_channels) ** 0.5
