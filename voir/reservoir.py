from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .albedo_features import fixed_albedo_features
from .mage_sampler import MageDPMppSDEEditSampler, MageSamplerCache, MageSamplerConfig
from .projection import feature_hash_project
from .state import ReservoirState


@dataclass(frozen=True)
class CaptureConfig:
    layers: tuple[int, ...] = (0, 12, 23)
    projection_channels: int = 64
    projection_seed: int = 1337
    steps: int = 4
    alpha: float = 0.60
    beta: float = 0.80
    start: float = 1.0
    end: float = 0.0
    shift: float = 6.0
    train_timesteps: int = 1000
    eta: float = 1.0
    s_noise: float = 1.0
    r: float = 0.5
    prefer_torchsde: bool = True


def _infer_grid(token_count: int, height: int, width: int) -> tuple[int, int]:
    target_ratio = height / max(width, 1)
    best = None
    for h in range(1, int(token_count**0.5) + 1):
        if token_count % h:
            continue
        w = token_count // h
        for gh, gw in ((h, w), (w, h)):
            error = abs((gh / gw) - target_ratio)
            if best is None or error < best[0]:
                best = (error, gh, gw)
    if best is None:
        raise ValueError(f"cannot infer a grid for {token_count} tokens")
    return best[1], best[2]


class _HiddenCollector:
    def __init__(self, transformer: nn.Module, config: CaptureConfig, output_size: tuple[int, int], num_refs: int):
        self.transformer = transformer
        self.config = config
        self.output_size = output_size
        self.num_refs = num_refs
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.layers = config.layers
        self.by_layer: dict[int, list[torch.Tensor]] = {i: [] for i in self.layers}
        self.eval_sigmas: list[float] = []
        self.current_sigma: float | None = None
        self.current_eval_index: int | None = None

    def begin_eval(self, sigma: float, eval_index: int) -> None:
        self.current_sigma = float(sigma)
        self.current_eval_index = int(eval_index)
        self.eval_sigmas.append(float(sigma))

    def _hook(self, layer_idx: int):
        def collect(_module, _inputs, output):
            image_tokens = output[1] if isinstance(output, (tuple, list)) else output
            if self.current_sigma is None:
                raise RuntimeError("hidden-state hook fired outside a sampler model evaluation")
            target_tokens = image_tokens.shape[1] // (self.num_refs + 1)
            x = image_tokens[:, :target_tokens].detach()
            x = feature_hash_project(x, self.config.projection_channels, self.config.projection_seed + layer_idx)
            gh, gw = _infer_grid(target_tokens, *self.output_size)
            x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], gh, gw)
            self.by_layer[layer_idx].append(x.float().cpu())
        return collect

    def __enter__(self):
        blocks = self.transformer.transformer_blocks
        depth = len(blocks)
        normalized = []
        for idx in self.config.layers:
            n = idx if idx >= 0 else depth + idx
            if n < 0 or n >= depth:
                raise IndexError(f"layer {idx} is outside Mage depth {depth}")
            normalized.append(n)
        self.layers = tuple(normalized)
        self.by_layer = {i: [] for i in self.layers}
        for idx in self.layers:
            self.handles.append(blocks[idx].register_forward_hook(self._hook(idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def stack(self) -> tuple[torch.Tensor, torch.Tensor]:
        counts = {idx: len(items) for idx, items in self.by_layer.items()}
        if not counts or min(counts.values(), default=0) == 0:
            raise RuntimeError("no Mage hidden states were captured")
        evaluations = min(min(counts.values()), len(self.eval_sigmas))
        if evaluations == 0:
            raise RuntimeError("no sampler evaluation sigmas were captured")
        layer_tensors = [torch.cat(self.by_layer[idx][:evaluations], dim=0) for idx in self.layers]
        features = torch.stack(layer_tensors, dim=1)
        sigmas = torch.tensor(self.eval_sigmas[:evaluations], dtype=torch.float32)
        return features, sigmas


class MageEditReservoir:
    """Frozen Mage-Flow-Edit-Turbo state extractor using native Beta + DPM++ SDE."""

    def __init__(self, pipeline, config: CaptureConfig | None = None):
        self.pipeline = pipeline
        self.config = config or CaptureConfig()
        self.pipeline.model.eval().requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        repo: str = "microsoft/Mage-Flow-Edit-Turbo",
        device: str = "cuda",
        config: CaptureConfig | None = None,
    ) -> "MageEditReservoir":
        try:
            from mage_flow.pipeline import MageFlowPipeline
        except ImportError as exc:
            raise RuntimeError(
                "The Microsoft Mage source package is required. Clone microsoft/Mage and install it editable."
            ) from exc
        pipe = MageFlowPipeline.from_pretrained(repo, device=device)
        return cls(pipe, config=config)

    @torch.no_grad()
    def capture(
        self,
        image: Image.Image | str | Path,
        prompt: str = "remove illumination, shadows, highlights, and reflections; output diffuse albedo only",
        seed: int = 42,
        max_size: int = 512,
    ) -> tuple[ReservoirState, Image.Image]:
        state, edited, _ = self.capture_detailed(image, prompt=prompt, seed=seed, max_size=max_size)
        return state, edited

    @torch.no_grad()
    def capture_detailed(
        self,
        image: Image.Image | str | Path,
        prompt: str = "remove illumination, shadows, highlights, and reflections; output diffuse albedo only",
        seed: int = 42,
        max_size: int = 512,
    ) -> tuple[ReservoirState, Image.Image, MageSamplerCache]:
        """Capture projected hidden states plus the complete sampler/conditioning cache."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        image = image.convert("RGB")
        w, h = image.size
        scale = min(1.0, max_size / max(w, h)) if max_size else 1.0
        out_h = max(16, int(h * scale) // 16 * 16)
        out_w = max(16, int(w * scale) // 16 * 16)

        sampler_config = MageSamplerConfig(
            steps=self.config.steps,
            alpha=self.config.alpha,
            beta=self.config.beta,
            start=self.config.start,
            end=self.config.end,
            shift=self.config.shift,
            train_timesteps=self.config.train_timesteps,
            eta=self.config.eta,
            s_noise=self.config.s_noise,
            r=self.config.r,
            prefer_torchsde=self.config.prefer_torchsde,
        )
        sampler = MageDPMppSDEEditSampler(self.pipeline, sampler_config)
        with _HiddenCollector(
            self.pipeline.model.transformer,
            self.config,
            (out_h, out_w),
            num_refs=1,
        ) as collector:
            result = sampler.edit_detailed(
                image,
                prompt,
                seed=int(seed),
                max_size=None,
                height=out_h,
                width=out_w,
                on_model_eval=collector.begin_eval,
                capture_cache=True,
            )
        if result.cache is None:
            raise RuntimeError("detailed Mage capture did not return a sampler cache")
        features, eval_sigmas = collector.stack()

        resized = image.resize((out_w, out_h), Image.Resampling.BICUBIC)
        source = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1)
        aux_features = fixed_albedo_features(source, size=tuple(features.shape[-2:])).cpu()

        state = ReservoirState(
            features=features,
            output_size=(out_h, out_w),
            sigmas=eval_sigmas,
            layer_indices=collector.layers,
            source="microsoft/Mage-Flow-Edit-Turbo",
            aux_features=aux_features,
            metadata={
                "prompt": prompt,
                "seed": int(seed),
                "projection_channels": self.config.projection_channels,
                "projection_seed": self.config.projection_seed,
                "sampler": result.trace.sampler,
                "schedule_sigmas": result.trace.schedule_sigmas.tolist(),
                "model_eval_sigmas": result.trace.model_eval_sigmas.tolist(),
                "noise_backend": result.trace.noise_backend,
                "eta": result.trace.eta,
                "s_noise": result.trace.s_noise,
                "r": result.trace.r,
                "beta_alpha": self.config.alpha,
                "beta_beta": self.config.beta,
                "sigma_start": self.config.start,
                "sigma_end": self.config.end,
                "shift": self.config.shift,
                "trajectory_channels": int(features.shape[0] * features.shape[1] * features.shape[2]),
                "auxiliary_channels": int(aux_features.shape[0]),
                "sampler_cache_format": "voir_mage_sampler_cache_v1",
                "sampler_model_evaluations": int(result.cache.model_eval_sigmas.numel()),
            },
        ).validate()
        return state, result.output, result.cache


class ToyImageReservoir(nn.Module):
    """Small frozen nonlinear recurrent image reservoir for CPU tests and CI."""

    def __init__(self, channels: int = 24, steps: int = 4, seed: int = 1234):
        super().__init__()
        self.channels = channels
        self.steps = steps
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.input_conv = nn.Conv2d(3, channels, 3, padding=1, bias=False)
            self.recurrent_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.mix_conv = nn.Conv2d(channels, channels, 1, bias=False)
            nn.init.orthogonal_(self.mix_conv.weight.reshape(channels, channels))
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def capture(self, image: torch.Tensor) -> ReservoirState:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must be [B,3,H,W] or [3,H,W]")
        if image.shape[0] != 1:
            raise ValueError("ToyImageReservoir.capture currently accepts one image at a time")
        base = self.input_conv(image)
        state = torch.zeros_like(base)
        states = []
        leak_rates = torch.linspace(0.35, 0.80, self.steps).tolist()
        for leak in leak_rates:
            state = torch.tanh(
                base
                + float(leak) * self.recurrent_conv(state)
                + 0.15 * self.mix_conv(state)
            )
            states.append(state.detach().cpu())
        features = torch.stack(states, dim=0).unsqueeze(1)[:, :, 0]
        h, w = image.shape[-2:]
        aux_features = fixed_albedo_features(image[0].detach().cpu())
        return ReservoirState(
            features=features,
            output_size=(h, w),
            sigmas=torch.linspace(1, 0, self.steps),
            layer_indices=(0,),
            source="voir/ToyImageReservoir",
            aux_features=aux_features,
            metadata={
                "trajectory_channels": int(self.steps * self.channels),
                "auxiliary_channels": int(aux_features.shape[0]),
            },
        ).validate()
