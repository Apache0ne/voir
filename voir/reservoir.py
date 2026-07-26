from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from .projection import feature_hash_project
from .schedule import beta_flow_sigmas
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

    def _hook(self, layer_idx: int):
        def collect(_module, _inputs, output):
            image_tokens = output[1] if isinstance(output, (tuple, list)) else output
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

    def stack(self) -> torch.Tensor:
        counts = {idx: len(items) for idx, items in self.by_layer.items()}
        if not counts or min(counts.values(), default=0) == 0:
            raise RuntimeError("no Mage hidden states were captured")
        steps = min(counts.values())
        layer_tensors = [torch.cat(self.by_layer[idx][:steps], dim=0) for idx in self.layers]
        return torch.stack(layer_tensors, dim=1)


class _BetaFlowScheduler:
    """FlowMatchEuler scheduler whose timesteps are fixed to the VOIR beta schedule."""

    def __init__(self, sigmas: torch.Tensor):
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler
        except ImportError as exc:
            raise RuntimeError("Install VOIR with the 'mage' extra to capture Mage states") from exc
        self._scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False
        )
        self._voir_sigmas = sigmas.detach().cpu().float()
        self.timesteps = None
        self.sigmas = None

    def set_shift(self, _shift: float) -> None:
        return None

    def set_timesteps(self, *args, device=None, **kwargs) -> None:
        self._scheduler.set_timesteps(sigmas=self._voir_sigmas[:-1].tolist(), device=device)
        self.timesteps = self._scheduler.timesteps
        self.sigmas = self._scheduler.sigmas

    def step(self, *args, **kwargs):
        return self._scheduler.step(*args, **kwargs)


class MageEditReservoir:
    """Frozen Mage-Flow-Edit-Turbo state extractor."""

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
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        image = image.convert("RGB")
        w, h = image.size
        scale = min(1.0, max_size / max(w, h)) if max_size else 1.0
        out_h = max(16, int(h * scale) // 16 * 16)
        out_w = max(16, int(w * scale) // 16 * 16)
        sigmas = beta_flow_sigmas(
            self.config.steps, self.config.alpha, self.config.beta,
            self.config.start, self.config.end, self.config.shift,
        )
        model = self.pipeline.model
        prior_scheduler = getattr(model, "scheduler", None)
        model.scheduler = _BetaFlowScheduler(sigmas)
        try:
            with _HiddenCollector(model.transformer, self.config, (out_h, out_w), num_refs=1) as collector:
                edited = self.pipeline.edit(
                    [prompt], [[image]], seeds=[int(seed)], steps=self.config.steps,
                    cfg=1.0, heights=[out_h], widths=[out_w],
                )[0]
            features = collector.stack()
            captured_layers = collector.layers
        finally:
            model.scheduler = prior_scheduler
        state = ReservoirState(
            features=features,
            output_size=(out_h, out_w),
            sigmas=sigmas,
            layer_indices=captured_layers,
            source="microsoft/Mage-Flow-Edit-Turbo",
            metadata={
                "prompt": prompt,
                "seed": int(seed),
                "projection_channels": self.config.projection_channels,
                "sampler": "flow_euler_beta",
                "requested_comfy_sampler": "dpmpp_sde_gpu",
            },
        ).validate()
        return state, edited


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
        for _ in range(self.steps):
            state = torch.tanh(base + 0.75 * self.recurrent_conv(state))
            states.append(state.detach().cpu())
        features = torch.stack(states, dim=0).unsqueeze(1)
        h, w = image.shape[-2:]
        return ReservoirState(
            features=features[:, :, 0],
            output_size=(h, w),
            sigmas=torch.linspace(1, 0, self.steps + 1),
            layer_indices=(0,),
            source="voir/ToyImageReservoir",
        ).validate()
