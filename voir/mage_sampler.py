from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from PIL import Image

from .sampling import SamplerTrace, sample_dpmpp_sde_gpu
from .schedule import beta_flow_sigmas


@dataclass(frozen=True)
class MageSamplerConfig:
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
    vl_cond_long_edge: int = 384
    prefer_torchsde: bool = True


def _cpu_tensor(value: torch.Tensor | None, float_dtype: torch.dtype = torch.float16) -> torch.Tensor | None:
    if value is None:
        return None
    result = value.detach().cpu().contiguous()
    if result.is_floating_point():
        result = result.to(float_dtype)
    return result


def _python_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return tuple(_python_value(item) for item in value)
    if isinstance(value, list):
        return [_python_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _python_value(item) for key, item in value.items()}
    return value


@dataclass
class MageSamplerCache:
    """Portable CPU cache for one complete Mage edit trajectory.

    Large floating-point tensors are stored as float16. The cache includes the
    reference VAE tokens, initial/final target tokens, conditioning tensors, and
    every DPM++ SDE model-evaluation latent, denoised estimate, and velocity.
    """

    reference_tokens: torch.Tensor
    reference_ids: torch.Tensor
    reference_shapes: Any
    initial_target_tokens: torch.Tensor
    final_target_tokens: torch.Tensor
    target_ids: torch.Tensor
    image_ids: torch.Tensor
    image_cu: torch.Tensor
    image_shapes: Any
    text_tokens: torch.Tensor
    text_cu: torch.Tensor
    text_mask: torch.Tensor | None
    text_vector: torch.Tensor
    text_lengths: list[int]
    schedule_sigmas: torch.Tensor
    model_eval_sigmas: torch.Tensor
    eval_latents: torch.Tensor
    eval_denoised: torch.Tensor
    eval_velocity: torch.Tensor
    eval_solver_indices: torch.Tensor
    eval_stages: torch.Tensor
    target_grid: tuple[int, int]
    output_size: tuple[int, int]
    packed_length: int
    target_length: int
    metadata: dict[str, Any]

    def validate(self) -> "MageSamplerCache":
        evaluations = int(self.model_eval_sigmas.numel())
        for name, tensor in (
            ("eval_latents", self.eval_latents),
            ("eval_denoised", self.eval_denoised),
            ("eval_velocity", self.eval_velocity),
        ):
            if tensor.ndim != 4 or tensor.shape[0] != evaluations:
                raise ValueError(f"{name} must be [T,B,N,C] with T={evaluations}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite values")
        if self.eval_solver_indices.numel() != evaluations or self.eval_stages.numel() != evaluations:
            raise ValueError("solver indices and stages must match model evaluations")
        if self.initial_target_tokens.shape != self.final_target_tokens.shape:
            raise ValueError("initial and final target token shapes must match")
        if self.initial_target_tokens.shape[1] != self.target_length:
            raise ValueError("target_length does not match target token tensor")
        return self

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "reference_tokens": self.reference_tokens,
            "reference_ids": self.reference_ids,
            "reference_shapes": self.reference_shapes,
            "initial_target_tokens": self.initial_target_tokens,
            "final_target_tokens": self.final_target_tokens,
            "target_ids": self.target_ids,
            "image_ids": self.image_ids,
            "image_cu": self.image_cu,
            "image_shapes": self.image_shapes,
            "text_tokens": self.text_tokens,
            "text_cu": self.text_cu,
            "text_mask": self.text_mask,
            "text_vector": self.text_vector,
            "text_lengths": self.text_lengths,
            "schedule_sigmas": self.schedule_sigmas,
            "model_eval_sigmas": self.model_eval_sigmas,
            "eval_latents": self.eval_latents,
            "eval_denoised": self.eval_denoised,
            "eval_velocity": self.eval_velocity,
            "eval_solver_indices": self.eval_solver_indices,
            "eval_stages": self.eval_stages,
            "target_grid": self.target_grid,
            "output_size": self.output_size,
            "packed_length": self.packed_length,
            "target_length": self.target_length,
            "metadata": self.metadata,
        }


@dataclass
class MageDetailedEditResult:
    output: Image.Image
    trace: SamplerTrace
    target_length: int
    cache: MageSamplerCache | None


class MageDPMppSDEEditSampler:
    """Standalone Mage-Flow-Edit sampler using Beta + DPM++ SDE device noise.

    This class does not import or call ComfyUI. The scheduler and solver are ports
    of the relevant source equations, adapted to Mage's rectified-flow velocity:
    ``x0 = x_sigma - sigma * velocity``.
    """

    def __init__(self, pipeline, config: MageSamplerConfig | None = None):
        self.pipeline = pipeline
        self.model = pipeline.model
        self.device = torch.device(pipeline.device)
        self.config = config or MageSamplerConfig()
        self.model.eval().requires_grad_(False)

    @staticmethod
    def _load_image(image: Image.Image | str | Path) -> Image.Image:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        return image.convert("RGB")

    @torch.no_grad()
    def edit(
        self,
        image: Image.Image | str | Path | Sequence[Image.Image | str | Path],
        prompt: str,
        *,
        seed: int = 42,
        max_size: int | None = 512,
        height: int | None = None,
        width: int | None = None,
        gs_key=None,
        on_model_eval: Callable[[float, int], None] | None = None,
    ) -> tuple[Image.Image, SamplerTrace, int]:
        """Legacy compact API retained for existing callers."""
        result = self.edit_detailed(
            image,
            prompt,
            seed=seed,
            max_size=max_size,
            height=height,
            width=width,
            gs_key=gs_key,
            on_model_eval=on_model_eval,
            capture_cache=False,
        )
        return result.output, result.trace, result.target_length

    @torch.no_grad()
    def edit_detailed(
        self,
        image: Image.Image | str | Path | Sequence[Image.Image | str | Path],
        prompt: str,
        *,
        seed: int = 42,
        max_size: int | None = 512,
        height: int | None = None,
        width: int | None = None,
        gs_key=None,
        on_model_eval: Callable[[float, int], None] | None = None,
        on_sampler_eval: Callable[[dict[str, Any]], None] | None = None,
        capture_cache: bool = True,
    ) -> MageDetailedEditResult:
        try:
            from einops import rearrange
            from mage_flow.pipeline import (
                _build_pack_ctx,
                _decode_one,
                _edit_target_size,
                _encode_edits_packed,
                _lens_to_cu,
                _preprocess_ref_image,
                _resize_long_edge,
                _slice_packed,
                _template_info,
                _velocity,
            )
            from mage_flow.models.modules.mage_latent import encode_noise, resolve_gs_key
            from mage_flow.models.utils import get_noise
        except ImportError as exc:
            raise RuntimeError(
                "Install the Microsoft Mage source package before using the Mage sampler"
            ) from exc

        refs_in = list(image) if isinstance(image, (list, tuple)) else [image]
        if not refs_in:
            raise ValueError("at least one reference image is required")
        refs = [self._load_image(item) for item in refs_in]
        out_h, out_w = _edit_target_size(refs[0], max_size, height, width)

        verdict = self.model.txt_enc.screen_edit(prompt, refs)
        if verdict.violates:
            raise ValueError(verdict.banner())

        dev = self.device
        torch.manual_seed(int(seed))
        ref_tensors = [_preprocess_ref_image(ref, out_h, out_w, dev) for ref in refs]
        ref_tok, ref_shapes, ref_ids = self.model.compute_vae_encodings(ref_tensors, with_ids=True)
        ref_tok = ref_tok.to(torch.bfloat16)

        channels = self.model.vae.latent_channels
        base_noise = get_noise(
            num_samples=1,
            channel=channels,
            height=out_h,
            width=out_w,
            device=dev,
            dtype=torch.bfloat16,
            seed=int(seed),
        )
        noise = encode_noise(
            tuple(base_noise.shape[1:]),
            key=resolve_gs_key(gs_key),
            seed=int(seed),
            device=dev,
            dtype=torch.bfloat16,
        )
        _, _, gh, gw = noise.shape
        target = rearrange(noise, "b c h w -> b (h w) c").float()
        initial_target = target.detach().clone()
        target_len = target.shape[1]

        target_ids = torch.zeros(gh, gw, 3, device=dev)
        target_ids[..., 1] += torch.arange(gh, device=dev)[:, None]
        target_ids[..., 2] += torch.arange(gw, device=dev)[None, :]
        target_ids = rearrange(target_ids, "h w c -> (h w) c").unsqueeze(0)

        img_ids = torch.cat([target_ids, ref_ids.to(dev)], dim=1)
        packed_len = target_len + ref_tok.shape[1]
        img_cu = _lens_to_cu([packed_len], dev)
        shape_seq = [(1, gh, gw)] + [shape[0] for shape in ref_shapes]
        img_shapes = [shape_seq]

        info = _template_info("mage-flow-edit")
        template = info.get("template", "{}")
        drop_idx = int(info.get("start_idx", 0))
        vl_refs = [_resize_long_edge(ref, self.config.vl_cond_long_edge) for ref in refs]
        txt_flat, vec_all, text_lens = _encode_edits_packed(
            self.model, [vl_refs], [prompt], template, drop_idx, dev
        )
        txt, txt_cu, txt_mask, vec = _slice_packed(txt_flat, vec_all, text_lens, 0, 1, dev)
        ctx = _build_pack_ctx(
            img_ids,
            img_cu,
            img_shapes,
            [packed_len],
            txt,
            txt_cu,
            txt_mask,
            vec,
            None,
            None,
            None,
            None,
            1.0,
            False,
            False,
            dev,
        )

        schedule = beta_flow_sigmas(
            steps=self.config.steps,
            alpha=self.config.alpha,
            beta=self.config.beta,
            start=self.config.start,
            end=self.config.end,
            shift=self.config.shift,
            train_timesteps=self.config.train_timesteps,
            device=dev,
        )
        eval_index = 0
        velocity_evals: list[torch.Tensor] = []
        latent_evals: list[torch.Tensor] = []
        denoised_evals: list[torch.Tensor] = []
        solver_indices: list[int] = []
        stages: list[int] = []

        def predict_x0(x: torch.Tensor, sigma_batch: torch.Tensor) -> torch.Tensor:
            nonlocal eval_index
            sigma_value = float(sigma_batch.reshape(-1)[0])
            if on_model_eval is not None:
                on_model_eval(sigma_value, eval_index)
            eval_index += 1
            packed = torch.cat([x.to(ref_tok.dtype), ref_tok], dim=1)
            velocity = _velocity(self.model.transformer, packed, ctx, sigma_value)
            velocity_target = velocity[:, :target_len].float()
            if capture_cache:
                velocity_evals.append(_cpu_tensor(velocity_target))
            sigma_view = sigma_batch.reshape(-1, *([1] * (x.ndim - 1))).to(x.dtype)
            return x - sigma_view * velocity_target

        def sampler_callback(payload: dict[str, Any]) -> None:
            if capture_cache:
                latent_evals.append(_cpu_tensor(payload["x"]))
                denoised_evals.append(_cpu_tensor(payload["denoised"]))
                solver_indices.append(int(payload["i"]))
                stages.append(int(payload["stage"]))
            if on_sampler_eval is not None:
                on_sampler_eval(payload)

        sampled, trace = sample_dpmpp_sde_gpu(
            predict_x0,
            target,
            schedule,
            seed=int(seed),
            eta=self.config.eta,
            s_noise=self.config.s_noise,
            r=self.config.r,
            shift=self.config.shift,
            callback=sampler_callback,
            prefer_torchsde=self.config.prefer_torchsde,
        )
        output = _decode_one(self.model, sampled, out_h, out_w, dev)

        cache = None
        if capture_cache:
            if not (len(latent_evals) == len(denoised_evals) == len(velocity_evals)):
                raise RuntimeError("sampler cache lists are misaligned")
            cache = MageSamplerCache(
                reference_tokens=_cpu_tensor(ref_tok),
                reference_ids=_cpu_tensor(ref_ids, torch.float32),
                reference_shapes=_python_value(ref_shapes),
                initial_target_tokens=_cpu_tensor(initial_target),
                final_target_tokens=_cpu_tensor(sampled),
                target_ids=_cpu_tensor(target_ids, torch.float32),
                image_ids=_cpu_tensor(img_ids, torch.float32),
                image_cu=_cpu_tensor(img_cu, torch.float32),
                image_shapes=_python_value(img_shapes),
                text_tokens=_cpu_tensor(txt),
                text_cu=_cpu_tensor(txt_cu, torch.float32),
                text_mask=_cpu_tensor(txt_mask, torch.float32),
                text_vector=_cpu_tensor(vec),
                text_lengths=[int(value) for value in text_lens],
                schedule_sigmas=trace.schedule_sigmas.float().cpu(),
                model_eval_sigmas=trace.model_eval_sigmas.float().cpu(),
                eval_latents=torch.stack(latent_evals, dim=0),
                eval_denoised=torch.stack(denoised_evals, dim=0),
                eval_velocity=torch.stack(velocity_evals, dim=0),
                eval_solver_indices=torch.tensor(solver_indices, dtype=torch.int16),
                eval_stages=torch.tensor(stages, dtype=torch.int8),
                target_grid=(int(gh), int(gw)),
                output_size=(int(out_h), int(out_w)),
                packed_length=int(packed_len),
                target_length=int(target_len),
                metadata={
                    "prompt": prompt,
                    "seed": int(seed),
                    "sampler": trace.sampler,
                    "noise_backend": trace.noise_backend,
                    "eta": trace.eta,
                    "s_noise": trace.s_noise,
                    "r": trace.r,
                    "beta_alpha": self.config.alpha,
                    "beta_beta": self.config.beta,
                    "sigma_start": self.config.start,
                    "sigma_end": self.config.end,
                    "shift": self.config.shift,
                    "train_timesteps": self.config.train_timesteps,
                    "vl_cond_long_edge": self.config.vl_cond_long_edge,
                    "template": template,
                    "template_drop_index": drop_idx,
                    "reference_count": len(refs),
                    "latent_channels": int(channels),
                    "storage_float_dtype": "float16",
                },
            ).validate()

        return MageDetailedEditResult(output, trace, target_len, cache)
