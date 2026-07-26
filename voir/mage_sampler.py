from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

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


class MageDPMppSDEEditSampler:
    """Standalone Mage-Flow-Edit sampler using Beta + DPM++ SDE device noise.

    This class does not import or call ComfyUI. The scheduler and solver are ports
    of the relevant source equations, adapted to Mage's rectified-flow velocity:
    x0 = x_sigma - sigma * velocity.
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

        def predict_x0(x: torch.Tensor, sigma_batch: torch.Tensor) -> torch.Tensor:
            nonlocal eval_index
            sigma_value = float(sigma_batch.reshape(-1)[0])
            if on_model_eval is not None:
                on_model_eval(sigma_value, eval_index)
            eval_index += 1
            packed = torch.cat([x.to(ref_tok.dtype), ref_tok], dim=1)
            velocity = _velocity(self.model.transformer, packed, ctx, sigma_value)
            velocity_target = velocity[:, :target_len].float()
            sigma_view = sigma_batch.reshape(-1, *([1] * (x.ndim - 1))).to(x.dtype)
            return x - sigma_view * velocity_target

        sampled, trace = sample_dpmpp_sde_gpu(
            predict_x0,
            target,
            schedule,
            seed=int(seed),
            eta=self.config.eta,
            s_noise=self.config.s_noise,
            r=self.config.r,
            shift=self.config.shift,
            prefer_torchsde=self.config.prefer_torchsde,
        )
        output = _decode_one(self.model, sampled, out_h, out_w, dev)
        return output, trace, target_len
