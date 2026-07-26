from __future__ import annotations

import torch

try:
    from ..voir.readout import AlbedoReadout
    from ..voir.reservoir import CaptureConfig, MageEditReservoir
    from ..voir.state import ReservoirState
except ImportError:
    from voir.readout import AlbedoReadout
    from voir.reservoir import CaptureConfig, MageEditReservoir
    from voir.state import ReservoirState


class VoirBetaSamplingScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
            "alpha": ("FLOAT", {"default": 0.60, "min": 0.01, "max": 50.0, "step": 0.01}),
            "beta": ("FLOAT", {"default": 0.80, "min": 0.01, "max": 50.0, "step": 0.01}),
        }}
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "build"
    CATEGORY = "VOIR/sampling"

    def build(self, model, steps, alpha, beta):
        import comfy.samplers
        sigmas = comfy.samplers.beta_scheduler(
            model.get_model_object("model_sampling"), steps, alpha=alpha, beta=beta
        )
        return (sigmas,)


class VoirSigmasRescale:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sigmas": ("SIGMAS",),
            "start": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            "end": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
        }}
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas_rescaled",)
    FUNCTION = "rescale"
    CATEGORY = "VOIR/sampling"

    def rescale(self, sigmas, start, end):
        lo, hi = sigmas.min(), sigmas.max()
        if torch.isclose(lo, hi):
            return (torch.full_like(sigmas, start),)
        return (((sigmas - lo) * (start - end)) / (hi - lo) + end,)


class VoirKSamplerSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sampler_name": (["dpmpp_sde_gpu", "dpmpp_sde"], {"default": "dpmpp_sde_gpu"})
        }}
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "select"
    CATEGORY = "VOIR/sampling"

    def select(self, sampler_name):
        import comfy.samplers
        return (comfy.samplers.sampler_object(sampler_name),)


class VoirMageTurboSamplingPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
            "alpha": ("FLOAT", {"default": 0.60, "min": 0.01, "max": 50.0, "step": 0.01}),
            "beta": ("FLOAT", {"default": 0.80, "min": 0.01, "max": 50.0, "step": 0.01}),
            "start": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            "end": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            "sampler_name": (["dpmpp_sde_gpu", "dpmpp_sde"], {"default": "dpmpp_sde_gpu"}),
        }}
    RETURN_TYPES = ("SIGMAS", "SAMPLER")
    RETURN_NAMES = ("sigmas", "sampler")
    FUNCTION = "build"
    CATEGORY = "VOIR/sampling"

    def build(self, model, steps, alpha, beta, start, end, sampler_name):
        sigmas = VoirBetaSamplingScheduler().build(model, steps, alpha, beta)[0]
        sigmas = VoirSigmasRescale().rescale(sigmas, start, end)[0]
        sampler = VoirKSamplerSelect().select(sampler_name)[0]
        return sigmas, sampler


_PIPELINES = {}


class VoirMageFlowEditTurboLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("STRING", {"default": "microsoft/Mage-Flow-Edit-Turbo"}),
            "device": (["cuda", "cpu"], {"default": "cuda"}),
        }}
    RETURN_TYPES = ("VOIR_MAGE_PIPELINE",)
    FUNCTION = "load"
    CATEGORY = "VOIR/Mage"

    def load(self, model, device):
        key = (model, device)
        if key not in _PIPELINES:
            _PIPELINES[key] = MageEditReservoir.from_pretrained(model, device)
        return (_PIPELINES[key],)


class VoirMageCaptureAlbedoStates:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "reservoir": ("VOIR_MAGE_PIPELINE",),
            "image": ("IMAGE",),
            "prompt": ("STRING", {
                "default": "remove illumination, shadows, highlights, and reflections; output diffuse albedo only",
                "multiline": True,
            }),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            "max_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
            "layers": ("STRING", {"default": "0,12,23"}),
            "projection_channels": ("INT", {"default": 64, "min": 0, "max": 4096}),
        }}
    RETURN_TYPES = ("VOIR_RESERVOIR_STATE", "IMAGE")
    RETURN_NAMES = ("state", "mage_preview")
    FUNCTION = "capture"
    CATEGORY = "VOIR/Mage"

    def capture(self, reservoir, image, prompt, seed, max_size, layers, projection_channels):
        import numpy as np
        from PIL import Image

        layer_tuple = tuple(int(x.strip()) for x in layers.split(",") if x.strip())
        reservoir.config = CaptureConfig(layers=layer_tuple, projection_channels=projection_channels)
        arr = (image[0].detach().cpu().clamp(0, 1).numpy() * 255).astype("uint8")
        state, preview = reservoir.capture(Image.fromarray(arr), prompt, seed, max_size)
        out = torch.from_numpy(np.asarray(preview).copy()).float().div(255.0).unsqueeze(0)
        return state, out


class VoirAlbedoReadoutLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "checkpoint": ("STRING", {"default": ""}),
            "device": (["cuda", "cpu"], {"default": "cuda"}),
        }}
    RETURN_TYPES = ("VOIR_ALBEDO_READOUT",)
    FUNCTION = "load"
    CATEGORY = "VOIR/Albedo"

    def load(self, checkpoint, device):
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        return (AlbedoReadout.from_checkpoint(payload, device).eval(),)


class VoirAlbedoReadoutApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "readout": ("VOIR_ALBEDO_READOUT",),
            "state": ("VOIR_RESERVOIR_STATE",),
        }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "VOIR/Albedo"

    def apply(self, readout, state: ReservoirState):
        device = next(readout.parameters()).device
        out = readout.predict_state(state, device)[0].permute(1, 2, 0).cpu().unsqueeze(0)
        return (out,)
