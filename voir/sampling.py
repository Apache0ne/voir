from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from .schedule import flow_percent_to_sigma


class DenoisedModel(Protocol):
    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class SamplerTrace:
    schedule_sigmas: torch.Tensor
    model_eval_sigmas: torch.Tensor
    sampler: str
    noise_backend: str
    eta: float
    s_noise: float
    r: float


def _sort_times(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    if float(a) < float(b):
        return a, b, 1
    return b, a, -1


class _NativeBrownianBridge:
    """Query-order deterministic Brownian bridge used when torchsde is unavailable.

    It provides path-consistent Brownian increments and can keep all randomness on
    the latent device. The stochastic process is mathematically equivalent to a
    Brownian tree, although its exact random stream is not bitwise-identical to
    torchsde.BrownianTree.
    """

    def __init__(self, x: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor, seed: int | None, cpu: bool):
        self.output_device = x.device
        self.output_dtype = x.dtype
        self.tree_device = torch.device("cpu") if cpu else x.device
        self.tree_dtype = torch.float32
        self.shape = tuple(x.shape)
        self.generator = torch.Generator(device=self.tree_device)
        if seed is None:
            seed = int(torch.randint(0, 2**63 - 1, ()).item())
        self.generator.manual_seed(int(seed))

        lo, hi, _ = _sort_times(t0.detach().float(), t1.detach().float())
        self.lo = self._key(float(lo))
        self.hi = self._key(float(hi))
        if not self.hi > self.lo:
            raise ValueError("Brownian interval must have nonzero length")
        self._values: dict[float, torch.Tensor] = {
            self.lo: torch.zeros(self.shape, device=self.tree_device, dtype=self.tree_dtype),
        }
        end_noise = torch.randn(
            self.shape,
            generator=self.generator,
            device=self.tree_device,
            dtype=self.tree_dtype,
        )
        self._values[self.hi] = end_noise * (self.hi - self.lo) ** 0.5

    @staticmethod
    def _key(t: float) -> float:
        return round(float(t), 12)

    def value(self, t: torch.Tensor | float) -> torch.Tensor:
        key = self._key(float(torch.as_tensor(t)))
        if key in self._values:
            return self._values[key]
        if not (self.lo < key < self.hi):
            raise ValueError(f"Brownian query {key} outside [{self.lo}, {self.hi}]")

        points = sorted(self._values)
        right_index = next(i for i, point in enumerate(points) if point > key)
        left, right = points[right_index - 1], points[right_index]
        w_left, w_right = self._values[left], self._values[right]
        span = right - left
        left_weight = (right - key) / span
        right_weight = (key - left) / span
        mean = left_weight * w_left + right_weight * w_right
        variance = (key - left) * (right - key) / span
        noise = torch.randn(
            self.shape,
            generator=self.generator,
            device=self.tree_device,
            dtype=self.tree_dtype,
        )
        value = mean + noise * variance**0.5
        self._values[key] = value
        return value

    def increment(self, t0: torch.Tensor | float, t1: torch.Tensor | float) -> torch.Tensor:
        a = torch.as_tensor(t0, dtype=torch.float32)
        b = torch.as_tensor(t1, dtype=torch.float32)
        lo, hi, sign = _sort_times(a, b)
        delta = (self.value(hi) - self.value(lo)) * sign
        return delta.to(device=self.output_device, dtype=self.output_dtype)


class _TorchSDEBrownian:
    def __init__(self, x: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor, seed: int | None, cpu: bool):
        import torchsde  # type: ignore

        self.cpu_tree = bool(cpu)
        t0, t1, self.sign = _sort_times(t0, t1)
        w0 = torch.zeros_like(x)
        if seed is None:
            seed = int(torch.randint(0, 2**63 - 1, ()).item())
        if self.cpu_tree:
            t0, w0, t1 = t0.detach().cpu(), w0.detach().cpu(), t1.detach().cpu()
        self.tree = torchsde.BrownianTree(t0, w0, t1, entropy=int(seed))

    def increment(self, t0: torch.Tensor | float, t1: torch.Tensor | float) -> torch.Tensor:
        a = torch.as_tensor(t0)
        b = torch.as_tensor(t1)
        lo, hi, sign = _sort_times(a, b)
        device, dtype = a.device, a.dtype
        if self.cpu_tree:
            lo, hi = lo.detach().cpu().float(), hi.detach().cpu().float()
        return self.tree(lo, hi).to(device=device, dtype=dtype) * (self.sign * sign)


class BrownianTreeNoiseSampler:
    """Standalone equivalent of Comfy's BrownianTreeNoiseSampler."""

    def __init__(
        self,
        x: torch.Tensor,
        sigma_min: torch.Tensor,
        sigma_max: torch.Tensor,
        seed: int | None = None,
        *,
        cpu: bool = False,
        prefer_torchsde: bool = True,
    ):
        self.output_device = x.device
        self.output_dtype = x.dtype
        self.backend = "native_bridge"
        tree = None
        if prefer_torchsde:
            try:
                tree = _TorchSDEBrownian(x, sigma_min, sigma_max, seed, cpu)
                self.backend = "torchsde_cpu" if cpu else "torchsde_device"
            except ImportError:
                tree = None
        if tree is None:
            tree = _NativeBrownianBridge(x, sigma_min, sigma_max, seed, cpu)
            self.backend = "native_bridge_cpu" if cpu else "native_bridge_device"
        self.tree = tree

    def __call__(self, sigma: torch.Tensor, sigma_next: torch.Tensor) -> torch.Tensor:
        sigma = torch.as_tensor(sigma, device=self.output_device, dtype=torch.float32)
        sigma_next = torch.as_tensor(sigma_next, device=self.output_device, dtype=torch.float32)
        dt = (sigma_next - sigma).abs()
        if float(dt) == 0.0:
            return torch.zeros((), device=self.output_device, dtype=self.output_dtype)
        increment = self.tree.increment(sigma, sigma_next)
        return increment.to(device=self.output_device, dtype=self.output_dtype) / dt.sqrt().to(self.output_dtype)


def _get_ancestral_step(sigma_from: torch.Tensor, sigma_to: torch.Tensor, eta: float) -> tuple[torch.Tensor, torch.Tensor]:
    if eta == 0.0:
        return sigma_to, torch.zeros_like(sigma_to)
    sigma_up = eta * torch.sqrt(
        torch.clamp(sigma_to.square() * (sigma_from.square() - sigma_to.square()) / sigma_from.square(), min=0.0)
    )
    sigma_up = torch.minimum(sigma_to, sigma_up)
    sigma_down = torch.sqrt(torch.clamp(sigma_to.square() - sigma_up.square(), min=0.0))
    return sigma_down, sigma_up


def flow_sigma_to_half_log_snr(sigma: torch.Tensor) -> torch.Tensor:
    """CONST/discrete-flow log(alpha/sigma) with alpha=1-sigma."""
    eps = torch.finfo(sigma.dtype).eps
    sigma = sigma.clamp(eps, 1.0 - eps)
    return -torch.logit(sigma)


def flow_half_log_snr_to_sigma(half_log_snr: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(-half_log_snr)


def offset_first_sigma_for_flow(sigmas: torch.Tensor, shift: float, percent_offset: float = 1e-4) -> torch.Tensor:
    if sigmas.numel() <= 1 or float(sigmas[0]) < 1.0:
        return sigmas
    out = sigmas.clone()
    out[0] = flow_percent_to_sigma(percent_offset, shift=shift).to(device=out.device, dtype=out.dtype)
    return out


@torch.no_grad()
def sample_dpmpp_sde_gpu(
    model: DenoisedModel,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    seed: int = 0,
    eta: float = 1.0,
    s_noise: float = 1.0,
    r: float = 0.5,
    shift: float = 6.0,
    callback: Callable[[dict], None] | None = None,
    prefer_torchsde: bool = True,
) -> tuple[torch.Tensor, SamplerTrace]:
    """Standalone DPM-Solver++ SDE for Mage's CONST rectified-flow parameterization.

    The ``_gpu`` name follows Comfy: Brownian noise is generated on the latent
    device (``cpu=False``), not that the function refuses CPU tensors. This makes
    the exact same code testable on CPU and device-resident on CUDA.
    """
    if sigmas.ndim != 1:
        raise ValueError("sigmas must be one-dimensional")
    if sigmas.numel() <= 1:
        trace = SamplerTrace(sigmas.detach().cpu(), torch.empty(0), "dpmpp_sde_gpu", "none", eta, s_noise, r)
        return x, trace
    if not (0.0 < r <= 1.0):
        raise ValueError("r must be in (0, 1]")
    if torch.any(sigmas[:-1] < sigmas[1:]):
        raise ValueError("sigmas must be descending")
    positive = sigmas[sigmas > 0]
    if positive.numel() == 0:
        raise ValueError("schedule requires at least one positive sigma")

    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    noise_sampler = BrownianTreeNoiseSampler(
        x,
        positive.min().to(x.device),
        positive.max().to(x.device),
        seed=seed,
        cpu=False,
        prefer_torchsde=prefer_torchsde,
    )
    internal_sigmas = offset_first_sigma_for_flow(sigmas, shift=shift)
    s_in = x.new_ones([x.shape[0]])
    eval_sigmas: list[float] = []

    for i in range(len(internal_sigmas) - 1):
        sigma_i = internal_sigmas[i]
        sigma_next = internal_sigmas[i + 1]
        eval_sigmas.append(float(sigma_i))
        denoised = model(x, sigma_i * s_in)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma_i, "denoised": denoised, "stage": 1})
        if float(sigma_next) == 0.0:
            x = denoised
            continue

        lambda_s = flow_sigma_to_half_log_snr(sigma_i)
        lambda_t = flow_sigma_to_half_log_snr(sigma_next)
        h = lambda_t - lambda_s
        lambda_s_1 = lambda_s + r * h
        fac = 1.0 / (2.0 * r)
        sigma_s_1 = flow_half_log_snr_to_sigma(lambda_s_1)

        alpha_s = sigma_i * lambda_s.exp()
        alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
        alpha_t = sigma_next * lambda_t.exp()

        # DPM++ SDE stage 1, matching Comfy's sample_dpmpp_sde.
        sigma_from = torch.exp(-lambda_s)
        sigma_to_mid = torch.exp(-lambda_s_1)
        sd, su = _get_ancestral_step(sigma_from, sigma_to_mid, eta)
        lambda_s_1_down = -sd.log()
        h_down = lambda_s_1_down - lambda_s
        x_2 = (alpha_s_1 / alpha_s) * torch.exp(-h_down) * x - alpha_s_1 * torch.expm1(-h_down) * denoised
        if eta > 0.0 and s_noise > 0.0:
            x_2 = x_2 + alpha_s_1 * noise_sampler(sigma_i, sigma_s_1) * s_noise * su

        eval_sigmas.append(float(sigma_s_1))
        denoised_2 = model(x_2, sigma_s_1 * s_in)
        if callback is not None:
            callback({"x": x_2, "i": i, "sigma": sigma_s_1, "denoised": denoised_2, "stage": 2})

        # DPM++ SDE stage 2.
        sigma_to = torch.exp(-lambda_t)
        sd, su = _get_ancestral_step(sigma_from, sigma_to, eta)
        lambda_t_down = -sd.log()
        h_down = lambda_t_down - lambda_s
        denoised_d = (1.0 - fac) * denoised + fac * denoised_2
        x = (alpha_t / alpha_s) * torch.exp(-h_down) * x - alpha_t * torch.expm1(-h_down) * denoised_d
        if eta > 0.0 and s_noise > 0.0:
            x = x + alpha_t * noise_sampler(sigma_i, sigma_next) * s_noise * su

    trace = SamplerTrace(
        schedule_sigmas=sigmas.detach().cpu(),
        model_eval_sigmas=torch.tensor(eval_sigmas, dtype=torch.float32),
        sampler="dpmpp_sde_gpu",
        noise_backend=noise_sampler.backend,
        eta=float(eta),
        s_noise=float(s_noise),
        r=float(r),
    )
    return x, trace
