import torch

from voir.sampling import sample_dpmpp_sde_gpu
from voir.schedule import beta_flow_sigmas, beta_scheduler_from_model_sigmas, flow_model_sigmas


def test_beta_schedule_matches_discrete_flow_selection():
    table = flow_model_sigmas(timesteps=1000, shift=6.0)
    direct = beta_scheduler_from_model_sigmas(table, 4, 0.6, 0.8)
    sigmas = beta_flow_sigmas(4, 0.6, 0.8, 1.0, 0.0, 6.0, 1000)
    assert torch.equal(direct, sigmas)
    expected = torch.tensor([1.0, 0.9371, 0.7925, 0.4683, 0.0])
    assert torch.allclose(sigmas, expected, atol=5e-5, rtol=0)


def test_dpmpp_sde_gpu_runs_on_cpu_and_is_deterministic():
    sigmas = beta_flow_sigmas()
    x = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(1))
    target = torch.tanh(torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(2)))
    calls = []

    def model(current, sigma):
        calls.append(float(sigma[0]))
        s = sigma.reshape(-1, 1, 1, 1)
        return target + 0.03 * s * torch.tanh(current)

    out1, trace1 = sample_dpmpp_sde_gpu(model, x.clone(), sigmas, seed=77, prefer_torchsde=False)
    calls_first = list(calls)
    calls.clear()
    out2, trace2 = sample_dpmpp_sde_gpu(model, x.clone(), sigmas, seed=77, prefer_torchsde=False)

    assert torch.equal(out1, out2)
    assert len(calls_first) == 2 * (len(sigmas) - 2) + 1
    assert trace1.sampler == "dpmpp_sde_gpu"
    assert trace1.noise_backend == "native_bridge_device"
    assert torch.equal(trace1.model_eval_sigmas, trace2.model_eval_sigmas)
    assert torch.isfinite(out1).all()


def test_final_denoised_step_is_used():
    sigmas = beta_flow_sigmas()
    x = torch.randn(1, 2, 4, 4)
    target = torch.full_like(x, 0.25)

    def exact_x0(_current, _sigma):
        return target

    out, _ = sample_dpmpp_sde_gpu(exact_x0, x, sigmas, seed=3, prefer_torchsde=False)
    assert torch.equal(out, target)
