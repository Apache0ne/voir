import torch

from voir.mage_sampler import MageSamplerCache
from voir.state import ReservoirState


def test_reservoir_flattened_trajectory_then_auxiliary():
    trajectory = torch.arange(2 * 1 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 1, 3, 2, 2)
    auxiliary = torch.full((4, 2, 2), 99.0)
    state = ReservoirState(
        features=trajectory,
        output_size=(2, 2),
        sigmas=torch.tensor([1.0, 0.5]),
        layer_indices=(0,),
        aux_features=auxiliary,
    ).validate()
    flattened = state.flattened()
    assert flattened.shape == (1, 10, 2, 2)
    assert torch.equal(flattened[:, :6], trajectory.reshape(1, 6, 2, 2))
    assert torch.equal(flattened[:, 6:], auxiliary.unsqueeze(0))


def test_mage_sampler_cache_round_trip_payload():
    evaluations, batch, tokens, channels = 7, 1, 8, 4
    cache = MageSamplerCache(
        reference_tokens=torch.randn(1, tokens, channels).half(),
        reference_ids=torch.zeros(1, tokens, 3),
        reference_shapes=[[(1, 2, 4)]],
        initial_target_tokens=torch.randn(batch, tokens, channels).half(),
        final_target_tokens=torch.randn(batch, tokens, channels).half(),
        target_ids=torch.zeros(1, tokens, 3),
        image_ids=torch.zeros(1, tokens * 2, 3),
        image_cu=torch.tensor([0, tokens * 2], dtype=torch.int32),
        image_shapes=[[(1, 2, 4), (1, 2, 4)]],
        text_tokens=torch.randn(1, 5, 6).half(),
        text_cu=torch.tensor([0, 5], dtype=torch.int32),
        text_mask=torch.ones(1, 5),
        text_vector=torch.randn(1, 6).half(),
        text_lengths=[5],
        schedule_sigmas=torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0]),
        model_eval_sigmas=torch.linspace(1.0, 0.2, evaluations),
        eval_latents=torch.randn(evaluations, batch, tokens, channels).half(),
        eval_denoised=torch.randn(evaluations, batch, tokens, channels).half(),
        eval_velocity=torch.randn(evaluations, batch, tokens, channels).half(),
        eval_solver_indices=torch.tensor([0, 0, 1, 1, 2, 2, 3], dtype=torch.int16),
        eval_stages=torch.tensor([1, 2, 1, 2, 1, 2, 1], dtype=torch.int8),
        target_grid=(2, 4),
        output_size=(32, 64),
        packed_length=tokens * 2,
        target_length=tokens,
        metadata={"sampler": "dpmpp_sde_gpu"},
    ).validate()
    payload = cache.to_payload()
    assert payload["eval_latents"].shape[0] == evaluations
    assert payload["metadata"]["sampler"] == "dpmpp_sde_gpu"
