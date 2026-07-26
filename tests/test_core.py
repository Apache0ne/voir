import torch

from voir.readout import AlbedoReadout
from voir.reservoir import ToyImageReservoir
from voir.schedule import beta_flow_sigmas, rescale_sigmas
from voir.state import ReservoirState


def test_rescale_matches_requested_endpoints():
    x = torch.tensor([9.0, 5.0, 1.0, 0.0])
    y = rescale_sigmas(x, 1.0, 0.0)
    assert torch.isclose(y.max(), torch.tensor(1.0))
    assert torch.isclose(y.min(), torch.tensor(0.0))
    assert torch.all(y[:-1] >= y[1:])


def test_beta_flow_defaults_are_monotonic():
    sigmas = beta_flow_sigmas(steps=4, alpha=0.6, beta=0.8)
    assert sigmas.shape == (5,)
    assert torch.isclose(sigmas[0], torch.tensor(1.0))
    assert torch.isclose(sigmas[-1], torch.tensor(0.0))
    assert torch.all(sigmas[:-1] >= sigmas[1:])


def test_cpu_reservoir_is_frozen_and_readout_shapes_match():
    x = torch.rand(1, 3, 32, 48)
    reservoir = ToyImageReservoir(channels=8, steps=3)
    state = reservoir.capture(x)
    features = state.flattened()
    model = AlbedoReadout(features.shape[1], width=16, depth=1)
    out = model(features, state.output_size)
    assert out.shape == (1, 3, 32, 48)
    out.mean().backward()
    assert all(parameter.grad is None for parameter in reservoir.parameters())
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_state_round_trip(tmp_path):
    state = ReservoirState(
        features=torch.randn(4, 3, 8, 6, 5),
        output_size=(96, 80),
        sigmas=torch.linspace(1, 0, 5),
        layer_indices=(0, 12, 23),
        source="test",
    )
    path = tmp_path / "state.pt"
    state.save(path)
    loaded = ReservoirState.load(path)
    assert torch.equal(loaded.features, state.features)
    assert loaded.readout_channels == 4 * 3 * 8
