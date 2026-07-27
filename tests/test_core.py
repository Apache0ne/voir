import torch

from voir.albedo_features import AUXILIARY_ALBEDO_CHANNELS, fixed_albedo_features
from voir.readout import AlbedoReadout
from voir.reservoir import ToyImageReservoir
from voir.state import ReservoirState


def test_fixed_albedo_feature_bank_shape_and_finiteness():
    image = torch.rand(2, 3, 32, 48)
    features = fixed_albedo_features(image)
    assert features.shape == (2, AUXILIARY_ALBEDO_CHANNELS, 32, 48)
    assert torch.isfinite(features).all()


def test_cpu_reservoir_is_frozen_and_readout_shapes_match():
    x = torch.rand(1, 3, 32, 48)
    reservoir = ToyImageReservoir(channels=8, steps=3)
    state = reservoir.capture(x)
    features = state.flattened()
    assert state.auxiliary_channels == AUXILIARY_ALBEDO_CHANNELS
    assert state.trajectory_channels == 3 * 8
    assert state.readout_channels == AUXILIARY_ALBEDO_CHANNELS + 3 * 8
    model = AlbedoReadout(features.shape[1], width=16, depth=3)
    out = model(features, state.output_size)
    assert out.shape == (1, 3, 32, 48)
    out.mean().backward()
    assert all(parameter.grad is None for parameter in reservoir.parameters())
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_state_round_trip_preserves_auxiliary_first_order(tmp_path):
    state = ReservoirState(
        features=torch.zeros(4, 3, 8, 6, 5),
        output_size=(96, 80),
        sigmas=torch.linspace(1, 0, 4),
        layer_indices=(0, 12, 23),
        source="test",
        aux_features=torch.ones(7, 6, 5),
    )
    path = tmp_path / "state.pt"
    state.save(path)
    loaded = ReservoirState.load(path)
    flat = loaded.flattened()
    assert torch.equal(loaded.features, state.features)
    assert torch.equal(loaded.aux_features, state.aux_features)
    assert loaded.readout_channels == 7 + 4 * 3 * 8
    assert torch.equal(flat[:, :7], torch.ones(1, 7, 6, 5))
    assert torch.equal(flat[:, 7:], torch.zeros(1, 4 * 3 * 8, 6, 5))


def test_v2_and_legacy_checkpoint_round_trip():
    x = torch.rand(1, 11, 8, 8)
    for architecture, width, depth in (("dilated_v2", 16, 3), ("legacy", 16, 1)):
        model = AlbedoReadout(11, width=width, depth=depth, architecture=architecture)
        payload = model.checkpoint()
        loaded = AlbedoReadout.from_checkpoint(payload)
        assert loaded.architecture == architecture
        assert torch.equal(model(x), loaded(x))
