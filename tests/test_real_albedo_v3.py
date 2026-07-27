import torch

from voir.albedo_features import AUXILIARY_ALBEDO_CHANNELS, fixed_albedo_features
from voir.losses import masked_albedo_loss
from voir.readout import AlbedoReadout
from voir.reservoir import ToyImageReservoir


def _features(image: torch.Tensor, reservoir: ToyImageReservoir) -> torch.Tensor:
    base = reservoir.input_conv(image)
    state = torch.zeros_like(base)
    states = []
    for leak in torch.linspace(0.35, 0.80, reservoir.steps).tolist():
        state = torch.tanh(
            base
            + float(leak) * reservoir.recurrent_conv(state)
            + 0.15 * reservoir.mix_conv(state)
        )
        states.append(state)
    return torch.cat([torch.cat(states, dim=1), fixed_albedo_features(image)], dim=1)


def test_intrinsic_v3_shape_gradients_and_checkpoint():
    torch.manual_seed(7)
    image = torch.rand(2, 3, 32, 40)
    reservoir = ToyImageReservoir(channels=8, steps=4, seed=11)
    features = _features(image, reservoir)
    trajectory_channels = reservoir.channels * reservoir.steps
    model = AlbedoReadout(
        in_channels=features.shape[1],
        width=16,
        depth=3,
        architecture="intrinsic_v3",
        trajectory_channels=trajectory_channels,
        auxiliary_channels=AUXILIARY_ALBEDO_CHANNELS,
    )
    prediction = model(features)
    assert prediction.shape == image.shape
    assert torch.isfinite(prediction).all()
    prediction.mean().backward()
    assert all(parameter.grad is None for parameter in reservoir.parameters())
    assert any(parameter.grad is not None for parameter in model.parameters())

    payload = model.checkpoint()
    restored = AlbedoReadout.from_checkpoint(payload)
    with torch.no_grad():
        assert torch.equal(model(features), restored(features))


def test_masked_loss_ignores_invalid_region():
    target = torch.zeros(1, 3, 16, 16)
    prediction = target.clone()
    prediction[..., :8, :] = 1.0
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., 8:, :] = 1.0
    loss, parts = masked_albedo_loss(prediction, target, mask)
    assert float(loss) < 1e-3
    assert parts["rgb"] < 1e-3
