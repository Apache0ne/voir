import torch

from voir.readout import AlbedoReadout
from voir.transfer import transfer_intrinsic_readout


def test_transfer_preserves_auxiliary_only_function():
    torch.manual_seed(13)
    old_trajectory = 12
    new_trajectory = 37
    auxiliary = 63
    source = AlbedoReadout(
        in_channels=old_trajectory + auxiliary,
        width=16,
        depth=3,
        architecture="intrinsic_v3",
        trajectory_channels=old_trajectory,
        auxiliary_channels=auxiliary,
    )
    source.eval()
    aux = torch.rand(2, auxiliary, 24, 20)
    old_features = torch.cat([torch.zeros(2, old_trajectory, 24, 20), aux], dim=1)
    with torch.no_grad():
        expected = source(old_features)

    transferred, report = transfer_intrinsic_readout(
        source.checkpoint(),
        trajectory_channels=new_trajectory,
        auxiliary_channels=auxiliary,
        freeze_shared=True,
    )
    new_features = torch.cat([torch.randn(2, new_trajectory, 24, 20), aux], dim=1)
    transferred.eval()
    with torch.no_grad():
        actual = transferred(new_features)
    assert torch.equal(expected, actual)
    assert report["old_trajectory_channels"] == old_trajectory
    assert report["new_trajectory_channels"] == new_trajectory
    assert all(
        parameter.requires_grad == name.startswith("trajectory_proj")
        for name, parameter in transferred.named_parameters()
    )
