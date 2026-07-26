"""VOIR: frozen image reservoirs with trainable dense readouts."""

from .readout import AlbedoReadout
from .reservoir import CaptureConfig, MageEditReservoir, ToyImageReservoir
from .schedule import beta_flow_sigmas, rescale_sigmas
from .state import ReservoirState

__all__ = [
    "AlbedoReadout",
    "CaptureConfig",
    "MageEditReservoir",
    "ReservoirState",
    "ToyImageReservoir",
    "beta_flow_sigmas",
    "rescale_sigmas",
]

__version__ = "0.1.0"
