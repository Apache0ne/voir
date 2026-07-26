from .nodes import (
    VoirAlbedoReadoutApply,
    VoirAlbedoReadoutLoader,
    VoirBetaSamplingScheduler,
    VoirKSamplerSelect,
    VoirMageCaptureAlbedoStates,
    VoirMageFlowEditTurboLoader,
    VoirMageTurboSamplingPreset,
    VoirSigmasRescale,
)

NODE_CLASS_MAPPINGS = {
    "VoirBetaSamplingScheduler": VoirBetaSamplingScheduler,
    "VoirSigmasRescale": VoirSigmasRescale,
    "VoirKSamplerSelect": VoirKSamplerSelect,
    "VoirMageTurboSamplingPreset": VoirMageTurboSamplingPreset,
    "VoirMageFlowEditTurboLoader": VoirMageFlowEditTurboLoader,
    "VoirMageCaptureAlbedoStates": VoirMageCaptureAlbedoStates,
    "VoirAlbedoReadoutLoader": VoirAlbedoReadoutLoader,
    "VoirAlbedoReadoutApply": VoirAlbedoReadoutApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoirBetaSamplingScheduler": "VOIR Beta Sampling Scheduler",
    "VoirSigmasRescale": "VOIR Sigmas Rescale",
    "VoirKSamplerSelect": "VOIR KSampler Select",
    "VoirMageTurboSamplingPreset": "VOIR Mage Turbo Sampling Preset",
    "VoirMageFlowEditTurboLoader": "VOIR Mage Flow Edit Turbo Loader",
    "VoirMageCaptureAlbedoStates": "VOIR Mage Capture Albedo States",
    "VoirAlbedoReadoutLoader": "VOIR Albedo Readout Loader",
    "VoirAlbedoReadoutApply": "VOIR Albedo Readout Apply",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
