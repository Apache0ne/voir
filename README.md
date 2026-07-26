# VOIR — Mage reservoir computing for albedo

VOIR freezes **Microsoft Mage-Flow-Edit-Turbo** as a high-dimensional nonlinear image reservoir, captures selected internal transformer states across a four-step flow trajectory, and trains only a compact dense readout. The first task is diffuse **albedo** prediction.

## Current architecture

```text
RGB image + albedo instruction
          │
          ▼
Frozen Mage VAE + text encoder + Mage-Flow-Edit-Turbo transformer
          │
          ├── beta-spaced flow state 1: selected hidden layers
          ├── beta-spaced flow state 2: selected hidden layers
          ├── beta-spaced flow state 3: selected hidden layers
          └── beta-spaced flow state 4: selected hidden layers
          │
          ▼
Detached persistent ReservoirState [step, layer, channel, H, W]
          │
          ▼
Trainable AlbedoReadout only
          │
          ▼
Diffuse RGB albedo
```

The Mage backbone is set to `eval()`, every parameter has `requires_grad=False`, and captured features are detached before persistence. CPU and CUDA use the same state format and the same readout implementation.

## Requested sampling defaults

- steps: `4`
- beta scheduler alpha: `0.60`
- beta scheduler beta: `0.80`
- sigma rescale start: `1.00`
- sigma rescale end: `0.00`
- Comfy sampler preset: `dpmpp_sde_gpu`

The ComfyUI nodes reproduce the native Comfy beta scheduler, the RES4LYF min/max rescale, and the requested KSampler selector. The standalone Mage capture path currently uses the model-correct rectified-flow Euler update with the same beta/rescaled trajectory. DPM++ SDE is exposed for Comfy-native model paths; it is not silently substituted into Mage's rectified-flow pipeline.

## CPU proof

The 4B Mage model is not intended for practical CPU inference. The repository includes a frozen nonlinear recurrent `ToyImageReservoir` so all reservoir/state/readout/training logic is testable on CPU:

```bash
python -m pip install -e .[dev]
python scripts/cpu_smoke.py
pytest
```

This writes `outputs/cpu_albedo_smoke.png` and verifies that only the readout receives gradients.

## Mage CUDA setup

```bash
git clone https://github.com/microsoft/Mage.git
python -m pip install -e ./Mage
python -m pip install -e .[mage]
```

Capture one state bundle:

```bash
voir capture-mage input.png states/input.pt \
  --preview-output outputs/mage_albedo_prompt.png \
  --device cuda --max-size 512 \
  --steps 4 --alpha 0.60 --beta 0.80 \
  --layers 0,12,23 --projection-channels 64
```

The default prompt is:

```text
remove illumination, shadows, highlights, and reflections; output diffuse albedo only
```

Create a JSONL manifest with one cached state and exact/teacher albedo target per line:

```json
{"state":"states/a.pt","albedo":"targets/a.png"}
```

Train only the readout:

```bash
voir train-albedo dataset.jsonl checkpoints/albedo_v1.pt \
  --device cuda --epochs 20 --batch-size 2
```

Inference from a cached state can run on CPU or CUDA:

```bash
voir predict-albedo states/input.pt checkpoints/albedo_v1.pt outputs/albedo.png --device cpu
```

## ComfyUI installation

Clone this repository into `ComfyUI/custom_nodes/voir`, install the package in the Comfy Python environment, and restart ComfyUI.

Nodes:

- **VOIR Beta Sampling Scheduler** — defaults to 4 / 0.60 / 0.80.
- **VOIR Sigmas Rescale** — defaults to 1.00 → 0.00.
- **VOIR KSampler Select** — defaults to `dpmpp_sde_gpu`.
- **VOIR Mage Turbo Sampling Preset** — combines all three settings.
- **VOIR Mage Flow Edit Turbo Loader**.
- **VOIR Mage Capture Albedo States**.
- **VOIR Albedo Readout Loader**.
- **VOIR Albedo Readout Apply**.

## Dataset direction for albedo v1

Use exact synthetic supervision first, then mix real-image pseudo-labels:

1. Rendered objects with known base-color/albedo, randomized HDRI, key/fill lights, exposure, roughness, metallic and background.
2. Human/clothing/armor renders with exact material IDs and diffuse maps.
3. Real photographs labeled by a strong intrinsic-image or PBR teacher, with lower loss weight.
4. Hard examples: colored illumination, cast shadows, specular highlights, glossy armor, skin subsurface scattering and textured cloth.

Do not train Mage in stage 1. Establish linear/small-decoder probe quality first; optional LoRA belongs in a later ablation.
