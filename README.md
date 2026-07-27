# VOIR — standalone Mage reservoir sampling for albedo

VOIR freezes **Microsoft Mage-Flow-Edit-Turbo**, runs its edit trajectory with a standalone **Beta scheduler + DPM++ SDE device-noise sampler**, preserves selected internal transformer states, and trains only a compact albedo readout.

**ComfyUI is not required or used.** Scheduler construction, rectified-flow conversion, Brownian noise, DPM++ SDE, state capture, and readout training live directly in this repository.

## Pipeline

```text
RGB image + albedo instruction
          │
          ▼
Frozen Mage VAE + text encoder + transformer
          │
          ▼
Beta schedule: 4 steps, alpha 0.60, beta 0.80
          │
          ▼
DPM++ SDE, device-resident Brownian noise
          │
          ├── stage-1 transformer evaluation
          ├── midpoint transformer evaluation
          └── terminal denoised evaluation
          │
          ▼
Detached Mage states [evaluation, layer, channel, H, W]
+ fixed 63-channel source-image feature bank
          │
          ▼
80,163-parameter dilated global-context readout
          │
          ▼
Diffuse RGB albedo
```

For four scheduled steps, DPM++ SDE performs **seven Mage transformer evaluations**. VOIR preserves all seven. Source RGB is transformed into a fixed, non-trainable 63-channel bank containing chromaticity, nonlinear color transforms, multiscale illumination estimates, Retinex-style ratios, and gradients. Only the final readout is optimized.

## Exact sampling defaults

```text
sampler       = dpmpp_sde_gpu
steps         = 4
beta alpha    = 0.60
beta beta     = 0.80
sigma start   = 1.00
sigma end     = 0.00
flow shift    = 6.00
eta           = 1.00
s_noise       = 1.00
r             = 0.50
```

Default schedule:

```text
[1.0000000, 0.9371303, 0.7925298, 0.4682927, 0.0000000]
```

### Beta construction

VOIR reproduces the source ordering:

1. Build the 1,000-entry discrete-flow table with `shift*t / (1 + (shift-1)*t)`.
2. Evaluate beta-distribution quantiles.
3. Multiply by `999`, round to table indices, and remove consecutive duplicates.
4. Select the exact model sigmas.
5. Append terminal zero.
6. Rescale the complete schedule from `1.0` to `0.0`.

### DPM++ SDE conversion for Mage

Mage predicts rectified-flow velocity. DPM++ expects a denoised estimate:

```text
x0 = x_sigma - sigma * velocity
```

The discrete-flow/CONST half-log-SNR mapping is:

```text
lambda = log((1 - sigma) / sigma)
sigma(lambda) = 1 / (1 + exp(lambda))
```

An initial sigma of exactly `1` is offset with `percent_to_sigma(1e-4)` before log-SNR evaluation. `dpmpp_sde_gpu` means Brownian noise is generated on the latent device (`cpu=False`). The same implementation accepts CPU tensors for validation and CUDA tensors for actual Mage inference.

When `torchsde` is installed, VOIR uses its Brownian tree. The built-in path-consistent Brownian bridge is the dependency-free fallback; it is mathematically equivalent but not bitwise identical to the `torchsde` stream.

## Best measured CPU result

The readout was trained in three cached-data phases followed by four low-rate online phases. Each online batch was newly generated, and checkpoints were selected using two disjoint validation seeds. Final evaluation used three seeds with 96 images each, including untouched seed `4040`.

```text
training pool:          512 cached + 7,680 online samples
held-out evaluation:    288 images across 3 seeds
resolution:             48x48
fixed input channels:   127
trainable parameters:   80,163

three-seed average
MAE:                    0.028975
MSE:                    0.002378
PSNR:                   26.243 dB
SSIM, local 7x7:         0.925872
SSIM, global:            0.935953

untouched seed 4040
MAE:                    0.029538
MSE:                    0.002510
PSNR:                   26.003 dB
SSIM, local 7x7:         0.925669
SSIM, global:            0.932271
```

Relative to the previous v2 checkpoint, the three-seed average improved by:

```text
MAE:          -3.19%
MSE:          -5.47%
PSNR:         +0.245 dB
local SSIM:   +0.00283
global SSIM:  +0.00360
```

These are **held-out synthetic CPU-surrogate results**, not full-Mage or real-photo results. The benchmark contains sharp material boundaries, spatially varying illumination, colored light, specular highlights, exposure changes, and gamma changes. Exact values are recorded in `benchmarks/cpu_albedo_v2_generalized.json`.

Run the reproducible full benchmark:

```bash
python -m pip install -e .[dev]
python scripts/cpu_albedo_benchmark.py
```

Fast verification:

```bash
pytest
python scripts/cpu_sampler_smoke.py
python scripts/cpu_albedo_benchmark.py --quick
```

The full 4B Mage checkpoint is not practical in this CPU environment. Actual Mage capture uses the same sampler and state format on CUDA.

## Mage CUDA setup

```bash
git clone https://github.com/microsoft/Mage.git
python -m pip install -e ./Mage
python -m pip install -e .[mage]
```

Capture one albedo state:

```bash
voir capture-mage input.png states/input.pt \
  --preview-output outputs/mage_albedo_prompt.png \
  --device cuda --max-size 512 \
  --steps 4 --alpha 0.60 --beta 0.80 \
  --sigma-start 1.0 --sigma-end 0.0 \
  --eta 1.0 --s-noise 1.0 --r 0.5 \
  --layers 0,12,23 --projection-channels 64
```

Default instruction:

```text
remove illumination, shadows, highlights, and reflections; output diffuse albedo only
```

Each state records the scheduled sigmas, every actual model-evaluation sigma, sampler parameters, Brownian backend, selected layers, fixed projection configuration, trajectory-channel count, and auxiliary-channel count.

## Train only the readout

Training manifest:

```json
{"state":"states/a.pt","albedo":"targets/a.png"}
```

Validation manifest uses the same format. Training selects the best validation-MAE checkpoint:

```bash
voir train-albedo train.jsonl checkpoints/albedo_v2.pt \
  --validation-manifest validation.jsonl \
  --device cuda --epochs 20 --batch-size 2 \
  --architecture dilated_v2 --width 32 --depth 7
```

Inference from a cached state works on CPU or CUDA:

```bash
voir predict-albedo states/input.pt checkpoints/albedo_v2.pt outputs/albedo.png --device cpu
```

## Dataset order for real albedo

1. Rendered objects with exact base color and randomized HDRI, key/fill lighting, exposure, roughness, metallic response, and backgrounds.
2. Human, clothing, and armor renders with exact material IDs and diffuse maps.
3. Real photographs labeled by a strong intrinsic-image or PBR teacher at lower loss weight.
4. Hard cases: colored illumination, cast shadows, glossy armor, skin subsurface scattering, reflections, and textured cloth.

Mage remains frozen in stage 1. Only the readout is optimized.
