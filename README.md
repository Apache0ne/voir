# VOIR — standalone Mage reservoir sampling for albedo

VOIR freezes **Microsoft Mage-Flow-Edit-Turbo**, runs its edit trajectory with a standalone **Beta scheduler + DPM++ SDE device-noise sampler**, preserves selected internal transformer states, and trains only a compact albedo readout.

**ComfyUI is not required or used by the pipeline.** The relevant scheduler, flow parameterization, Brownian noise, and DPM++ SDE equations are implemented directly in `voir/schedule.py` and `voir/sampling.py`.

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
Detached states [model evaluation, layer, channel, H, W]
          │
          ▼
Trainable AlbedoReadout only
          │
          ▼
Diffuse RGB albedo
```

For four scheduled steps, DPM++ SDE performs **seven Mage transformer evaluations**. VOIR preserves all seven, rather than pretending the trajectory has only four internal states.

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

The default standalone schedule is:

```text
[1.0000000, 0.9371303, 0.7925298, 0.4682927, 0.0000000]
```

### How Beta is reproduced

VOIR does not apply a continuous beta curve directly to four points. It reproduces the source ordering:

1. Build the 1,000-entry discrete-flow sigma table with `shift*t / (1 + (shift-1)*t)`.
2. Evaluate beta-distribution quantiles.
3. Multiply by `999`, round to model-table indices, and remove consecutive duplicate indices.
4. Select those exact model sigmas.
5. Append terminal zero.
6. Apply the requested min/max rescale from `1.0` to `0.0`.

### How DPM++ SDE is adapted to Mage

Mage predicts rectified-flow velocity. DPM++ expects a denoised estimate, so every model call is converted with the discrete-flow/CONST identity:

```text
x0 = x_sigma - sigma * velocity
```

The solver uses the CONST half-log-SNR mapping:

```text
lambda = log((1 - sigma) / sigma)
sigma(lambda) = 1 / (1 + exp(lambda))
```

As in the source implementation, an initial sigma of exactly `1` is offset using `percent_to_sigma(1e-4)` before log-SNR evaluation. `dpmpp_sde_gpu` means the Brownian tree is generated on the latent device (`cpu=False`). The same function accepts CPU tensors for deterministic validation here and uses CUDA-resident noise when the latent is on CUDA.

When `torchsde` is installed, VOIR uses its Brownian tree. A built-in path-consistent Brownian-bridge implementation is available as a dependency-free fallback; it is mathematically equivalent but not bitwise identical to the `torchsde` random stream.

## CPU validation

The complete scheduler, DPM++ SDE solver, Brownian process, state format, frozen toy reservoir, readout, and training path run on CPU:

```bash
python -m pip install -e .[dev]
pytest
python scripts/cpu_sampler_smoke.py
```

The full 4B Mage checkpoint is not practical in this CPU environment. Actual Mage capture is the same code path on CUDA.

## Mage CUDA setup

```bash
git clone https://github.com/microsoft/Mage.git
python -m pip install -e ./Mage
python -m pip install -e .[mage]
```

Capture one albedo reservoir state:

```bash
voir capture-mage input.png states/input.pt \
  --preview-output outputs/mage_albedo_prompt.png \
  --device cuda --max-size 512 \
  --steps 4 --alpha 0.60 --beta 0.80 \
  --sigma-start 1.0 --sigma-end 0.0 \
  --eta 1.0 --s-noise 1.0 --r 0.5 \
  --layers 0,12,23 --projection-channels 64
```

The default instruction is:

```text
remove illumination, shadows, highlights, and reflections; output diffuse albedo only
```

Each saved state records:

- the five scheduled sigmas;
- every actual model-evaluation sigma;
- sampler name and DPM++ parameters;
- Brownian backend;
- selected transformer layers;
- fixed projection width and seed.

## Train only the readout

Manifest line:

```json
{"state":"states/a.pt","albedo":"targets/a.png"}
```

Training:

```bash
voir train-albedo dataset.jsonl checkpoints/albedo_v1.pt \
  --device cuda --epochs 20 --batch-size 2
```

Inference from cached states works on CPU or CUDA:

```bash
voir predict-albedo states/input.pt checkpoints/albedo_v1.pt outputs/albedo.png --device cpu
```

## Albedo data order

Start with exact synthetic base-color supervision, then add lower-weight real-image pseudo-labels:

1. Rendered objects with known base color and randomized HDRI, key/fill lighting, exposure, roughness, metallic response, and backgrounds.
2. Human, clothing, and armor renders with exact material IDs and diffuse maps.
3. Real photographs labeled by an intrinsic-image or PBR teacher.
4. Hard cases: colored illumination, cast shadows, glossy armor, skin subsurface scattering, reflections, and textured cloth.

Mage remains frozen in stage 1. Only the readout is optimized.
