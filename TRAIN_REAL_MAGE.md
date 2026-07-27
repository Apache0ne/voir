# Train the actual Mage-conditioned albedo model

This path trains a full-resolution decoder from the public cache dataset:

```text
ApacheOne/voir-mage-flow-edit-turbo-cache16
```

Unlike the CPU surrogate benchmark, every input feature was produced by the real
frozen `microsoft/Mage-Flow-Edit-Turbo` checkpoint. The model consumes:

- 2,240 projected transformer-state channels: 7 evaluations × 5 blocks × 64 channels
- 2,688 sampler channels: latent, denoised, and velocity trajectories
- 63 fixed intrinsic-image channels
- full-resolution source RGB

The decoder is under one million trainable parameters and upsamples the 21×32
Mage grid to the paired 336×512 albedo target. Training uses the 12/4 split stored
inside the cache, aligned random crops, horizontal/vertical flips, masked real-photo
albedo loss, BF16 on CUDA, EMA checkpoint selection, and held-out validation.

## Colab / CUDA

```bash
python -m pip install -e .
python -m pip install -U huggingface_hub hf_xet
python scripts/train_hf_real_mage.py \
  --hf-dataset ApacheOne/voir-mage-flow-edit-turbo-cache16 \
  --dataset-dir /content/voir_mage_cache16 \
  --output-dir /content/voir_real_mage_albedo_v1 \
  --epochs 50 --batch-size 2 --repeats 4 \
  --patch-grid 16,24 --width 64 --depth 6
```

To upload the checkpoint and validation comparisons after training:

```bash
python scripts/train_hf_real_mage.py \
  --upload-repo ApacheOne/voir-real-mage-albedo-v1
```

`HF_TOKEN` must be present in the environment for upload.

## Cached-state inference

```bash
python scripts/predict_real_mage_cache.py \
  /path/to/mage_cache/real16/states/012.pt \
  /path/to/voir_real_mage_albedo_v1.pt \
  prediction.png \
  --root /path/to/mage_cache/real16 \
  --comparison comparison.png
```

## Scope

This is a real Mage-conditioned model, not a synthetic surrogate. It is still an
initial narrow-domain checkpoint because only twelve real aerial pairs are used for
optimization. The held-out four-image split measures memorization-resistant progress
within that domain; broad people, clothing, armor, indoor, and general-photo behavior
requires substantially more paired caches.
