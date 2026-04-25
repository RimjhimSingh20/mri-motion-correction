# MRI Motion Correction — Deep Learning Pipeline for Ultra-High Field (≥7T) MRI

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## Overview

Motion artifacts are among the most disruptive sources of image degradation in
ultra-high field (≥7T) 3-D submillimeter MRI. Even sub-millimetre rigid-body
displacements during long acquisitions corrupt k-space consistency, producing
ghosting, blurring, and signal drop-out that impair both qualitative assessment
and quantitative analyses.

This repository implements a **full deep-learning pipeline** for motion
estimation and correction in 3-D MRI volumes:

- **Physics-grounded data synthesis** — intra-scan rigid-body motion simulated
  directly in k-space (no image-domain shortcuts)
- **3-D U-Net** trained on paired clean/corrupted volumes with a combined
  L1 + SSIM loss
- **Patch-based training and inference** with Gaussian-weighted sliding-window
  aggregation for whole-volume prediction
- **Quantitative evaluation** with 3-D SSIM, PSNR, and NRMSE
- **Classical baselines** (FSL FLIRT, SimpleITK rigid registration) for
  head-to-head comparison

---

## Architecture

### End-to-End Pipeline

```
Raw NIfTI volumes
       │
       ▼
┌─────────────────────────────┐
│   Motion Simulator          │  k-space rigid-body corruption
│   (data/motion_simulator.py)│  — rotations 0–6°, translations 0–4 mm
│   severity: mild/mod/severe │  — 1–3 motion events per volume
└────────────┬────────────────┘
             │  paired (clean, corrupted)
             ▼
┌─────────────────────────────┐
│   MRIMotionDataset          │  random 64³ patch extraction
│   (data/dataset.py)         │  z-score normalisation per volume
│   + augmentations           │  random flip, small rotation
└────────────┬────────────────┘
             │  [B, 1, 64, 64, 64]
             ▼
┌─────────────────────────────────────────────────────────────┐
│                       3-D U-Net                             │
│                                                             │
│  Input                                                      │
│  [B,1,64³]                                                  │
│     │                                                       │
│     ├─ Enc0: DoubleConv(1→32)   ─────────────────skip0──┐  │
│     │  MaxPool↓2                                         │  │
│     ├─ Enc1: DoubleConv(32→64)  ──────────────skip1──┐  │  │
│     │  MaxPool↓2                                      │  │  │
│     ├─ Enc2: DoubleConv(64→128) ─────────skip2──┐    │  │  │
│     │  MaxPool↓2                                │    │  │  │
│     ├─ Enc3: DoubleConv(128→256)───skip3──┐     │    │  │  │
│     │  MaxPool↓2                          │     │    │  │  │
│     │                                     │     │    │  │  │
│     └─ Bottleneck: DoubleConv(256→512)    │     │    │  │  │
│                          │                │     │    │  │  │
│     ┌────────────────────┘                │     │    │  │  │
│     ├─ Dec3: Up+concat(skip3)→256  ───────┘     │    │  │  │
│     ├─ Dec2: Up+concat(skip2)→128  ─────────────┘    │  │  │
│     ├─ Dec1: Up+concat(skip1)→64   ──────────────────┘  │  │
│     ├─ Dec0: Up+concat(skip0)→32   ─────────────────────┘  │
│     └─ OutConv: 1×1×1 Conv → 1 channel                     │
│                                                             │
│  Each ConvBlock: Conv3d → InstanceNorm3d → LeakyReLU(0.01) │
│  Upsampling: trilinear (no checkerboard)                    │
│  Parameters: ~10 M  |  Patch memory: ~4 GB (AMP, bs=2)     │
└────────────┬────────────────────────────────────────────────┘
             │  [B, 1, 64, 64, 64]  corrected patch
             ▼
┌─────────────────────────────┐
│  Sliding-Window Inference   │  Gaussian-weighted patch blending
│  (evaluation/evaluator.py)  │  overlap=0.5, batch=4 patches
└────────────┬────────────────┘
             │
             ▼
     Corrected NIfTI volume
```

### Loss Function

```
L_total = 1.0 × L1(pred, target)
        + 0.5 × (1 − SSIM3D(pred, target))
```

SSIM is computed with a 3-D Gaussian kernel (σ=1.5, k=11) directly on the
training patches, providing perceptual supervision beyond pixel-level fidelity.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Normalisation | InstanceNorm3d | MRI patches have variable intensity distributions; small batch sizes (2–4) make BatchNorm unstable |
| Activation | LeakyReLU (α=0.01) | Prevents dead neurons in deep 3-D networks |
| Upsampling | Trilinear + 1×1 conv | Avoids checkerboard artifacts produced by ConvTranspose3d |
| Motion simulation | K-space PE-line replacement | Physically accurate model of intra-scan Cartesian acquisition; image-domain warping would not reproduce Gibbs ringing or ghosting |
| Central k-space | Protected (35–65 % of PE lines) | Low-frequency content determines gross image contrast; corrupting only peripheral k-space yields realistic artifact patterns |
| Precision | AMP (fp16 forward, fp32 gradients) | ~2× throughput on A6000; GradScaler prevents underflow |
| LR schedule | Cosine annealing + linear warmup | Stabilises early training; avoids aggressive initial updates on randomly initialised weights |
| Patch inference | Gaussian-weighted blending | Suppresses visible seams at patch boundaries during whole-volume inference |

---

## Dataset

### IXI Brain Development Dataset — T1 Weighted

| Property | Value |
|---|---|
| Source | [brain-development.org/ixi-dataset](http://brain-development.org/ixi-dataset) |
| Modality | T1-weighted (MPRAGE) |
| Subjects | 326 healthy adults |
| Acquisition sites | Guys Hospital, Hammersmith Hospital, IOP London |
| Typical volume shape | 256 × 256 × 150 voxels |
| Voxel size | ~0.94 × 0.94 × 1.2 mm |
| Format | NIfTI (.nii.gz) |

### Simulated Artifact Severity Levels

| Severity | Max rotation | Max translation | Motion events | PE lines corrupted | Mean PSNR |
|---|---|---|---|---|---|
| Mild | 2° | 1 mm | 1 | ~8 % | ~36 dB |
| Moderate | 4° | 2 mm | 2 | ~20 % | ~30.8 dB |
| Severe | 6° | 4 mm | 3 | ~40 % | ~24 dB |

---

## Results

> Training results will be populated here after full-dataset training runs.

### Motion Simulation Baseline (10 IXI-T1 volumes, pre-correction)

| Volume | PSNR mild (dB) | SSIM mild | PSNR mod (dB) | SSIM mod | PSNR severe (dB) | SSIM severe |
|---|---|---|---|---|---|---|
| IXI002-Guys-0828 | — | — | — | — | — | — |

*See [results/metrics_table.md](results/metrics_table.md) for full simulation baselines.*

### Post-Training Benchmarks (placeholder)

| Method | PSNR ↑ (dB) | SSIM ↑ | NRMSE ↓ | Notes |
|---|---|---|---|---|
| No correction (input) | 30.8 | — | — | Moderate severity, 10 vols |
| FSL FLIRT (rigid) | — | — | — | 6-DOF, NormCorr |
| SimpleITK (rigid) | — | — | — | Mattes MI, 200 iter |
| **3-D U-Net (ours)** | — | — | — | 200 epochs, moderate |

---

## Repository Structure

```
mri_motion_correction/
├── configs/
│   └── default.yaml              # All hyperparameters
├── data/
│   ├── dataset.py                # MRIMotionDataset — patch-based NIfTI loader
│   ├── motion_simulator.py       # K-space motion simulation (CLI + library)
│   └── transforms.py             # Augmentation: flip, rotation, k-space motion
├── models/
│   ├── blocks.py                 # ConvBlock, DoubleConv, EncoderBlock, DecoderBlock
│   └── unet3d.py                 # 3-D U-Net
├── losses/
│   └── combined.py               # L1 + SSIM loss
├── metrics/
│   └── image_quality.py          # ssim3d, psnr, nrmse, MetricTracker
├── training/
│   └── trainer.py                # AMP training loop, cosine-warmup LR, checkpointing
├── evaluation/
│   └── evaluator.py              # Sliding-window inference + Evaluator
├── baselines/
│   └── classical.py              # FSL FLIRT and SimpleITK wrappers
├── utils/
│   ├── io.py                     # NIfTI load/save, normalisation
│   └── visualization.py          # Slice comparison, training curves, metric plots
├── scripts/
│   ├── prepare_data.py           # Download IXI-T1, extract, simulate pairs
│   ├── train.py                  # Training entry point
│   ├── evaluate.py               # Test-set evaluation
│   ├── infer.py                  # Single-volume inference
│   └── generate_results.py       # Reproduce charts and tables in results/
├── tests/
│   └── test_metrics.py           # 14 pytest tests for SSIM/PSNR/NRMSE
└── results/
    ├── metrics_table.md           # Simulation metrics (all volumes × severities)
    ├── fig1_psnr_by_severity.png
    ├── fig2_nrmse_by_severity.png
    ├── fig3_sample_slices.png     # Clean vs corrupted vs residual
    ├── fig4_kspace_corruption.png # K-space magnitude comparison
    └── fig5_severity_boxplot.png  # Metric distributions across volumes
```

---

## Setup

### Requirements

- Python ≥ 3.9
- CUDA-capable GPU (≥ 16 GB VRAM recommended for 64³ patches at batch size 2)
- Tested on NVIDIA A6000 / RTX 6000

### Installation

```bash
git clone git@github.com:RimjhimSingh20/mri-motion-correction.git
cd mri-motion-correction
pip install -r requirements.txt
# or install as a package:
pip install -e .
```

### Data Preparation

```bash
# Download IXI-T1 (~2.1 GB), extract 10 volumes, simulate motion pairs
python scripts/prepare_data.py --n-volumes 10 --severity moderate

# Full dataset (326 volumes, all severities)
python scripts/prepare_data.py --n-volumes 326 --severity moderate
python scripts/prepare_data.py --n-volumes 326 --severity mild     --corrupted-dir data/processed/corrupted_mild
python scripts/prepare_data.py --n-volumes 326 --severity severe   --corrupted-dir data/processed/corrupted_severe
```

### Training

```bash
python scripts/train.py --config configs/default.yaml

# Resume from checkpoint
python scripts/train.py --config configs/default.yaml --resume outputs/unet3d_motion_correction/checkpoints/best.pt
```

Training logs and checkpoints are written to `outputs/<experiment_name>/`.
Open TensorBoard with:

```bash
tensorboard --logdir outputs/unet3d_motion_correction/tensorboard
```

### Evaluation

```bash
# Full test-set evaluation with sliding-window inference
python scripts/evaluate.py \
    --checkpoint outputs/unet3d_motion_correction/checkpoints/best.pt \
    --data-dir data/test

# Single-volume inference
python scripts/infer.py \
    --checkpoint outputs/unet3d_motion_correction/checkpoints/best.pt \
    --input path/to/corrupted.nii.gz \
    --output path/to/corrected.nii.gz
```

### Reproduce Results Figures

```bash
python scripts/generate_results.py
# outputs written to results/
```

### Tests

```bash
pytest tests/ -v
```

---

## Configuration

All hyperparameters live in [`configs/default.yaml`](configs/default.yaml).
Key fields:

```yaml
model:
  base_features: 32   # double each encoder level → [32,64,128,256,512]
  depth: 4

training:
  batch_size: 2
  learning_rate: 1.0e-4
  mixed_precision: true
  gradient_accumulation_steps: 4   # effective batch = 8

loss:
  l1_weight: 1.0
  ssim_weight: 0.5
```

---

## Citation

If you use this code or the simulation framework in your research, please cite:

```bibtex
@misc{singh2026mri,
  author       = {Singh, Rimjhim},
  title        = {Deep Learning Pipeline for Motion Correction in Ultra-High Field MRI},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/RimjhimSingh20/mri-motion-correction}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
