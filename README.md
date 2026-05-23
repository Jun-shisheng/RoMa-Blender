# Low-Confidence Region Optimization in Feature Matching Using Blender and RoMa

A comprehensive framework integrating **Blender synthetic scene generation**, **RoMa feature matching optimization**, and **low-confidence region detection & visualization**.

---

## Overview

Feature matching is a cornerstone of computer vision, but low-confidence matches in complex regions—object edges, shadow boundaries, textureless surfaces—remain a critical challenge. This project presents a three-part framework:

1. **Blender-Based Scene Generator** — Automated pipeline for generating large-scale visual sequences with controlled variations in illumination, camera viewpoint, and object layout.
2. **Encoder-Optimized Matching Model** — A modified RoMa architecture replacing the original feature encoder with a VGG19-BN backbone to enhance local feature extraction in low-texture regions.
3. **Low-Confidence Evaluation Toolkit** — Quantitative metrics and 3D visualization tools for identifying, quantifying, and analyzing unreliable correspondence regions.

Experiments on 47 synthetic vehicle-view scenes demonstrate significant improvements in low-confidence region detection accuracy.

---

## Repository Structure

```
.
├── 01_Blender_Scripts/          # Blender Python scripts for scene generation & visualization
│   ├── scene_generator.py       # Automated 3D scene rendering with controlled variations
│   ├── roma_matcher_blender.py  # RoMa feature matching integration within Blender
│   └── visualize_fail_points.py # 3D visualization of low-confidence regions on mesh surfaces
│
├── 02_Data/                     # Experimental data
│   ├── match_results/           # CSV matching reports
│   └── fail_cases/              # Filtered failure cases (JSON + CSV)
│
├── 03_Python_Scripts/           # Training & evaluation pipeline
│   ├── evaluate_roma.py         # Quantitative evaluation comparing baseline vs fine-tuned
│   ├── train_roma.py            # VGG19-BN + FeatureCycleLoss self-supervised training
│   ├── filter_fail_cases.py     # Failure case filtering
│   ├── generate_dataset_from_raw.py  # Training dataset construction
│   └── view_csv_report.py       # CSV matching report viewer & statistics
│
├── 04_TrainingData/             # Training artifacts
│   ├── roma_self_supervised_dataset/  # Masks and metadata
│   └── evaluation_report.csv    # Baseline vs fine-tuned comparison results
│
└── romatch/                     # Original RoMa library (forked from Parskatt/RoMa)
```

> Note: Large binary assets (rendered images, model checkpoints, Blender .blend files) are excluded via `.gitignore`. They can be regenerated using the provided scripts.

---

## Pipeline

```
Blender Scene Generator          RoMa Feature Matching          Evaluation & Visualization
┌─────────────────────┐     ┌──────────────────────┐     ┌──────────────────────────┐
│ 3D Vehicle Model    │     │ Regression Matcher   │     │ Low-Confidence Mask Gen  │
│ Controlled Camera   │────▶│ VGG19-BN Encoder     │────▶│ Failure Case Filtering   │
│ Illumination Vars   │     │ FeatureCycleLoss     │     │ 3D Surface Projection    │
└─────────────────────┘     └──────────────────────┘     └──────────────────────────┘
```

---

## Usage

### Requirements

- Blender 3.3.1+
- Python 3.9+
- PyTorch 1.12.1+
- CUDA 11.6+
- OpenCV 4.6.0+

### Setup

```bash
git clone https://github.com/Jun-shisheng/RoMa.git
cd RoMa
pip install -e .
```

### Scene Generation

```bash
blender --background --python 01_Blender_Scripts/scene_generator.py
```

### Feature Matching & Analysis

```bash
python 01_Blender_Scripts/roma_matcher_blender.py
```

### Training the Optimized Model

```bash
python 03_Python_Scripts/train_roma.py
```

### Evaluation

```bash
python 03_Python_Scripts/evaluate_roma.py
```

---

## Built Upon

This project builds on [RoMa](https://github.com/Parskatt/RoMa) (CVPR 2024) by Edstedt et al.

- Original RoMa code (except DINOv2): MIT License
- DINOv2: Apache 2.0 License
- Added research code: MIT License
