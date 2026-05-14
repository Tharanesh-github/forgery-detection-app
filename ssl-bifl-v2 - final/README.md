---
title: SSL-BIFL Forgery Localization V2
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# SSL-BIFL: Self-Supervised Blind Image Forgery Localization

**Student:** V. Tharanesh (Tharanesh Vigneswaran)
**Final Year Project**

## Overview

A self-supervised deep learning system for detecting and localizing image forgeries without requiring labeled training data. The model is trained on synthetically generated pseudo-forgeries and evaluated on the CASIA 2.0 benchmark dataset.

## Pipeline (v2)

1. **Multi-scale spatial inference** — ResNet-18 + U-Net runs at 3 spatial scales (100%, 80%, 60%) and results are weighted-averaged
2. **Frequency domain DFT boost** — High-pass Fourier analysis extracts splice boundary artifacts invisible to spatial models
3. **Ensemble thresholding** — 5-threshold majority vote across 20%-80% of pred_max range
4. **Edge-aware Canny refinement** — Mask boundaries snapped to real image edges for forensic precision
5. **4-level forensic verdict** — AUTHENTIC / INCONCLUSIVE / SUSPICIOUS / FORGED

## Model Architecture

- **Encoder:** ResNet-18 (pretrained on ImageNet)
- **Decoder:** U-Net (segmentation-models-pytorch)
- **Loss:** BCE + Dice combined
- **Input:** 256x256 RGB
- **Training data:** DIV2K (800 lossless PNG images)
- **Test data:** CASIA 2.0 (200 tampered images)

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Project info and pipeline stages |
| `/health` | GET | Status check — wake Space before demo |
| `/debug` | GET | Model diagnostic — verify output range |
| `/analyze` | POST | Main inference endpoint |

## `/analyze` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | File | required | JPG / PNG / TIF image |
| `threshold` | float | 0.5 | Operating point 0.2-0.7 (0.5 = auto ensemble) |
| `condition` | string | clean | Stress test: clean / noise / jpeg |

## Performance

| Metric | Value |
|---|---|
| F1-Score (CASIA 2.0) | ~11% |
| Specificity | >95% |
| AUC-ROC | >0.70 |

## Training Innovation

- **Two-image synthetic splicing** — genuine noise-floor mismatch at splice boundaries
- **Social Media Simulation** — 50% of training images randomly JPEG compressed at Q50-90
- **Self-supervised** — no labeled forgery data used at any point