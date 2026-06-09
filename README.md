# Chest X-Ray Pathology Detection Pipeline

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/smpduong/chest-xray-pathology-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/smpduong/chest-xray-pathology-pipeline/actions)

> **⚠️ Medical Disclaimer:** This is a research-grade demonstration pipeline.  
> It is **NOT** an FDA-approved diagnostic device. All outputs require review by a qualified radiologist.

## Overview

A reproducible, production-oriented pipeline for chest X-ray analysis using
`torchxrayvision`'s DenseNet121, trained on NIH, CheXpert, MIMIC, PadChest,
RSNA, and Google/OpenI datasets.

**Key Features**
- **Primary/Local Input:** Drop a `chest_xray_input.png` next to the script
- **Fallback Download:** Automatically fetches from the COVID Chest X-ray dataset
- **Synthetic Fallback:** Generates bilateral lung phantoms if offline
- **Radiological Preprocessing:** Single-channel, [-1024, 1024] Hounsfield-like normalization
- **Multi-Label Inference:** 18 independent pathology probabilities (not single-class `argmax`)
- **Structured Logging:** Dual file/console output with timestamps

## Quick Start

```bash
# Clone
git clone https://github.com/smpduong/chest-xray-pathology-pipeline.git
cd chest-xray-pathology-pipeline

# Install
pip install -r requirements.txt

# Run with your own image (optional)
# Place a file named `chest_xray_input.png` in the repo root
python -m src.pipeline
```

**Outputs:**
- `output/sample_chest_xray.png`
- `output/xray_pathologies.png`
- `output/pipeline.log`

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Local Image    │────▶│  XRV Normalize   │────▶│  DenseNet121    │
│  (Primary)      │     │  [-1024, 1024]   │     │  (18 pathologies│
└─────────────────┘     └──────────────────┘     │  multi-label)   │
       │                                          └─────────────────┘
       │                                                   │
       ▼                                                   ▼
┌─────────────────┐                              ┌──────────────────┐
│  COVID Dataset  │                              │  Sigmoid +       │
│  (Fallback)     │                              │  Thresholding    │
└─────────────────┘                              └──────────────────┘
       │                                                   │
       ▼                                                   ▼
┌─────────────────┐                              ┌──────────────────┐
│  Synthetic      │                              │  Bar Chart + Log │
│  Generator      │                              │  (Top-10 probs)  │
└─────────────────┘                              └──────────────────┘
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **XRV DenseNet over ImageNet ResNet** | Outputs real pathologies (Effusion, Pneumothorax) vs. 1000 natural objects |
| **Multi-label sigmoid** | Chest X-rays often present multiple co-occurring pathologies |
| **Single-channel (1, 224, 224)** | Matches radiological training domain; RGB destroys performance |
| **File + console logging** | `output/pipeline.log` persists for HPC/batch; stdout for interactive |
| **Synthetic fallback** | Pipeline degrades gracefully when network is unavailable |

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Project Structure

```
src/
├── config.py          # Immutable configuration dataclass
├── pipeline.py        # End-to-end orchestration
├── preprocessing.py   # XRV normalization and geometric transforms
├── visualization.py   # Matplotlib pathology bar charts
└── utils.py           # Logging setup
tests/
├── test_preprocessing.py  # Tensor shape and range validation
└── test_pipeline.py       # End-to-end dry run
```

## License

MIT — See [LICENSE](LICENSE)

## Author

[Steven Duong](https://github.com/smpduong)
