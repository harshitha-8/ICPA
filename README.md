# Agronomist-in-the-Loop Semantic 3D Reconstruction of Cotton Boll Morphology from UAV Imagery

**Target Conference:** 17th International Conference on Precision Agriculture (ICPA) / 11th ConBAP  
**Location:** Porto Alegre, Brazil — July 13–16, 2026  
**Submission Deadline:** May 22, 2026

## Overview

This repository implements a novel pipeline for morphological analysis of cotton bolls using UAV imagery, combining:

1. **Semantic Feature Fields** — DINOv2-based dense correspondence across multi-view UAV images
2. **Zero-Shot Instance Segmentation** — SAM 2 for individual boll segmentation with temporal consistency
3. **Semantic 3D Reconstruction** — Feature-aligned reconstruction surpassing classical SfM limitations
4. **Morphological Measurement** — Automated extraction of boll diameter, volume, curvature, and visibility metrics
5. **Agronomist-in-the-Loop Reasoning** — Edge-deployable LLM for translating geometric signals into management recommendations

## Key Hypothesis

Traditional Structure-from-Motion fails for cotton boll reconstruction because white cotton lacks texture, wind introduces temporal inconsistency, and leaf occlusion prevents feature matching. Semantic correspondence fields extracted using vision foundation models enable reconstruction where pixel matching fails.

## Dataset

- **Pre-defoliation:** 680 DJI UAV images (dense canopy, occlusion-heavy)
- **Post-defoliation:** 869 DJI UAV images (exposed boll structures)
- **Captured:** September 29, 2025, ~3h interval between flights

## Repository Structure

```
ICPA/
├── paper/           # Manuscript sections (Markdown → Word)
├── experiments/     # Experiment scripts and results
├── pipeline/        # Core processing pipeline
├── llm/             # LLM reasoning engine
├── configs/         # Configuration files
├── logs/            # Experiment logs
└── outputs/         # Generated artifacts
```

## Requirements

```
torch>=2.0
torchvision
transformers
segment-anything-2
open3d
opencv-python
numpy
scipy
scikit-learn
pillow
exifread
tqdm
```

## Quick Start

```bash
# 1. Prepare dataset
python pipeline/preprocessing/prepare_dataset.py

# 2. Extract DINOv2 features
python pipeline/feature_alignment/extract_dinov2_features.py

# 3. Run SAM2 segmentation
python pipeline/segmentation/run_sam2_segmentation.py

# 4. Run COLMAP baseline
python pipeline/reconstruction/run_colmap_baseline.py

# 5. Run semantic reconstruction
python pipeline/reconstruction/semantic_bundle_adjustment.py

# 6. Extract morphology
python pipeline/morphology_extraction/extract_boll_morphology.py

# 7. Run LLM reasoning
python llm/reasoning_engine/reasoning_engine.py
```

## License

Research use only. Contact authors for permissions.
