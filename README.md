# Agronomist-in-the-Loop Semantic 3D Reconstruction of Cotton Boll Morphology from UAV Imagery

**Target Conference:** 17th International Conference on Precision Agriculture (ICPA) / 11th ConBAP  
**Location:** Porto Alegre, Brazil — July 13–16, 2026  
**Submission Deadline:** May 22, 2026

## Overview

This repository is an active ICPA 2026 research workspace for UAV-based cotton
boll phenotyping under pre- and post-defoliation conditions. The current system
focuses on defensible MVP outputs:

1. **Cotton boll counting** from UAV RGB imagery
2. **Pre/post-defoliation scouting** as a controlled visibility comparison
3. **Row-column plot mapping** for field-level inspection
4. **Mask-guided measurement candidates** for visible bolls
5. **Proxy morphology reports** that remain clearly separated from calibrated
   3D metrology

The next research stage is local UAV-to-boll 3D reconstruction: reconstruct the
scene or orthomosaic-supported region, select well-exposed boll patches, and
only then estimate calibrated boll length, diameter, and volume when scale
metadata or physical validation are available.

## Key Hypothesis

Traditional Structure-from-Motion can be fragile for cotton boll reconstruction
because white cotton lint is repetitive, wind introduces temporal inconsistency,
and leaf occlusion prevents stable feature matching. Defoliation therefore acts
as a visibility intervention: it may expose boll structure that is difficult to
measure in pre-defoliation imagery. The project tests whether detector- and
mask-guided local reconstruction can move cotton phenotyping from 2D count
toward organ-scale measurement without overclaiming metric 3D before calibration.

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

Key planning documents:

- `docs/uav_to_boll_3d_reconstruction_strategy.md` — staged reconstruction plan
  from UAV/orthomosaic scene to local visible-boll 3D investigation.
- `docs/cotton_boll_extraction_protocol.md` — mask-guided boll extraction and
  measurement protocol.
- `docs/plot_grid_mapping_protocol.md` — row-column scouting map protocol.
- `paper/word/` — Word-first ICPA manuscript draft artifacts.

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
