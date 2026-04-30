# Architecture: Detection-Guided 3D Cotton Boll Phenotyping

## One-Sentence System Definition

We build a weakly supervised, detection-guided 3D phenotyping system that converts pre- and post-defoliation UAV RGB imagery into boll-level 3D location, count, diameter, volume, visibility, occlusion, and morphology-change measurements.

## Core Architecture

```text
Pre/Post UAV RGB images
  -> dataset audit and phase split
  -> 2D cotton boll detection from prior accepted work
  -> detection-guided mask refinement
  -> COLMAP/SfM camera poses and scene reconstruction
  -> DINOv2 semantic feature correspondences
  -> multi-view boll association
  -> 3D boll localization
  -> morphology extraction
  -> pre/post defoliation analysis
  -> optional structured agronomic report
```

## Module Roles

| Module | Input | Output | Purpose |
|---|---|---|---|
| Dataset audit | Raw UAV folders | Clean manifests, counts, timestamps, duplicate report | Prevent dataset leakage and reconstruction confusion. |
| 2D boll detection | RGB images | Candidate boll boxes, centers, counts | Reuse prior accepted detector and avoid dense manual annotation. |
| Mask refinement | Candidate boxes/centers | Approximate boll masks | Convert detections into silhouettes for geometry and size estimation. |
| Camera geometry | UAV image sequence | Camera intrinsics/extrinsics, sparse/dense reconstruction | Establish the metric 3D frame. |
| Semantic matching | Image pairs, DINOv2 features | Robust correspondences in texture-poor regions | Compensate for weak photometric texture on cotton lint. |
| Multi-view association | Detections, masks, poses, semantic features | Same-boll tracks across views | Identify which 2D detections correspond to the same physical boll. |
| 3D localization | Multi-view tracks and poses | Boll centers and 3D clusters | Place bolls into the reconstructed field/canopy. |
| Morphology extraction | 3D boll instances and masks | Count, diameter, volume, visibility, occlusion | Produce the actual phenotyping contribution. |
| Optional reporting | Morphology table/JSON | Human-readable summary | Interpret measurements after the algorithm has already computed them. |

## What Is Novel

1. Detection-guided multi-view boll association at field scale.
2. Semantic correspondence support for textureless cotton lint.
3. Pre/post-defoliation as a controlled visibility intervention.
4. Boll-level morphology estimation: count, 3D location, diameter, volume, visibility, and occlusion.
5. Minimal manual annotation by reusing prior detector outputs as weak supervision.

## What The LLM Does And Does Not Do

The LLM is not used for 3D reconstruction, triangulation, diameter estimation, or volume estimation.

The LLM may be used only after morphology extraction to summarize a structured report, flag uncertainty, and prepare an agronomist-facing explanation. The paper must remain valid if the LLM block is removed.

## CVPR-Style Framing

We introduce a weakly supervised, detection-guided 3D phenotyping pipeline that couples foundation-model semantic features with multi-view geometry for cotton boll reconstruction. Unlike conventional SfM pipelines that rely primarily on photometric texture, our method uses detection-derived object priors and DINOv2 semantic correspondences to improve matching and instance association in textureless cotton lint regions. The system estimates boll-level 3D location, diameter, volume, and visibility from UAV imagery without requiring dense manual annotation.

## IEEE-Style Framing

The proposed system is organized as a modular UAV-based phenotyping architecture. The input layer receives pre- and post-defoliation RGB imagery. The perception layer detects and refines cotton boll candidates. The reconstruction layer estimates camera geometry and semantic correspondences. The association layer links boll candidates across overlapping views. The measurement layer extracts boll-level morphological traits, including count, diameter, volume, visibility, and occlusion. An optional decision-support layer converts quantitative measurements into structured agronomic summaries with human expert review.
