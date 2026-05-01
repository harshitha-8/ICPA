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
  -> optional 3D Gaussian Splatting reconstruction
  -> DINOv2 semantic feature correspondences
  -> multi-view boll association
  -> 3D boll localization
  -> morphology extraction
  -> pre/post defoliation analysis
  -> field-scale 3D viewer with camera path and boll overlays
  -> optional structured agronomic report
```

## Module Roles

| Module | Input | Output | Purpose |
|---|---|---|---|
| Dataset audit | Raw UAV folders | Clean manifests, counts, timestamps, duplicate report | Prevent dataset leakage and reconstruction confusion. |
| 2D boll detection | RGB images | Candidate boll boxes, centers, counts | Reuse prior accepted detector and avoid dense manual annotation. |
| Mask refinement | Candidate boxes/centers | Approximate boll masks | Convert detections into silhouettes for geometry and size estimation. |
| Camera geometry | UAV image sequence | Camera intrinsics/extrinsics, sparse/dense reconstruction | Establish the metric 3D frame. |
| 3DGS reconstruction | Camera poses and images | Field/plant Gaussian scene | Provide high-fidelity rendering and a comparator to point-cloud reconstruction. |
| Semantic matching | Image pairs, DINOv2 features | Robust correspondences in texture-poor regions | Compensate for weak photometric texture on cotton lint. |
| Multi-view association | Detections, masks, poses, semantic features | Same-boll tracks across views | Identify which 2D detections correspond to the same physical boll. |
| 3D localization | Multi-view tracks and poses | Boll centers and 3D clusters | Place bolls into the reconstructed field/canopy. |
| Morphology extraction | 3D boll instances and masks | Count, diameter, volume, visibility, occlusion | Produce the actual phenotyping contribution. |
| Field-scale viewer | Reconstruction, camera path, boll measurements | Interactive scene with splat/point cloud view and 3D boll anchors | Communicate geometry, coverage, and failure cases without replacing metric evaluation. |
| Optional reporting | Morphology table/JSON | Human-readable summary | Interpret measurements after the algorithm has already computed them. |

## What Is Novel

1. Detection-guided multi-view boll association at field scale.
2. Semantic correspondence support for textureless cotton lint.
3. Pre/post-defoliation as a controlled visibility intervention.
4. Boll-level morphology estimation: count, 3D location, diameter, volume, visibility, and occlusion.
5. Minimal manual annotation by reusing prior detector outputs as weak supervision.
6. A field-scale visualization layer that shows the reconstructed row, acquisition path, and boll-level anchors in one inspectable scene.

## What The LLM Does And Does Not Do

The LLM is not used for 3D reconstruction, triangulation, diameter estimation, or volume estimation.

The LLM may be used only after morphology extraction to summarize a structured report, flag uncertainty, and prepare an agronomist-facing explanation. The paper must remain valid if the LLM block is removed.

## CVPR-Style Framing

We introduce a weakly supervised, detection-guided 3D phenotyping pipeline that couples foundation-model semantic features with multi-view geometry for cotton boll reconstruction. Unlike conventional SfM pipelines that rely primarily on photometric texture, our method uses detection-derived object priors and DINOv2 semantic correspondences to improve matching and instance association in textureless cotton lint regions. The system estimates boll-level 3D location, diameter, volume, and visibility from UAV imagery without requiring dense manual annotation.

## IEEE-Style Framing

The proposed system is organized as a modular UAV-based phenotyping architecture. The input layer receives pre- and post-defoliation RGB imagery. The perception layer detects and refines cotton boll candidates. The reconstruction layer estimates camera geometry and semantic correspondences. The association layer links boll candidates across overlapping views. The measurement layer extracts boll-level morphological traits, including count, diameter, volume, visibility, and occlusion. An optional decision-support layer converts quantitative measurements into structured agronomic summaries with human expert review.

## Field-Scale Splat Viewer Extension

The VPS/splat-style interface is feasible as a visualization and inspection layer for this project. The agricultural version should reconstruct a cotton row or plot from UAV/ground video, render it as a point cloud or Gaussian-splat scene, show the camera acquisition path, and overlay each reconstructed boll as a 3D anchor with count, diameter, volume, visibility, and confidence metadata.

This layer should not be treated as the measurement algorithm by itself. The paper should state that the viewer is used for interpretability, quality control, and supplementary visual evidence, while the reported numbers come from calibrated geometry, multi-view association, and morphology extraction. That framing lets the work benefit from modern 3D visual quality without weakening the scientific claim.

Recommended implementation path:

```text
video/images
  -> frame sampling and blur filtering
  -> COLMAP or VGGT/MASt3R camera geometry
  -> dense point cloud or Gaussian-splat training
  -> boll detections projected into 3D
  -> cross-view clustering to remove duplicate bolls
  -> viewer export with camera path and semantic anchors
```

The strongest paper figure is a split panel: input UAV/ground frames, reconstructed field-scale scene, 3D boll anchors, and pre/post-defoliation visibility change. The strongest supplementary asset is an interactive scene or rendered fly-through.
