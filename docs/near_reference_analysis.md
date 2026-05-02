# Near-Reference Analysis For Cotton 3D Phenotyping

## Why These References Matter

The two ScienceDirect papers are close enough that they should shape the paper's positioning and experiments. They do not block the project, but they make the contribution boundary sharper.

## Reference 1: UAV 3D Cotton Boll Reconstruction

**Paper:** *3D reconstruction and characterization of cotton bolls in situ based on UAV technology*, ISPRS Journal of Photogrammetry and Remote Sensing, 2024.

**Core idea:** The paper uses UAV imagery and a cross-circling oblique route to reconstruct field-scale cotton boll point clouds and estimate boll number, volume, spatial distribution, and yield. It shows that oblique/circling capture can reduce occlusion compared with nadir capture.

**Key numbers reported by the abstract/search page:**

| Trait | CCO route | Nadir route |
|---|---:|---:|
| Boll count R2 | 0.92 | 0.73 |
| Seed cotton yield R2 | 0.70 | 0.62 |
| Lint cotton yield R2 | 0.75 | 0.55 |

**What we learn:**

- The paper makes UAV-based 3D cotton boll phenotyping a proven close baseline.
- Acquisition route matters as much as the reconstruction algorithm.
- Occlusion in middle/lower canopy is a central difficulty.
- Boll count and boll volume can be linked to yield, but count and volume may be better for different yield components.
- Its deep-forest component is best treated as a tabular prediction baseline after 3D features are extracted, not as a reconstruction method.

**How we differentiate:**

- We have a paired pre/post-defoliation dataset, which can be framed as a controlled visibility intervention.
- We can evaluate how defoliation changes visible boll count, candidate density, and eventual 3D morphology confidence.
- We can add semantic/foundation-model correspondence for white, texture-poor cotton lint where classical photogrammetry may fail.
- We can compare not only nadir/oblique route quality, but pre/post visibility and semantic association quality.
- We can include a cascade-forest baseline for candidate validation, adhered-boll rejection, or plot-cell trait prediction from extracted morphology features.

## Reference 2: Cotton3DGaussians

**Paper:** *Cotton3DGaussians: Multiview 3D Gaussian Splatting for boll mapping and plant architecture analysis*, Computers and Electronics in Agriculture, 2025.

**Core idea:** The paper reconstructs single cotton plants with 3D Gaussian Splatting from 360-degree smartphone RGB images. It maps 2D boll masks from multiple views into 3D space, removes redundant bolls by cross-view clustering, and estimates boll number, boll volume, plant height, and canopy size.

**Key numbers reported by the abstract/search page:**

| Result | Reported value |
|---|---:|
| YOLOv11x vs SAM F1 improvement | +5.9 percentage points |
| 3DGS PSNR over NeRF | +6.91 |
| Boll number MAPE | 9.23% |
| Canopy size MAPE | 3.66% |
| Plant height MAPE | 2.38% |
| Boll volume MAPE | 8.17% |
| Plant weight error from convex boll volume | 19.3% |

**What we learn:**

- 3D Gaussian Splatting is now directly relevant to cotton boll phenotyping.
- A strong pipeline is: multiview RGB -> SfM camera parameters -> 3DGS -> 2D masks -> mask projection into 3D -> cross-view clustering -> trait extraction.
- YOLO-style supervised masks can outperform SAM for cotton boll instance masks.
- LiDAR comparison is a strong validation protocol.

**How we differentiate:**

- Cotton3DGaussians is single-plant/smartphone 360-degree scanning; our dataset is field/UAV-scale pre/post-defoliation imagery.
- Their contribution is high-fidelity 3DGS plus multiview mask projection; ours should emphasize field-scale paired defoliation, visibility/occlusion change, and detection-guided semantic reconstruction.
- We can borrow the evaluation pattern: boll number, volume, plant/canopy structure, PSNR/visual quality if 3DGS is used, and MAPE where ground truth exists.

## Updated Method Direction

The strongest pipeline now is:

```text
Pre/post UAV RGB images
  -> phase-aware dataset audit
  -> inherited cotton boll detector / YOLO-style detector
  -> SAM or SAM2 mask refinement as optional comparison
  -> COLMAP/VGGT/MASt3R camera poses
  -> point cloud and/or 3D Gaussian Splatting reconstruction
  -> project 2D boll masks into 3D
  -> cross-view clustering to remove duplicate bolls
  -> extract count, 3D center, volume proxy, visibility, and occlusion
  -> compare pre vs post defoliation
  -> produce field-scale viewer/flythrough with semantic anchors
```

## Experiments To Add

| Experiment | Baselines | Metrics |
|---|---|---|
| Reconstruction quality | COLMAP point cloud, 3DGS, 2.5D scaffold, optional VGGT/MASt3R | Registered images, point density, visual quality, PSNR if held-out views exist |
| Acquisition/visibility | pre-defoliation vs post-defoliation | visible count, raw candidates, occlusion score, view support per boll |
| Mask source | inherited detector, SAM/SAM2, YOLO if weights are available | count MAE, mIoU/AP if labels exist, duplicate rate after 3D clustering |
| 3D boll mapping | 2D-only count, 3D projected masks, cross-view clustered bolls | duplicate reduction, 3D center stability, volume distribution |
| Trait estimation | count-only, volume proxy, count+volume | relation to manual/yield/plot labels if available |
| Tabular prediction | heuristic threshold, cascade/deep forest, optional XGBoost/MLP | valid-boll F1, adhered-boll F1, yield/trait MAE/R2 if labels exist |

## Paper Framing Adjustment

Do not claim that the work is the first 3D cotton boll reconstruction paper. That is not defensible.

The more defensible claim is:

> This work studies detection-guided semantic 3D cotton boll phenotyping from paired pre- and post-defoliation UAV imagery. Unlike prior cotton 3D reconstruction and single-plant 3DGS studies, the proposed setting evaluates defoliation as a visibility intervention and uses detection-guided multi-view association to estimate field-scale boll morphology and uncertainty.

## Visual Target From YouTube References

The two YouTube references appear to indicate the desired visual style: camera movement through a reconstructed scene, dense point/splat geometry, visible path, and semantic anchors. The local MVP should therefore evolve from a slideshow into:

- colored point-cloud or 3DGS flythrough,
- camera trajectory overlay,
- 3D boll anchors,
- pre/post color-coded markers,
- side panel with count/visibility/volume summaries,
- optional supplementary video exported as MP4.

The current `cotton_pointcloud_flythrough.mp4` is a 2.5D scaffold toward this direction. The next scientific version should replace the synthetic height field with camera poses and geometry from COLMAP, VGGT, MASt3R, or 3DGS training.

## Sources

- 3D reconstruction and characterization of cotton bolls in situ based on UAV technology: https://www.sciencedirect.com/science/article/pii/S0924271624000364
- Cotton3DGaussians: Multiview 3D Gaussian Splatting for boll mapping and plant architecture analysis: https://www.sciencedirect.com/science/article/pii/S0168169925003990
- YouTube visual reference 1: https://www.youtube.com/watch?v=dFBguDtpaZg
- YouTube visual reference 2: https://www.youtube.com/watch?v=SqubR1GKakY
