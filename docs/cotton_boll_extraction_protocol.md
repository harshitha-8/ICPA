# Cotton Boll Extraction Protocol

## Motivation From Prior Cotton 3D Figures

The reference figures show three important ideas that should shape our app and
paper evaluation:

1. **Plot/column boundaries matter.** Cotton point clouds should be interpreted
   by plot or row segment, not as one unstructured cloud.
2. **Soil/background removal matters.** A vertical height profile can separate
   soil from canopy/boll structures before organ extraction.
3. **Adhered bolls are a real failure mode.** Convex-hull volume or bounding-box
   jumps can identify merged/adhered boll clusters.

## Current App Implementation

The local app now uses a stricter measurement-ready extraction layer:

```text
raw phase-aware detector candidates
  -> color evidence: white lint fraction
  -> canopy penalty: green fraction
  -> compactness/visibility score
  -> morphology-depth score
  -> size plausibility filter
  -> measurement-ready boll crops and proxy measurements
```

The app still reports raw candidates, but the gallery and measurement table use
the filtered measurement-ready subset. This makes the result closer to a real
extraction protocol instead of a loose detector visualization.

## What The Paper Should Report

| Stage | Metric |
|---|---|
| Raw detector | candidate count, adjusted count |
| Extraction filter | measurement-ready count, retained fraction |
| Cotton evidence | lint fraction, green fraction |
| 3D proxy | depth score, visibility score |
| Morphology | diameter proxy, volume proxy, high-confidence coverage |
| Failure mode | adhered/merged candidates, low-confidence pre-defoliation crops |

## Next Real 3D Step

Once real point clouds or Gaussian splats are available, this protocol should be
extended with:

- plot boundary segmentation from lowest points or row geometry;
- soil removal using vertical point-count profiles;
- 3D connected components for boll clusters;
- convex-hull or ellipsoid volume;
- merged/adhered boll rejection using sorted volume-difference thresholds.

Until then, the app measurements remain 2D/2.5D proxy measurements anchored to
the selected UAV frame.
