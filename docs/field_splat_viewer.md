# Field-Scale Splat Viewer For Cotton Boll Phenotyping

## What The Reference Video Suggests

The shared clip shows a navigation-style 3D scene with four useful ideas: a reconstructed environment, a camera or walking path, switchable scene views, and spatial anchors attached to objects or locations. For the cotton project, the closest scientific version is a field-scale 3D viewer that renders the reconstructed row and overlays boll-level measurements.

## What We Can Build

We can build a smaller, research-grade version for cotton:

1. Reconstruct a cotton row, plot, or plant cluster from UAV or ground video.
2. Export the geometry as a point cloud, mesh, or Gaussian-splat scene.
3. Draw the camera trajectory to show acquisition coverage.
4. Place boll anchors at reconstructed 3D centers.
5. Attach measurement metadata to each anchor: count id, diameter, volume, visibility score, occlusion score, and pre/post-defoliation status.
6. Render static paper figures and optional supplementary fly-through videos.

## What It Should Not Claim

The viewer should not be the source of the scientific measurements. It is a visualization and quality-control layer. The metric claims must come from calibrated camera geometry, multi-view association, triangulation, and morphology estimation.

## Why It Helps The ICPA Paper

This extension makes the work easier to understand for reviewers because they can see where the bolls were reconstructed, how camera coverage affects visibility, and why post-defoliation improves organ-level phenotyping. It also creates strong visual material for the paper and supplementary files without making the project dependent on an LLM.

## Paper Placement

Use this in:

- Method: as the final visualization/export layer.
- Experiments: qualitative reconstruction and failure-case analysis.
- Supplementary: interactive viewer, rendered fly-through, or per-plot scene inspection.
- Patent notes: field phenotyping system with semantic 3D organ anchors and decision support.

## Minimum Viable Prototype

The first prototype does not need real-time VPS. It only needs:

```text
input frames
  -> camera poses
  -> point cloud or splat
  -> boll centers and measurements
  -> exported viewer manifest
```

The viewer manifest can be a simple JSON file consumed by a later WebGL, Potree, Open3D, or SIBR/Gaussian-splat viewer.
