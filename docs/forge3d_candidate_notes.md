# forge3d Candidate Notes For Cotton UAV 3D Maps

## What It Is

forge3d is a Python-facing 3D terrain renderer built on a Rust/WebGPU core. It
is designed for GPU-accelerated rendering of DEMs, terrain scenes, overlays,
point clouds, camera animations, and high-resolution snapshots from Python.

The linked social example uses this kind of stack for 3D satellite timelapse:
Sentinel-2 imagery draped over Copernicus DEM terrain with an orbiting camera.

## Fit For This Project

forge3d is useful for the cotton project as a **publication-quality renderer**
and interactive 3D map layer. It is not a reconstruction or measurement
algorithm.

Good uses:

- render the pre/post UAV orthomap as a clean 3D terrain-like surface;
- create better videos than matplotlib for the current 2.5D UAV map;
- overlay row/column boundaries, detected boll centers, or plot-cell summaries;
- render PLY/point-cloud outputs from COLMAP, NAS3R, or 3D Gaussian Splatting
  once those reconstructions exist;
- produce supplementary fly-through visuals for the ICPA paper.

Not good for:

- estimating cotton boll diameter;
- estimating cotton boll volume;
- solving scale ambiguity;
- replacing COLMAP/SfM, NAS3R, MASt3R, VGGT, or local 3DGS;
- making scientific claims about metric 3D geometry.

## Decision

Use forge3d only after the geometry exists. In the current MVP, it can improve
the visual quality of the real UAV-textured 2.5D orthomap. For the final
scientific pipeline, use it as a renderer for:

1. orthomosaic or DEM-backed field map;
2. camera trajectory;
3. row/column plot grid;
4. detected boll anchors;
5. reconstructed point cloud or splat output.

The paper should phrase it as a visualization backend, not a phenotyping
method.

## Project Placement

Recommended stack:

```text
UAV images
  -> image selection / orthomosaic / COLMAP or learned reconstruction
  -> point cloud, mesh, splats, or 2.5D map
  -> boll detection and mask projection
  -> trait table with uncertainty
  -> forge3d or WebGL viewer for publication-quality map rendering
```

## Why It Is Still Useful

The current matplotlib 2.5D map is scientifically honest, but visually rough.
forge3d could make the same data look more professional without inventing fake
geometry. This is the right lesson from the linked post: use better rendering,
not fake reconstruction.

## Sources

- forge3d PyPI: https://pypi.org/project/forge3d/
- forge3d GitHub from PyPI project links: https://github.com/milos-agathon/forge3d
