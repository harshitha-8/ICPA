# SPZ 4 Candidate Notes For Cotton 3D Reconstruction Delivery

## What SPZ 4 Adds

SPZ 4 is Niantic Spatial's updated compressed format for 3D Gaussian splats. It
is best understood as a delivery and interchange format, not a reconstruction
method. The stated improvements are aimed at larger web/mobile splat scenes:

- about 3-5x faster encoding than prior SPZ versions;
- roughly 1.5-2x faster end-to-end loading in benchmarked scenes;
- about 10x smaller files than uncompressed PLY splats;
- removal of the earlier 10-million-point loader/deserializer limit;
- parallel ZSTD streams for positions, colors, scales, rotations, alphas, and
  spherical harmonics;
- plaintext header for quick metadata inspection;
- extension support, including camera framing metadata.

## Fit For This Project

SPZ 4 is useful for the cotton project only after a real Gaussian-splat scene
exists. It can make the field-scale viewer much more practical by compressing
large splat outputs and loading them faster in the browser.

Good uses:

- compress local 3DGS/NAS3R/other splat outputs for the app;
- deliver pre/post cotton row reconstructions in a browser viewer;
- store camera orbit/framing metadata for consistent paper demos;
- support large scenes where PLY is too heavy for web inspection;
- package supplementary visual evidence without massive files.

Not good for:

- reconstructing cotton from UAV images;
- estimating boll diameter;
- estimating boll volume;
- solving scale ambiguity;
- replacing COLMAP, NAS3R, MASt3R, VGGT, or 3DGS training.

## Decision

Use SPZ 4 as an **export and compression layer** for Gaussian splat outputs, not
as a phenotyping method. For the ICPA project, SPZ 4 belongs in the viewer and
supplementary-material pipeline:

```text
overlapping UAV images
  -> COLMAP/NAS3R/3DGS reconstruction
  -> Gaussian splat scene
  -> SPZ 4 compression
  -> browser/app viewer with plot grid, camera path, and boll anchors
```

The scientific measurements should still come from calibrated geometry and
validated mask/point association, not from the compressed file format.

## How It Helps The Paper

SPZ 4 can make the project look more mature in supplementary material:

1. a smaller shareable scene file;
2. faster browser rendering for reviewers or collaborators;
3. a consistent camera orbit around the same cotton row;
4. scalable visualization if the reconstruction grows from a local patch to a
   full plot.

The paper text should phrase this as:

> For interactive visualization, reconstructed Gaussian splat scenes can be
> exported to a compact web-deliverable format such as SPZ 4. This export layer
> affects rendering and sharing efficiency only; it is not used to compute
> agronomic measurements.

## Sources

- Niantic Spatial SPZ 4 announcement:
  https://www.nianticspatial.com/blog/spz4
- SPZ GitHub:
  https://github.com/nianticlabs/spz
