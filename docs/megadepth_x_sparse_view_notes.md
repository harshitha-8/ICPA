# MegaDepth-X Sparse-View Notes For Cotton 3D

## Why The X Post Is Useful

The linked post points to MegaDepth-X and `colmapview.github.io` as a way to
inspect cleaned camera poses, depths, and COLMAP-style collections. The useful
idea for this project is not the landmark dataset itself. The useful idea is the
evaluation setup: clean pose/depth supervision plus controlled sparse-view
subsets.

The MegaDepth-X paper frames long-tail reconstruction as the hard regime where
images are sparse, noisy, unevenly distributed, symmetric, or repetitive. Cotton
UAV rows have a related failure mode: many frames look similar, white lint is
low-texture, leaves occlude bolls, and pre/post flights may not align perfectly.

## How To Use This In ICPA

Add a sparse-view robustness experiment:

| Split | Construction | Cotton question |
|---|---|---|
| Dense | use every selected frame | best-case reconstruction and morphology |
| 2x sparse | every second frame | how fast geometry/measurement degrades |
| 4x sparse | every fourth frame | sparse UAV coverage stress test |
| Phase-balanced sparse | same frame budget for pre and post | fair defoliation comparison |

Report:

- registered image fraction;
- sparse point count and dense point count, if available;
- held-out reprojection or novel-view/mask consistency;
- recoverable boll count;
- median view support per boll;
- high-confidence morphology coverage;
- diameter/volume proxy variance across sparse splits.

## Paper-Safe Wording

> Motivated by recent work on long-tail Internet photo reconstruction, we
> evaluate the cotton UAV pipeline under controlled sparse-view subsets. This
> stress test is relevant because crop rows produce repetitive, weakly textured
> imagery in which classical feature matching and learned reconstruction models
> may fail even when many raw frames are available.

## Practical Incorporation

Use `algorithms/sparse_view_sampler.py` to create dense and sparse manifests from
`outputs/reconstruction_inputs/.../reconstruction_images.csv`. These manifests
can be passed to COLMAP, VGGT/MASt3R, 3DGS, or the local viewer pipeline.

## Sources

- X post by Yehe Liu: https://x.com/YeheLiu/status/2050291460222705789
- Long-tail Internet photo reconstruction / MegaDepth-X: https://arxiv.org/abs/2604.22714
- COLMAP collection viewer mentioned in the post: http://colmapview.github.io
