# Cotton Boll Reconstruction Method Scan

## Practical Decision

For the cotton project, the correct approach is a three-lane reconstruction
strategy:

1. **Immediate lane:** local single-image 2.5D boll/cluster reconstruction from
   the clearest crops. This creates visual and quantitative proxy artifacts now.
2. **Primary scientific lane:** local multi-view reconstruction from overlapping
   UAV frames using COLMAP/OpenDroneMap first, then DUSt3R/MASt3R/VGGT/MV-DUSt3R
   if classical matching fails.
3. **Viewer lane:** Gaussian Splatting/SPZ/forge3d only after geometry exists,
   for interactive visualization and supplementary demos.

This gives us fast experiments without pretending single-view proxy geometry is
validated boll volume.

## Methods Worth Considering

| Method family | Examples | Useful for cotton? | Role |
|---|---|---|---|
| Classical SfM/MVS | COLMAP, OpenDroneMap | Yes | First real geometry baseline for overlapping UAV images. |
| Pairwise foundation 3D | DUSt3R, MASt3R | Yes | Rescue path when cotton lint has weak SIFT texture. |
| Multi-view feed-forward 3D | MV-DUSt3R+ | Yes, if code/weights available | Faster sparse-view scene reconstruction without full calibration. |
| General 3D foundation models | VGGT | Yes, exploratory | Camera/depth/point estimation on local image windows. |
| Feed-forward Gaussian splats | NAS3R, NoPoSplat-style systems | Maybe | Unposed local scene reconstruction; needs validation. |
| Single-view object 3D | Triplane-Gaussian, Splatter Image/SHARP-style systems | Limited | Can inspire local proxy views, but not metric boll traits. |
| Monocular depth | Depth Anything V2, UniDepth, MoGe | Yes as proxy | Converts clear crops into local 2.5D inspection surfaces. |
| Semantic 3D splats | Feature 3DGS/FMGS-like systems | Later | Attach boll semantics to reconstructed scenes. |

## Why Single-Image 3D Is Not Enough

Single-view object reconstruction methods can infer plausible geometry from a
single image. That is useful for visualization, but cotton bolls are small,
clustered, repetitive, and partly occluded. A generated back side of a boll is
not a measurement. Therefore, single-image outputs must be called:

- local 2.5D proxy;
- monocular depth proxy;
- measurement-readiness visualization;
- hypothesis-generating reconstruction.

They should not be called:

- ground-truth 3D;
- calibrated boll volume;
- validated girth;
- bale estimate.

## Best Near-Term Experiment

Use the mined candidates to select highly visible post-defoliation crops:

```text
top ranked crop
  -> lint/vegetation/texture depth proxy
  -> local 2.5D PLY
  -> local 3D gallery figure
  -> candidate set for COLMAP/NAS3R/MASt3R local reconstruction
```

This produces an immediate figure for the paper and identifies the local targets
where real multi-view reconstruction should be attempted.

## Best Real-3D Experiment

Use the local reconstruction subsets already generated:

```text
20 or 60 neighboring UAV frames
  -> COLMAP/OpenDroneMap
  -> if weak, DUSt3R/MASt3R/VGGT/MV-DUSt3R
  -> project boll masks into the scene
  -> cluster repeated observations
  -> estimate traits only where scale and view support exist
```

The paper should compare pre vs post:

- registered frame count;
- point count;
- detected candidate count;
- reconstructed candidate count;
- multi-view support per candidate;
- trait-confidence distribution.

## Sources

- DUSt3R: https://arxiv.org/abs/2312.14132
- MASt3R: https://arxiv.org/abs/2406.09756
- MV-DUSt3R+, CVPR 2025: https://cvpr.thecvf.com/virtual/2025/poster/34325
- Triplane Meets Gaussian Splatting, CVPR 2024:
  https://cvpr.thecvf.com/virtual/2024/poster/31585
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- NAS3R: https://github.com/ranrhuang/NAS3R
- Cotton3DGaussians: https://www.sciencedirect.com/science/article/pii/S0168169925003990
