# NAS3R Candidate Notes For Cotton UAV Reconstruction

## What NAS3R Is

NAS3R is a CVPR 2026 self-supervised feed-forward 3D reconstruction framework
for novel-view synthesis. It jointly predicts camera parameters and explicit 3D
Gaussians from uncalibrated, unposed context views, then renders target views for
photometric supervision. The public implementation provides pretrained
RealEstate10K checkpoints and evaluation code for novel-view synthesis and pose
estimation.

The important technical pieces for this project are:

- unposed/uncalibrated image handling;
- camera prediction;
- depth-based Gaussian centers;
- compatibility with VGGT-style pretrained initialization;
- 2-view and multiview settings.

## Fit For The Cotton Project

NAS3R is relevant because our UAV images may not have a clean calibrated
multi-view setup ready on day one. It can be tested as a fast reconstruction
candidate for a small post-defoliation patch where cotton bolls are visible from
nearby overlapping frames.

It is not yet the primary measurement backend for the ICPA paper. The reasons
are practical:

1. The released checkpoints are trained on RealEstate10K, not cotton fields.
2. The method optimizes for novel-view synthesis and pose/geometry prediction,
   not agronomic trait measurement.
3. Cotton bolls are small, repetitive, bright objects, so failure can look
   visually plausible while still being metrically wrong.
4. Boll diameter and volume still require GSD, GCPs, camera metadata, or
   physical scale references.

## Recommended Role

Use NAS3R as an exploratory learned reconstruction baseline after the classical
baseline:

1. COLMAP/SfM-MVS on a local post-defoliation image subset.
2. DUSt3R/MASt3R or VGGT-style point reconstruction if available.
3. NAS3R on the same local subset for pose-free Gaussian reconstruction.
4. Compare all outputs using the same crop, masks, and held-out views.

For the paper, NAS3R should be described as a candidate learned reconstruction
module, not as the core novelty. The novelty remains the cotton-specific
pipeline: pre/post-defoliation visibility, boll detection, mask-guided local
reconstruction, plot-grid mapping, and calibrated trait estimation boundaries.

## How To Test It On Our Dataset

Start with a very small subset:

- phase: post-defoliation;
- frames: 8-20 neighboring UAV images with clear overlap;
- target: one row segment or one local patch with visible cotton bolls;
- image size: downsampled to the model-supported resolution first;
- output: predicted cameras, Gaussian/depth representation if exportable,
  rendered target views, and pose consistency.

Evaluation checks:

| Check | Why it matters |
|---|---|
| Registered/usable view count | Confirms whether the method can use our UAV subset. |
| Held-out view rendering quality | Catches obvious reconstruction failures. |
| Boll-mask reprojection consistency | Tests whether detected bolls land in stable 3D regions. |
| Local scale availability | Determines whether any length/volume claim is allowed. |
| Failure cases on pre-defoliation | Shows whether occlusion/foliage breaks the method. |

## Decision

NAS3R is useful for our roadmap, especially because it attacks the exact pain
point of unposed images. It should be incorporated as an **exploratory
reconstruction baseline** and possible supplementary visual comparison. It
should not replace COLMAP/SfM-MVS as the first defensible baseline, and it
should not be used for final cotton boll diameter or volume unless scale and
multi-view consistency are validated.

## Sources

- NAS3R project page: https://ranrhuang.github.io/nas3r/
- NAS3R GitHub: https://github.com/ranrhuang/NAS3R
- NAS3R paper link from project page: https://arxiv.org/abs/2603.27455
