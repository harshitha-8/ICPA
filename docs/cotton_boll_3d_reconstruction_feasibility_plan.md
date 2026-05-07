# Cotton Boll 3D Reconstruction Feasibility Plan

## Core Answer

The hardest part of this project is not making a 3D-looking map. The hard part
is recovering **organ-scale, metric 3D geometry of cotton bolls** from UAV
imagery where each boll is tiny, repetitive, partly occluded, and often visible
from only one near-nadir direction.

If the input is a single orthomosaic or one nadir UAV frame, true boll-scale 3D
reconstruction is not scientifically identifiable. We can build a 2.5D
visibility surface and mask-derived ellipsoid proxies, but we cannot honestly
claim calibrated boll diameter or volume without scale, parallax, or ground
truth.

The correct strategy is therefore staged:

1. use the orthomosaic/nadir frame for plot mapping and visible-boll candidate
   selection;
2. select local post-defoliation patches where bolls are exposed;
3. reconstruct those patches from overlapping raw UAV frames or small additional
   oblique captures;
4. project boll masks into the local 3D model;
5. estimate diameter and volume only for high-confidence, scale-calibrated
   boll clusters.

## What We Can Do From The Current Orthomosaic/Nadir Image

These outputs are defensible now:

- cotton row and column mapping;
- pre/post-defoliation visibility comparison;
- visible cotton boll candidate detection;
- mask length and width in pixels;
- mask area and visibility score;
- 2.5D canopy/lint visibility map;
- PLY export for inspection;
- ellipsoid volume proxy if a GSD value is supplied.

These outputs must be labeled as proxy:

- diameter from mask width;
- length from mask major axis;
- volume from ellipsoid assumption;
- local height from image-derived depth or brightness/texture proxy.

These outputs are not defensible from one orthomosaic alone:

- true 3D boll shape;
- true 3D boll diameter;
- true boll volume;
- bale or yield estimation from volume;
- metric cotton-row canopy structure.

## Why Single-Orthomosaic Boll 3D Is Underconstrained

A nadir orthomosaic collapses a 3D plant into one top-down texture. For a small
cotton boll, several different 3D shapes can create nearly the same top-down
appearance:

- a spherical boll;
- an elongated boll partly hidden by branch structure;
- two adjacent bolls touching each other;
- a bright soil/residue patch;
- lint visible on top of a boll whose lower half is occluded.

Without multiple viewpoints, shadows with known illumination, calibrated scale,
or a learned model trained specifically on cotton boll geometry, the inverse
problem has many valid 3D explanations. A top-tier paper should acknowledge this
instead of pretending the 3D geometry is solved.

## Practical Reconstruction Paths

### Path A: Real Local Multi-View Reconstruction From Existing UAV Frames

This is the best immediate path if the dataset contains overlapping raw frames
around the same plot.

Pipeline:

1. choose 20-80 neighboring post-defoliation frames;
2. run COLMAP/SfM-MVS or OpenDroneMap on the local subset;
3. record registered images, sparse points, dense points, and reprojection
   error;
4. select a local patch with visible cotton bolls;
5. run detector and SAM-style masks on the same frames;
6. project masks into the reconstructed space using camera poses;
7. cluster repeated observations into 3D boll candidates;
8. estimate traits only for clusters with multiple-view support.

This path can support real paper claims if scale is available through GSD,
camera metadata, GCPs, or measured field references.

### Path B: Learned Sparse-View Reconstruction When COLMAP Fails

If cotton lint is too textureless for classical matching, test learned
reconstruction models on the same local subset:

- DUSt3R/MASt3R-style dense matching and reconstruction;
- VGGT-style camera/depth/point estimation;
- NAS3R-style unposed feed-forward Gaussian reconstruction.

These methods can be valuable because they do not rely only on SIFT-like
photometric texture. However, they must be evaluated carefully. A visually
plausible learned reconstruction is not automatically a metric boll
measurement.

Evaluation:

- held-out view consistency;
- camera/pose stability;
- boll-mask reprojection consistency;
- point density inside boll masks;
- scale consistency against any known GSD or measured reference.

### Path C: Local 3D Gaussian Splatting After Pose Recovery

Once camera poses are stable, local 3D Gaussian Splatting can create a strong
visual reconstruction of the cotton row/patch. It is especially useful for:

- supplementary fly-through videos;
- web viewer demos;
- mask projection into a 3D scene;
- qualitative pre/post defoliation comparison.

It should not be the sole source of diameter or volume. Geometry measurements
must be validated against point clouds, reprojection consistency, or physical
measurements.

### Path D: Single-Image Proxy Morphology

If no overlapping frames work, the fallback is a transparent proxy pipeline:

1. detect visible bolls;
2. extract masks;
3. compute mask major/minor axes;
4. convert pixels to centimeters only if GSD is known;
5. estimate ellipsoid volume proxy;
6. report confidence and limitations.

This path is useful for ICPA if framed as **measurement-readiness and visibility
analysis**, not final 3D reconstruction.

### Path E: Minimal Extra Data To Unlock Real Boll 3D

The most efficient extra data collection would be small, not large:

- 3-5 representative post-defoliation row segments;
- 30-80 oblique images per segment;
- one measured scale marker or row spacing reference;
- optional phone/ground video around selected bolls;
- manual diameter/length measurements for 30-50 visible bolls.

This would let the paper report a calibrated local validation subset while still
using the large UAV dataset for field-scale scouting and pre/post analysis.

## Recommended Paper Framing

The paper should not claim:

> We reconstruct every cotton boll in 3D from an orthomosaic.

It should claim:

> We introduce a visibility-aware UAV phenotyping pipeline that uses
> pre/post-defoliation imagery to identify measurement-ready cotton boll
> candidates, builds local 2.5D and multi-view reconstruction evidence where
> possible, and estimates organ-scale morphology only under explicit
> scale/calibration constraints.

This is stronger because it is honest, testable, and reviewer-defensible.

## Immediate Experimental Plan

### Experiment 1: Orthomosaic/Nadir 2.5D Baseline

Use the existing 2.5D orthomap script:

```bash
python3 tools/build_uav_orthomap_3d_figure.py
```

Report:

- pre/post RGB map;
- canopy/lint visibility height proxy;
- PLY export;
- visible-boll candidate density.

### Experiment 2: Local Overlap Subset For Real Geometry

Create 3 local subsets:

| Subset | Phase | Frames | Purpose |
|---|---|---:|---|
| S1 | post | 20 | fastest geometry smoke test |
| S2 | post | 60 | stronger local reconstruction |
| S3 | pre | 60 | occlusion/failure comparison |

Run:

- COLMAP/SfM-MVS first;
- learned reconstruction second if COLMAP weakens;
- local 3DGS only after pose recovery.

### Experiment 3: Boll Mask Projection

For each reconstructed subset:

- detect bolls in every frame;
- extract masks;
- project mask pixels/rays into the local 3D model;
- cluster observations;
- report view support and confidence.

### Experiment 4: Trait Validation Boundary

Only report metric diameter/volume if:

- GSD or GCP/scale is available;
- the boll appears in multiple views;
- mask reprojection is stable;
- at least a small manual measurement set exists.

Otherwise report:

- proxy length;
- proxy width;
- proxy volume;
- confidence;
- failure reason.

## Method Ranking For This Specific Problem

| Rank | Method | Use | Claim Strength |
|---:|---|---|---|
| 1 | COLMAP/OpenDroneMap on overlapping local frames | First real geometry baseline | Strong if images register and scale exists |
| 2 | DUSt3R/MASt3R/VGGT-style learned reconstruction | Rescue path for weak texture | Strong only after consistency checks |
| 3 | Local 3DGS from stable poses | Visual reconstruction and mask projection | Good for figures; measurement needs validation |
| 4 | NAS3R feed-forward reconstruction | Exploratory unposed baseline | Useful supplement, not main claim yet |
| 5 | Single-image monocular depth/2.5D map | Visualization and proxy scouting | Not metric boll 3D |
| 6 | Ellipsoid from 2D mask | Fallback morphology proxy | Needs explicit proxy language |

## Bottom Line

For ICPA, the credible near-term contribution is not "perfect 3D cotton boll
reconstruction from one orthomosaic." The credible contribution is:

**pre/post-defoliation visibility-aware cotton boll phenotyping with a staged
path from field-scale UAV scouting to local, scale-aware 3D boll reconstruction.**

That gives the paper a clean research story:

- field-scale relevance from UAV orthomosaic/nadir imagery;
- organ-scale ambition through local reconstruction;
- honest uncertainty around volume;
- clear experimental path for stronger results.

## Useful References To Cite Or Test

- 3D reconstruction and characterization of cotton bolls in situ based on UAV
  technology: https://www.sciencedirect.com/science/article/pii/S0924271624000364
- Cotton3DGaussians: https://www.sciencedirect.com/science/article/pii/S0168169925003990
- DUSt3R: https://arxiv.org/abs/2312.14132
- MASt3R: https://arxiv.org/abs/2406.09756
- VGGT: https://arxiv.org/abs/2503.11651
- NAS3R: https://github.com/ranrhuang/NAS3R
- Depth Anything V2: https://arxiv.org/abs/2406.09414
