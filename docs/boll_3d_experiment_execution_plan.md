# Cotton Boll 3D Experiment Execution Plan

## Decision

The paper should not try to claim full boll-scale 3D reconstruction from one
orthomosaic. The defensible contribution is a **two-tier phenotyping system**:

1. **Field scale:** use pre/post UAV imagery for plot mapping, boll detection,
   visibility, and measurement-ready candidate selection.
2. **Organ scale:** reconstruct only selected local patches where the same bolls
   are visible across overlapping frames, then estimate boll girth and volume
   under explicit scale and confidence constraints.

This keeps the work ambitious but reviewer-defensible.

## Main Hypothesis

Post-defoliation imagery increases the number of cotton bolls that are
measurement-ready in 3D because foliage occlusion is reduced. Therefore, the
central result should not only be "how many bolls are detected," but:

- how many bolls are visible;
- how many are trackable across multiple UAV frames;
- how many can be projected into a local 3D reconstruction;
- how many have enough evidence for length, girth, and volume estimation.

## Experiment 1: Measurement-Ready Boll Mining

Run the detector on all pre/post images and rank candidates by measurement
readiness.

Per candidate:

- phase;
- image path;
- bbox;
- mask area;
- mask major axis;
- mask minor axis;
- lint fraction;
- green penalty;
- brightness;
- visibility score;
- estimated row/column cell;
- measurement-ready confidence.

Output:

- `measurement_ready_candidates.csv`
- pre/post count of candidates above confidence thresholds;
- top 100 post-defoliation crops for 3D reconstruction attempts.

Why this matters:

This produces a strong paper result even before true 3D succeeds. It quantifies
what defoliation unlocks.

## Experiment 2: Local Multi-Frame Patch Selection

Use image filenames/timestamps and neighboring frames to form small local
overlap windows.

Start with:

| Subset | Phase | Frames | Goal |
|---|---|---:|---|
| P1 | post | 20 | fast COLMAP smoke test |
| P2 | post | 60 | stronger local reconstruction |
| P3 | pre | 60 | occlusion comparison |

For each subset, save:

- copied image subset;
- image list CSV;
- candidate crops;
- phase metadata;
- expected reconstruction method.

## Experiment 3: Real Geometry Baselines

Run these in order:

1. COLMAP/SfM-MVS or OpenDroneMap;
2. DUSt3R/MASt3R/VGGT if COLMAP weakens;
3. NAS3R as an exploratory unposed Gaussian baseline;
4. local 3DGS only after poses are available.

Report:

- registered image count;
- sparse/dense point count;
- reprojection error or held-out view error;
- runtime;
- whether metric scale is available.

The paper should show pre vs post geometry success, not only visual examples.

## Experiment 4: Multi-View Boll Association

For each local reconstruction:

1. detect boll candidates in every frame;
2. extract SAM-style masks;
3. project detection centers or mask rays using camera poses/depth;
4. cluster candidate observations in 3D;
5. keep only clusters with at least two supporting views.

Per 3D cluster:

- center;
- view support;
- mean reprojection error;
- mask agreement;
- visibility;
- confidence;
- phase;
- row/column cell.

This is the true bridge from 2D counting to boll-level 3D.

## Experiment 5: Girth And Volume Estimation

Use a tiered trait policy:

### Tier A: Calibrated 3D Trait

Allowed only if:

- camera/GSD/GCP/scale is available;
- the boll has multi-view support;
- mask reprojection is stable;
- a manual measurement subset exists or can be collected.

Estimate:

- length;
- width/girth proxy;
- ellipsoid volume;
- uncertainty interval.

### Tier B: Multi-View Proxy Trait

Allowed if the boll is observed in multiple frames but lacks external scale.

Report:

- normalized length;
- normalized width;
- relative volume;
- confidence;
- no cm or cm3 claim.

### Tier C: Single-View Proxy Trait

Allowed for orthomosaic/nadir-only candidates.

Report:

- 2D mask major/minor axes;
- ellipsoid proxy if GSD is supplied;
- explicit proxy label.

## Equations

For a calibrated mask with GSD \(s\) cm/pixel:

```text
L_cm = L_px * s
W_cm = W_px * s
G_cm ~= pi * W_cm
V_ellipsoid = (4/3) * pi * (L_cm/2) * (W_cm/2) * (W_cm/2)
```

For multi-view clusters:

```text
C = median({C_i})
support = number_of_views(C)
reprojection_error = mean(|| project(C, camera_i) - bbox_center_i ||_2)
confidence = f(mask_quality, support, reprojection_error, visibility, scale_available)
```

## Minimum Paper Tables

| Table | Purpose |
|---|---|
| Dataset statistics | pre/post image counts, selected subsets, metadata availability |
| Measurement-ready candidates | pre vs post candidate count at confidence thresholds |
| Reconstruction success | COLMAP/learned method registration and point statistics |
| Multi-view boll association | 2D detections vs 3D clusters and duplicate reduction |
| Trait estimation | length/girth/volume proxy or calibrated error if manual measures exist |
| Failure cases | occlusion, touching bolls, white residue, weak overlap |

## The Core Claim

The strongest claim is:

> Defoliation is a visibility intervention that increases the number of cotton
> bolls that are not only countable, but measurement-ready for local
> multi-view 3D phenotyping.

This gives the paper a clear contribution even if full-field boll volume is not
yet possible.

## What To Avoid

Avoid claiming:

- full-field true 3D boll reconstruction from a single orthomosaic;
- bale estimation from proxy volume;
- generated or diffusion-refined boll geometry as measurement;
- cm/cm3 values without GSD, GCP, or manual scale validation.

## Next Code To Build

1. `algorithms/mine_measurement_ready_bolls.py`
   - scan all pre/post images;
   - produce candidate CSV and top crops.
2. `algorithms/build_local_reconstruction_subsets.py`
   - create 20/60-frame pre/post folders for COLMAP/NAS3R tests.
3. `pipeline/reconstruction/run_colmap_local_subset.py`
   - run COLMAP if installed or write command manifests.
4. `algorithms/associate_bolls_multiview.py`
   - prepare the 2D-to-3D cluster interface after poses exist.

The first two scripts are enough to start experiments immediately.
