# Architecture Specification: Mask-Guided Cotton Boll Phenotyping

## One-Paragraph Overview

The proposed system studies UAV RGB imagery of cotton before and after defoliation and treats the phase change as a controlled visibility intervention rather than a simple dataset split. The current implementation is a claim-bounded MVP: it detects bright cotton lint candidates, extracts detector-guided lint masks, ranks candidates by measurement readiness, constructs a morphology-aware 2.5D depth proxy for visual review, estimates proxy boll traits, and aggregates records over a plot grid. The architecture deliberately separates implemented evidence from future calibrated reconstruction: official SAM/SAM2, DINO-style foundation features, SfM/MVS, DUSt3R/MASt3R/VGGT, and 3D Gaussian Splatting are useful insertion points, but they are shown as optional or calibration-dependent modules unless those components are actually run and validated with camera scale, GCPs, or physical boll measurements.

## Ordered Pipeline

1. Load UAV RGB frames from pre-defoliation and post-defoliation folders.
2. Resolve phase from folder labels or infer it from canopy greenness when folder labels are unavailable.
3. Normalize illumination with CLAHE and enhance bright lint structures with multi-scale top-hat morphology.
4. Generate candidate boll boxes using Otsu thresholding, contour filtering, HSV saturation/value gates, and phase-aware heuristics.
5. Use each candidate box as a prompt for SAM-style lint mask extraction; retain the dominant lint component after color and morphology cleanup.
6. Score each candidate using lint fraction, visibility, brightness, shape regularity, size prior, green penalty, and phase evidence.
7. Project selected mask/crop evidence into a morphology-aware monocular 2.5D review space.
8. Estimate proxy traits: count, visibility, mask length, mask width, diameter proxy, ellipsoid volume proxy, and confidence.
9. Aggregate candidate records over a 4 by 43 plot-cell grid.
10. Evaluate detection, masks, proxy traits, robustness, ablations, and optional agronomist-facing report quality.

## Blocks / Modules

- **Input data:** UAV RGB imagery from pre- and post-defoliation cotton plots.
- **Phase resolver:** folder-name phase or ExG-style greenness heuristic.
- **Candidate detector:** CLAHE, multi-scale top-hat filtering, Otsu thresholding, contour filtering, HSV gates.
- **Mask extractor:** detector-box-prompted lint mask extraction; official SAM/SAM2 is a drop-in optional module.
- **Readiness scorer:** rule-based fusion of mask, color, shape, size, phase, and occlusion cues.
- **Proxy geometry module:** morphology-aware monocular 2.5D depth proxy and local crop review.
- **Optional calibrated geometry branch:** GSD/GCP/camera intrinsics plus SfM/MVS, DUSt3R/MASt3R/VGGT, or 3DGS.
- **Trait estimator:** count, visibility, length, width, diameter proxy, ellipsoid volume proxy, confidence.
- **Plot aggregator:** 4 by 43 row-column summaries in image coordinates.
- **Optional decision layer:** structured agronomic summaries from trait records; the LLM does not perform reconstruction.

## Arrows / Data Flow

- Solid arrows denote the implemented proxy pipeline.
- Dashed arrows denote optional, future, or calibration-dependent components.
- The phase resolver conditions detection and scoring.
- The detector provides boxes to the mask extractor.
- Masks and candidate boxes feed the readiness scorer and trait estimator.
- The proxy depth module produces review geometry; calibrated geometry modules can replace or validate it.
- Trait records feed plot aggregation, evaluation, and optional reporting.
- Evaluation returns to thresholds, ablations, and calibration decisions; it does not imply end-to-end training in the current MVP.

## Training Losses / Signals

- **Current MVP:** no end-to-end learned training loss is claimed.
- **Current scoring signals:** lint fraction, visibility, brightness, shape regularity, size prior, green penalty, and phase.
- **If supervised labels are added:** detection loss, mask BCE/Dice loss, trait MAE/Huber loss, reprojection or Chamfer loss for calibrated 3D, and structured-output validity for the reporting layer.
- **Validation metrics:** count MAE/RMSE/F1, mask IoU/boundary F1, length/width/volume error after physical calibration, reprojection consistency, Chamfer distance, plot-cell count error, schema validity, expert agreement, hallucination rate, and latency.

## Figure Labels

- **Fig. X(a):** pre/post UAV imagery as the input visibility intervention.
- **Fig. X(b):** phase-aware candidate localization and detector-box prompting.
- **Fig. X(c):** lint mask extraction and measurement-readiness scoring.
- **Fig. X(d):** morphology depth proxy, with a dashed calibrated-geometry alternative.
- **Fig. X(e):** trait estimation, plot-cell aggregation, evaluation, and optional agronomic report.

## Proposed Caption

**Figure X. Mask-guided cotton boll phenotyping architecture.** UAV RGB frames acquired before and after defoliation are processed by a phase-aware detector that localizes bright lint candidates using illumination normalization, morphology, thresholding, and color gates. Candidate boxes prompt a SAM-style lint mask extractor, after which a measurement-readiness score combines lint fraction, visibility, brightness, shape, size, greenness, and phase cues. The current implementation projects selected evidence into a morphology-aware 2.5D review space and estimates proxy traits including count, length, width, diameter, volume, and confidence. Plot-cell aggregation summarizes records over the field grid. Solid arrows denote the implemented proxy pipeline; dashed arrows denote optional or calibration-dependent modules such as official SAM/SAM2, DINO-style features, SfM/MVS, DUSt3R/MASt3R/VGGT, and Gaussian Splatting. Metric 3D traits require camera calibration, scale information, and physical validation.

## Image-Informed Design Notes

- The clean modular CVPR/NeurIPS architecture screenshots motivated the left-to-right data flow, compact module boxes, and sparse explanatory labels.
- The MoSA attention diagram motivated the solid-versus-dashed claim separation and small side notes for implementation boundaries.
- The VGGT-style reconstruction figure motivated the split between a current proxy geometry lane and an optional foundation-geometry lane.
- The cotton point-cloud and boll-volume screenshots motivated the output blocks for mask-to-geometry review, plot-grid aggregation, volume proxy, ablations, and validation metrics.
- The SAM gallery reference motivated the detector-box-to-mask prompt block.
- The Utonia, Stream3R, NAS3R, SPZ, Skyfall-GS, and forge3d references influenced the optional calibrated/future 3D branch, but they are not shown as completed components.

## Unknowns / Questions

- [UNKNOWN] Camera intrinsics, calibrated GSD, GCPs, and camera poses are not confirmed.
- [UNKNOWN] Ground-truth physical boll length, diameter, and volume measurements are not confirmed.
- [UNKNOWN] Official SAM/SAM2, DINOv2, DUSt3R/MASt3R/VGGT, COLMAP/SfM, or 3DGS have not been confirmed as fully run and validated for the current paper results.
- [UNKNOWN] No supervised training set for end-to-end detector/mask/trait learning is confirmed.
