# Paper USP And Execution Plan

## Core Constraint

Two close papers define the boundary:

1. The ISPRS 2024 UAV cotton paper already shows field-scale cotton boll point-cloud reconstruction and links boll traits to yield.
2. Cotton3DGaussians already shows single-plant 3D Gaussian Splatting, 2D mask projection into 3D, cross-view duplicate removal, and boll volume estimation.

Therefore, this paper must not sound like either paper repeated with a different dataset. The novelty must be narrower, clearer, and testable.

## Proposed USP

**Detection-guided semantic 3D cotton boll phenotyping from paired pre- and post-defoliation UAV imagery, with visibility-aware boll measurement and an evaluated agronomist/MoE reporting loop.**

This combines four pieces that the near papers do not jointly cover:

1. **Paired pre/post-defoliation visibility intervention.** Defoliation is treated as an experimental intervention, not only a condition label.
2. **Field-scale UAV setting.** The work is not limited to a single plant scanned by smartphone.
3. **Detection-guided semantic multi-view association.** Prior boll detections guide 3D association and reduce annotation burden.
4. **Measurement uncertainty and reporting.** Boll length/diameter/volume estimates are reported with confidence/visibility, and an MoE/LLM layer is evaluated only after geometry.

## What Measurements Are Plausible

The strongest realistic measurement set is:

| Trait | Can it be attempted now? | Notes |
|---|---|---|
| Visible boll count | Yes | Already running from inherited detector; must report raw and adjusted counts. |
| 3D boll center | Yes, after camera poses | Requires COLMAP/VGGT/MASt3R pose or depth estimates. |
| Boll diameter/width | Plausible | Estimate from projected masks plus scale; report only high-confidence bolls. |
| Boll length | Plausible but riskier | Cotton bolls are irregular and clustered; length should be reported with uncertainty and only for isolated/high-view-support instances. |
| Boll volume | Plausible as proxy | Use ellipsoid/convex-hull/splat occupancy proxy; validate against manual subset if possible. |
| Visibility/occlusion | Strong | Count number of supporting views, mask area consistency, and ray/point density. |
| Pre/post change | Strong | Compare visible-count, support views, occlusion, and recoverable morphology between phases. |

The paper should not claim whole-field perfect boll diameter or volume. It should claim **high-confidence morphology for reconstructed boll instances** and show coverage statistics.

## Minimum Experiment That Can Win

1. Run phase-aware counting over all images.
2. Select a compact pre/post subset with strong overlap and high sharpness.
3. Run at least one real reconstruction method: COLMAP if available, or VGGT/MASt3R if COLMAP cannot register enough images.
4. Project detector/SAM masks into the reconstructed space.
5. Cluster projected detections into 3D boll instances.
6. Estimate diameter/length/volume only for bolls with enough views and low mask variance.
7. Compare pre vs post:
   - visible candidate count,
   - 3D recoverable boll count,
   - mean supporting views per boll,
   - occlusion/visibility score,
   - diameter/volume distribution for high-confidence bolls.

## Novel Table Structure

| Table | Purpose | Must include |
|---|---|---|
| Table 1 | Dataset and phase summary | folders, image count, pre/post, raw/adjusted detections |
| Table 2 | Closest-prior comparison | ISPRS UAV 2024, Cotton3DGaussians 2025, this work |
| Table 3 | Reconstruction robustness | COLMAP, 3DGS scaffold/real 3DGS, VGGT/MASt3R if used |
| Table 4 | Boll instance recovery | 2D detections, projected masks, 3D clustered bolls |
| Table 5 | Morphology measurement | diameter, length, volume, visibility, view support |
| Table 6 | MoE/LLM reporting ablation | Qwen dense, Qwen MoE, Mixtral/DeepSeek MoE, AgriLLaMA, VLM |

## Benchmark-Style Evaluation Add-On

HY3D-Bench is useful as an evaluation-design reference because it treats 3D
outputs as structured assets rather than screenshots. We should borrow that
mindset without pretending HY3D-Bench is an agriculture baseline.

Add one robustness table or appendix table with:

| Axis | Cotton metric | Why it matters |
|---|---|---|
| Geometry validity | connected components, point density, optional non-manifold/hole count | prevents pretty but unusable reconstructions |
| Sampled point quality | Chamfer/F-score when reference scans or repeated reconstructions exist | standardizes point-cloud comparison |
| Multi-view consistency | held-out reprojection error, mask IoU across views | shows the 3D boll is not a single-view hallucination |
| Part/organ structure | boll/canopy/soil/branch separability | adapts part-level 3D evaluation to crop organs |
| Long-tail robustness | pre/post, dense canopy, exposed lint, shadowed rows | shows the method survives field variability |

## Closest-Prior Contrast Paragraph

The manuscript can use this positioning:

> Prior UAV cotton reconstruction has demonstrated that cross-circling oblique imagery can recover boll-level point clouds and improve boll/yield estimation, while recent Cotton3DGaussians work has shown that 3D Gaussian Splatting and multiview mask projection can characterize individual plants from close-range 360-degree image capture. The present study addresses a different gap: paired pre- and post-defoliation UAV imagery at field scale. In this setting, defoliation is used as a visibility intervention, and detection-guided semantic association is used to quantify which boll traits become recoverable in 3D. The reported measurements are therefore not only count estimates, but visibility-conditioned morphology estimates with uncertainty.

## MoE/LLM USP

The MoE layer should be framed as **agronomist-in-the-loop reporting**, not as reconstruction.

The input to the LLM is a fixed morphology JSON. The output is a constrained recommendation JSON. The ablation tests:

- dense general reasoning: Qwen3-32B,
- sparse MoE reasoning: Qwen3-30B-A3B or Mixtral,
- agriculture-domain language: AgriLLaMA,
- multimodal reasoning: Gemma/Mistral/InternVL,
- reasoning upper bound: DeepSeek MoE if compute is available.

Metrics:

- schema validity,
- measurement faithfulness,
- unsupported-claim rate,
- uncertainty quality,
- latency,
- expert alignment.

This becomes a USP because reviewers can see that the LLM layer is evaluated rigorously and cannot fabricate measurements.

## Risk Control

| Risk | Mitigation |
|---|---|
| COLMAP fails on cotton lint | Use VGGT/MASt3R or limited high-overlap subsets; report registration rate as a result. |
| Scale is uncertain | Use EXIF altitude/GSD, field markers, or manual scale subset; report scale uncertainty. |
| Diameter/length not reliable for all bolls | Report high-confidence subset plus coverage percentage. |
| 3DGS is expensive | Use 3DGS for a small subset and point cloud/VGGT for broader evaluation. |
| Near papers are too close | Make defoliation visibility and semantic association the main contribution, not generic 3D reconstruction. |
| LLM looks like gimmick | Keep it post-measurement and evaluate hallucination/schema/expert alignment. |

## Final Contribution Wording

1. A paired pre/post-defoliation UAV dataset protocol for visibility-aware cotton boll 3D phenotyping.
2. A detection-guided semantic multi-view association pipeline for reducing dense manual annotation.
3. A high-confidence boll morphology module estimating 3D center, diameter/length proxy, volume proxy, visibility, and occlusion.
4. A field-scale reconstruction viewer with semantic boll anchors for inspection and supplementary evidence.
5. An evaluated MoE/LLM agronomic reporting layer that converts measured morphology into structured recommendations under hallucination and schema checks.
