# 1. Introduction

Cotton (*Gossypium hirsutum* L.) is the world's most important natural fiber crop, with global production exceeding 25 million metric tons annually. Precision management of cotton requires accurate, timely assessment of boll development — the primary determinant of lint yield and fiber quality. Key management decisions, including plant growth regulator (PGR) application timing, harvest aid scheduling, and water stress intervention, depend critically on quantitative boll morphology: diameter, maturity stage, spatial distribution across the canopy, and growth rate trajectories.

Unmanned aerial vehicle (UAV) imagery has emerged as the dominant modality for high-throughput crop phenotyping, offering spatial resolution sufficient for organ-level analysis at field scale. The standard computational pipeline for three-dimensional reconstruction from UAV imagery is Structure-from-Motion (SfM) followed by Multi-View Stereo (MVS), typically implemented through COLMAP or commercial photogrammetry software. However, cotton presents a set of domain-specific challenges that systematically degrade SfM performance:

1. **Textureless surfaces.** Open cotton bolls exhibit near-uniform white reflectance with minimal local texture gradients. Classical feature descriptors (SIFT, ORB, SuperPoint) rely on repeatable local extrema in image intensity, which are sparse or absent on cotton lint surfaces. This results in failed or ambiguous feature matching across views.

2. **Wind-induced temporal inconsistency.** Cotton plants are structurally flexible; even moderate wind displaces boll positions by several centimeters between successive UAV image captures. SfM assumes rigid scene geometry between overlapping frames — an assumption violated by wind-driven canopy motion.

3. **Leaf occlusion.** Pre-defoliation cotton canopies present dense foliar cover that occludes 40–70% of the boll population from nadir UAV viewpoints. Feature correspondences established on leaf surfaces do not transfer to the underlying boll structures, creating systematic reconstruction gaps.

4. **Photometric ambiguity.** SfM optimizes photometric consistency (pixel-level intensity matching), not semantic consistency (matching corresponding object parts). On cotton, photometric matching may erroneously associate visually similar but geometrically distinct boll surfaces, introducing noise into the reconstructed point cloud.

These limitations motivate a fundamental question: *Can semantic correspondence fields replace pixel-level feature matching to enable 3D reconstruction on surfaces where classical photogrammetry fails?*

## 1.1 Semantic Feature Fields as an Alternative to Pixel Matching

Recent advances in self-supervised vision foundation models, particularly DINOv2, have demonstrated that dense, semantically meaningful feature representations can be extracted from arbitrary images without task-specific training. DINOv2 patch embeddings encode both local appearance and global semantic context, enabling correspondences between visually dissimilar but semantically equivalent regions across views. This property is precisely what cotton boll reconstruction requires: matching boll surfaces across viewpoints based on *what they are* rather than *how they look pixel-by-pixel*.

We propose constructing dense semantic feature fields across multi-view UAV imagery using frozen DINOv2 embeddings, and establishing multi-view correspondences in embedding space rather than pixel space. This approach offers three advantages over classical SfM:

- **Texture invariance.** Semantic embeddings remain discriminative even on textureless surfaces, as they encode structural and contextual information beyond local gradients.
- **Occlusion robustness.** Features that encode "cotton boll" semantics persist even when the boll is partially occluded by leaves, enabling partial-view matching.
- **Wind resilience.** Semantic identity is preserved under small geometric displacements, unlike pixel-level descriptors that are sensitive to exact spatial location.

## 1.2 Defoliation as a Natural Visibility Intervention

Chemical defoliation — the application of harvest aids to induce leaf abscission — is a standard agronomic practice performed 7–14 days before mechanical harvest. From a computer vision perspective, defoliation constitutes a controlled visibility intervention: it removes the primary source of occlusion (leaves) without altering the target structures (bolls). This creates a natural paired experiment:

- **Pre-defoliation:** Dense canopy, severe occlusion, challenging for both human assessment and computational reconstruction.
- **Post-defoliation:** Exposed boll architecture, minimal occlusion, favorable reconstruction conditions.

By capturing UAV imagery of the same field immediately before and after defoliation (within a single day), we obtain paired datasets that isolate the effect of occlusion on reconstruction quality. This experimental design enables rigorous quantification of how leaf occlusion degrades 3D reconstruction, and how semantic correspondence fields mitigate this degradation.

## 1.3 Agronomist-in-the-Loop Reasoning

Even accurate 3D morphological measurements are only valuable if they translate into actionable management decisions. We close the loop from geometric signals to agronomic recommendations by deploying a large language model (LLM) that receives structured boll morphology reports (diameter distributions, growth rates, maturity indices, visibility scores) and generates management recommendations. We benchmark both frontier cloud models (Gemini 2.5 Pro, GPT-4.1) and open-weight alternatives (GLM-4-9B, Gemma 3, Qwen2.5-VL) across multiple input modalities — text-only reports, UAV image frames, and structured morphology context graphs — to establish the cost-accuracy frontier for "agronomist-in-the-loop" reasoning.

## 1.4 Contributions

This paper makes the following contributions:

1. **Semantic correspondence fields for crop 3D reconstruction.** We demonstrate that DINOv2-based semantic correspondences outperform classical feature matching (SuperPoint+SuperGlue) on textureless cotton boll surfaces, enabling reconstruction where pixel-level methods fail.

2. **Defoliation as visibility intervention.** We formalize chemical defoliation as a controlled occlusion removal experiment and provide the first quantitative analysis of its impact on 3D reconstruction completeness and morphological measurement accuracy.

3. **End-to-end phenotyping pipeline.** We present a complete pipeline from raw UAV imagery to agronomic recommendations, integrating DINOv2 feature extraction, SAM 2 instance segmentation, semantic 3D reconstruction, morphological measurement, and LLM-based reasoning.

4. **Frontier and open-weight LLM reasoning.** We benchmark frontier models (Gemini 2.5 Pro, GPT-4.1, Claude) and open-weight models (GLM-4-9B, Gemma 3, Phi-4, Qwen2.5-VL, LLaVA-Video) for structured agronomic reasoning across input modalities (text, frames, morphology graphs), establishing the cost-accuracy Pareto frontier for phenotyping-to-recommendation pipelines.

5. **Paired pre/post-defoliation UAV benchmark.** We release evaluation metrics on 1,549 UAV images spanning both canopy conditions, establishing a benchmark for future work in visibility-aware 3D crop phenotyping.
