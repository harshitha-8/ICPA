# 2. Related Work

## 2.1 3D Plant Phenotyping from UAV Imagery

UAV-based 3D reconstruction for crop phenotyping has been extensively studied using SfM-MVS pipelines. Established software platforms including Agisoft Metashape, Pix4D, and the open-source COLMAP have been applied to generate point clouds and digital surface models (DSMs) for estimating plant height, canopy volume, and biomass across crops including wheat, maize, sorghum, and cotton.

For cotton specifically, prior work has demonstrated UAV-derived DSMs for canopy height estimation and plant counting. However, organ-level phenotyping — particularly boll-level measurement — has remained challenging due to the textural and structural properties described in Section 1. Recent work has employed 3D Gaussian Splatting (3DGS) for cotton boll visualization, reporting improved detail capture compared to traditional point clouds, and combining YOLOv8 detection with SAGA for instance-level boll analysis in 3D space. Neural Radiance Fields (NeRF) have also been explored for high-precision plant modeling, but computational cost limits field-scale applicability.

**Gap:** Existing approaches rely on photometric feature matching, which degrades on textureless cotton lint. No prior work has applied semantic feature fields from foundation models to crop 3D reconstruction. The defoliation-induced visibility change has not been formally exploited as an experimental condition.

## 2.2 Vision Foundation Models for Dense Feature Extraction

Self-supervised vision transformers, particularly DINOv2, have established new baselines for dense feature extraction. DINOv2 produces patch-level embeddings that are discriminative for both instance and semantic recognition without task-specific fine-tuning. These features have been shown to enable robust correspondence across views with significant appearance variation, including different lighting, viewpoints, and partial occlusion.

Recent developments directly relevant to our work include:

- **NeCo (Patch Neighbor Consistency, ICLR 2025):** Improves spatial consistency of DINOv2 features via self-supervised post-training, demonstrating significant gains in multi-view correspondence accuracy.
- **H3R (2025):** Explores trade-offs between "semantic-aligned" models (DINOv2) and "spatial-aligned" models (SD-VAE), suggesting hybrid approaches for geometric reconstruction.
- **dino.txt (2025):** Aligns DINOv2 features with language embeddings, enabling open-vocabulary 3D scene understanding.

In 3D reconstruction, DINOv2 features have been integrated into depth estimation models and multi-view stereo architectures, providing robustness to out-of-distribution data and texture-sparse regions. However, application to agricultural imagery — where "texture-sparse" is the dominant condition rather than an edge case — remains unexplored.

**Gap:** DINOv2 features have not been systematically evaluated for agricultural 3D reconstruction, where textureless surfaces are the norm rather than the exception. The correspondence stability of foundation model features under biological occlusion (leaves) has not been characterized.

## 2.3 Zero-Shot Segmentation in Agriculture

The Segment Anything Model (SAM) and its successor SAM 2 have demonstrated strong zero-shot segmentation performance on agricultural imagery. SAM 2 introduces a streaming memory module that enables consistent tracking and segmentation across video frames, which is directly applicable to temporally ordered UAV image sequences.

In agricultural contexts, SAM 2 has been paired with lightweight detectors (YOLOv8, YOLOv7) to segment cotton bolls, where the detector provides bounding box or point prompts and SAM 2 generates high-fidelity masks. Comparative studies indicate SAM 2 outperforms traditional architectures (U-Net, Mask R-CNN) in agricultural settings due to superior handling of complex backgrounds, occlusion, and variable illumination.

Adaptive frameworks such as ASAMPS have been developed to optimize prompt points for agricultural tasks without full model retraining, and deployment on edge devices (NVIDIA Jetson) using TensorRT has been demonstrated for real-time field processing.

**Gap:** SAM 2 segmentation quality has not been compared across defoliation states in a controlled setting. Temporal consistency of boll instance masks across multi-view UAV sequences (as opposed to true video) has not been quantified. Integration of SAM 2 masks with semantic feature fields for 3D-aware segmentation remains unexplored.

## 2.4 Neural Feature Fields and Semantic 3D Reconstruction

The integration of semantic features into 3D scene representations has advanced rapidly:

- **Feature 3DGS (CVPR 2024):** Distills 2D semantic features (from CLIP, LSeg, SAM) into 3D Gaussian primitives, enabling promptable open-vocabulary 3D segmentation.
- **SemanticSplat (2025):** Extends feature distillation to maintain multi-view consistency of semantic features in 3DGS representations.
- **DUSt3R and MASt3R (CVPR/ECCV 2024):** Transformer-based end-to-end 3D reconstruction treating geometry estimation as a dense matching problem, eliminating the need for classical SfM initialization.

These methods establish that 3D representations can encode semantic information beyond RGB appearance. However, they are designed for indoor/urban scenes and have not been adapted for agricultural environments where:
(a) scene scale is much larger (field vs. room),
(b) target objects (bolls) are small relative to scene extent,
(c) biological variability introduces distribution shifts across the field.

**Gap:** Neural feature field methods have not been applied to agricultural phenotyping. The concept of semantic bundle adjustment — where optimization operates in embedding space rather than pixel space — has not been formalized for crop reconstruction.

## 2.5 LLMs for Agricultural Decision Support

Large language models are increasingly being adapted for agricultural applications. Domain-specific systems including AgriGPT, AgroLLM, and AgriLLM have been developed with specialized training data for agricultural terminology and reasoning. These systems support decision-making on irrigation scheduling, pest management, and nutrient application.

For edge deployment, the GLM-4-9B series offers a favorable parameter-performance tradeoff, with recent "Thinking" variants incorporating chain-of-thought reasoning. Models in the 4–12B parameter range can be quantized to 4-bit and deployed on hardware including Apple M-series processors and NVIDIA Jetson Orin platforms.

**Gap:** No existing work connects 3D morphological measurements to LLM-based agronomic reasoning. The reliability of LLM-generated management recommendations when conditioned on quantitative phenotypic data (rather than natural language queries) has not been evaluated. Edge deployment feasibility for field-level phenotyping-to-recommendation pipelines remains uncharacterized.

## 2.6 Summary of Research Gaps

| Gap | Status | Our Contribution |
|-----|--------|-------------------|
| Semantic feature matching for crop 3D reconstruction | Unexplored | DINOv2-based correspondence pipeline |
| Defoliation as visibility intervention experiment | Unexploited | Formal paired comparison framework |
| SAM 2 segmentation stability across defoliation states | Unquantified | Temporal consistency metrics |
| Feature-aligned (semantic) bundle adjustment | Not formalized for agriculture | Semantic BA with embedding-space optimization |
| 3D morphology → LLM agronomic reasoning | No integration exists | End-to-end pipeline with edge deployment |
| Edge LLM for structured phenotypic reasoning | Unbenchmarked | Comparative evaluation of 9B-class models |
