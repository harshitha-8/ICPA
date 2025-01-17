# 3. Methods

We present a five-stage pipeline for semantic 3D reconstruction of cotton boll morphology from UAV imagery. Figure 1 provides an overview; each stage is detailed below.

## 3.1 Problem Formulation

Let $\mathcal{I}_\text{pre} = \{I_1, \ldots, I_M\}$ and $\mathcal{I}_\text{post} = \{I_1, \ldots, I_N\}$ denote sets of UAV images captured before and after chemical defoliation of a cotton field, where $M = 680$ and $N = 869$. Each image $I_k \in \mathbb{R}^{H \times W \times 3}$ is associated with camera intrinsics $K_k$ and an unknown extrinsic pose $[R_k | t_k] \in SE(3)$.

The objective is to reconstruct a 3D point cloud $\mathcal{P} = \{(p_i, s_i, f_i)\}$ where each point $p_i \in \mathbb{R}^3$ carries a semantic label $s_i$ (boll, leaf, stem, soil) and a feature vector $f_i \in \mathbb{R}^d$ from DINOv2 embeddings, then extract morphological measurements $\mathcal{M} = \{m_j\}$ for each boll instance $j$, and finally synthesize agronomic recommendations via LLM reasoning.

## 3.2 Stage 1: Semantic Feature Field Construction

### 3.2.1 Dense Feature Extraction

For each image $I_k$, we extract a dense feature map $F_k \in \mathbb{R}^{h \times w \times d}$ using a frozen DINOv2 ViT-L/14 backbone:

$$F_k = \text{DINOv2}(I_k)$$

where $h = H/14$, $w = W/14$ (patch size 14), and $d = 1024$ (ViT-L embedding dimension). Each spatial position $(u, v)$ in $F_k$ encodes a semantically rich representation of the corresponding $14 \times 14$ pixel patch in $I_k$.

### 3.2.2 Semantic Correspondence Matching

Given an image pair $(I_a, I_b)$, we establish dense correspondences by computing the cosine similarity between all patch embedding pairs:

$$S_{ab}(u_a, v_a, u_b, v_b) = \frac{F_a(u_a, v_a)^\top F_b(u_b, v_b)}{\|F_a(u_a, v_a)\| \cdot \|F_b(u_b, v_b)\|}$$

Correspondences are established via mutual nearest neighbor matching in embedding space with a similarity threshold $\tau$:

$$\mathcal{C}_{ab} = \{((u_a, v_a), (u_b, v_b)) : S_{ab} > \tau \wedge \text{mutual\_nn}(u_a, v_a, u_b, v_b)\}$$

We apply RANSAC-based geometric verification to filter outliers, yielding inlier correspondences $\mathcal{C}_{ab}^*$.

### 3.2.3 Comparison with RGB Feature Matching

As a baseline, we extract correspondences using SuperPoint for keypoint detection and SuperGlue for matching. We quantify:

- **Correspondence Stability Index (CSI):** For a set of $K$ overlapping views, CSI measures the fraction of feature matches that persist across $\geq k$ views:
  $$\text{CSI}(k) = \frac{|\{c \in \mathcal{C} : \text{views}(c) \geq k\}|}{|\mathcal{C}|}$$

- **Feature Consistency Ratio (FCR):** Variance of embedding similarity for corresponding points across multiple views:
  $$\text{FCR} = 1 - \text{Var}_{(a,b) \in \text{pairs}} [S_{ab}(c)]$$

## 3.3 Stage 2: Zero-Shot Instance Segmentation

### 3.3.1 SAM 2 Automatic Segmentation

We apply SAM 2 with the ViT-H backbone in automatic mask generation mode. For each image $I_k$, a grid of $32 \times 32$ prompt points generates candidate masks:

$$\mathcal{M}_k = \text{SAM2}(I_k, \text{grid}_{32 \times 32})$$

Masks are filtered by predicted IoU ($> 0.86$), stability score ($> 0.92$), and minimum area ($> 100$ pixels). Remaining masks are classified as boll, leaf, stem, or background using DINOv2 feature similarity to prototype embeddings.

### 3.3.2 Semantic-Guided Prompting

To improve boll detection, we augment automatic segmentation with DINOv2-guided prompts. We identify image regions whose embeddings have high cosine similarity to a learned boll prototype $f_\text{boll}$:

$$\text{boll\_score}(u, v) = \frac{F_k(u, v)^\top f_\text{boll}}{\|F_k(u, v)\| \cdot \|f_\text{boll}\|}$$

Peaks in the boll score map ($> 0.8$) serve as point prompts to SAM 2.

### 3.3.3 Temporal Instance Tracking

Across temporally ordered image pairs $(I_k, I_{k+1})$, boll instances are associated using a combined IoU and feature similarity criterion:

$$\text{match}(m_a, m_b) = \alpha \cdot \text{IoU}(m_a, m_b) + (1 - \alpha) \cdot \text{sim}(f_{m_a}, f_{m_b})$$

where $\alpha = 0.5$, $f_{m_a}$ is the mean DINOv2 embedding within mask $m_a$, and associations are resolved via the Hungarian algorithm.

### 3.3.4 Segmentation Metrics

- **Mask Stability Score (MSS):** Mean IoU of SAM 2 masks under small perturbations of confidence threshold.
- **Instance Retention Rate (IRR):** Fraction of boll instances tracked across $\geq 3$ consecutive frames.
- **Boundary Accuracy:** IoU of predicted masks against manual annotations on a 50-image validation subset.

## 3.4 Stage 3: Semantic 3D Reconstruction

### 3.4.1 Baseline: COLMAP SfM + MVS

We establish a photometric baseline using COLMAP with default parameters for feature extraction (SIFT), exhaustive matching, incremental SfM, and dense reconstruction via patch-match MVS.

### 3.4.2 Proposed: Feature-Aligned Reconstruction

Our proposed reconstruction replaces pixel-level matching with semantic correspondences from Stage 1. The pipeline operates in three steps:

**Step 1: Semantic SfM.** Camera poses are estimated using DINOv2 correspondences $\mathcal{C}_{ab}^*$ (upscaled to pixel coordinates) as input to an incremental bundle adjustment solver:

$$\min_{R_k, t_k, p_i} \sum_{(a,b)} \sum_{c \in \mathcal{C}_{ab}^*} \|\pi(R_a, t_a, p_c) - x_a^c\|^2$$

where $\pi(\cdot)$ denotes the camera projection function and $x_a^c$ is the observed 2D location of correspondence $c$ in image $I_a$.

**Step 2: Semantic Dense Matching.** For each pair of images with known relative pose, we compute a cost volume in embedding space rather than pixel space. The matching cost at depth hypothesis $d$ for pixel $x$ in image $I_a$ is:

$$C(x, d) = 1 - S_{ab}(F_a(x), F_b(\pi_b(K_b [R_b | t_b] K_a^{-1} \bar{x} \cdot d)))$$

where $\bar{x}$ is the homogeneous coordinate of $x$.

**Step 3: Feature-Metric Refinement.** We refine the dense point cloud by minimizing a combined photometric and semantic loss:

$$\mathcal{L} = \lambda_\text{geo} \mathcal{L}_\text{reproj} + \lambda_\text{sem} \mathcal{L}_\text{semantic} + \lambda_\text{reg} \mathcal{L}_\text{smooth}$$

where $\mathcal{L}_\text{semantic}$ penalizes embedding inconsistency across views for the same 3D point, and $\lambda_\text{geo} = 0.7$, $\lambda_\text{sem} = 0.3$.

### 3.4.3 Reconstruction Metrics

- **Reconstruction Completeness (RC):** Fraction of the field area (defined by GPS hull) covered by the point cloud at voxel resolution $v$:
  $$\text{RC} = \frac{|\text{occupied\_voxels}|}{|\text{total\_field\_voxels}|}$$

- **Boll Retention Rate (BRR):** Ratio of bolls present in 3D reconstruction to bolls detected in 2D imagery:
  $$\text{BRR} = \frac{|\text{3D\_bolls}|}{|\text{2D\_bolls}|}$$

- **Geometric Noise Level:** Mean nearest-neighbor distance in the point cloud.

## 3.5 Stage 4: Morphological Measurement Extraction

For each reconstructed boll instance $j$ (identified by projecting SAM 2 masks into 3D), we extract:

- **Boll diameter $d_j$:** Principal axis length of the oriented bounding box fitted to the boll point cluster.
- **Equatorial girth $g_j$:** Circumference of the maximum cross-section perpendicular to the principal axis.
- **Volume $V_j$:** Convex hull volume of the boll point cluster.
- **Surface curvature $\kappa_j$:** Mean Gaussian curvature estimated via local PCA on point neighborhoods.
- **Visibility score $\nu_j$:** Fraction of the boll convex hull surface visible from at least one UAV camera position via ray casting.
- **Occlusion index $o_j$:** $o_j = 1 - \nu_j$.

Statistical comparison between pre- and post-defoliation morphological distributions uses paired t-tests with Bonferroni correction.

## 3.6 Stage 5: Agronomist-in-the-Loop Reasoning

### 3.6.1 Input Representation

Morphological measurements are aggregated into a structured report:

```json
{
  "field_id": "TX-2025-001",
  "condition": "post_defoliation",
  "boll_count": 1247,
  "mean_diameter_mm": 28.3,
  "diameter_cv": 0.18,
  "maturity_index": 0.72,
  "growth_stagnation_pct": 12.4,
  "visibility_improvement": 0.34,
  "canopy_height_distribution": {"lower": 0.3, "middle": 0.5, "upper": 0.2}
}
```

### 3.6.2 Prompt Engineering

The LLM receives a system prompt encoding agronomic domain knowledge (cotton growth stages, PGR thresholds, harvest aid decision rules) and a structured user prompt containing the morphological report. The model generates:

1. **Growth assessment:** Current developmental stage and trajectory.
2. **Management recommendations:** Specific PGR, harvest aid, or irrigation actions.
3. **Confidence level:** Self-assessed certainty with reasoning justification.
4. **Risk flags:** Potential issues requiring human expert verification.

### 3.6.3 LLM Evaluation

We evaluate candidate models (GLM-4-9B, Gemma 3, Phi-4) on:

- **Consistency:** Agreement rate across 3 independent runs per input.
- **Expert alignment:** Scored against expert agronomist recommendations on 20 test cases.
- **Hallucination rate:** Proportion of claims not supported by input data.
- **Latency:** Wall-clock inference time on target edge hardware.
- **Structured output compliance:** Rate of valid JSON output matching the expected schema.
