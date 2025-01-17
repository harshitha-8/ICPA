# 4. Experiments

## 4.1 Dataset

We evaluate on 1,549 DJI UAV images of a cotton field captured on September 29, 2025. Pre-defoliation imagery (680 images, 09:57–10:07 local time) captures dense canopy with severe boll occlusion. Post-defoliation imagery (869 images, 12:41–12:51 local time) captures exposed boll structures following chemical defoliant application. All images are RGB at approximately 12 MP resolution (~11 MB each). The ~3-hour interval between flights ensures consistent field geometry while providing maximal contrast in canopy visibility.

## 4.2 Experiment 1: Feature Correspondence Quality

**Objective:** Evaluate feature matching methods on cotton boll surfaces across defoliation conditions.

**Table 1. Feature correspondence comparison across methods and conditions.** Pre- and post-defoliation image pairs evaluated on boll-containing regions. Best results **bold**, second-best underlined.

| Method | Condition | # Matches↑ | Inlier Ratio↑ | CSI(k=3)↑ | FCR↑ | Reproj. Error↓ |
|--------|-----------|-----------|---------------|-----------|------|----------------|
| SIFT | Pre-def | [TBD] | [TBD] | [TBD] | — | [TBD] |
| SIFT | Post-def | [TBD] | [TBD] | [TBD] | — | [TBD] |
| ORB | Pre-def | [TBD] | [TBD] | [TBD] | — | [TBD] |
| ORB | Post-def | [TBD] | [TBD] | [TBD] | — | [TBD] |
| SuperPoint+SuperGlue | Pre-def | [TBD] | [TBD] | [TBD] | — | [TBD] |
| SuperPoint+SuperGlue | Post-def | [TBD] | [TBD] | [TBD] | — | [TBD] |
| DINOv2 ViT-S/14 | Pre-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| DINOv2 ViT-S/14 | Post-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| DINOv2 ViT-B/14 | Pre-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| DINOv2 ViT-B/14 | Post-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **DINOv2 ViT-L/14 (Ours)** | Pre-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **DINOv2 ViT-L/14 (Ours)** | Post-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

**Setup:** 100 randomly sampled consecutive image pairs per condition. Metrics computed on boll-containing regions only (identified by SAM 2 masks). RANSAC threshold = 8.0 px for geometric verification.

## 4.3 Experiment 2: Segmentation Quality Across Defoliation States

**Objective:** Evaluate SAM 2 boll segmentation quality and temporal consistency, comparing automatic vs. semantically-prompted modes.

**Table 2. Instance segmentation evaluation on cotton bolls.** Evaluated on 50 manually annotated images per condition. AP = Average Precision, IRR = Instance Retention Rate across ≥3 consecutive frames.

| Method | Prompting | Condition | AP@50↑ | AP@75↑ | mIoU↑ | MSS↑ | IRR↑ | Boll Count |
|--------|-----------|-----------|--------|--------|-------|------|------|------------|
| Mask R-CNN (R-50) | Supervised | Pre-def | [TBD] | [TBD] | [TBD] | — | — | [TBD] |
| Mask R-CNN (R-50) | Supervised | Post-def | [TBD] | [TBD] | [TBD] | — | — | [TBD] |
| SAM 2 | Auto (32×32) | Pre-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| SAM 2 | Auto (32×32) | Post-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| SAM 2 | Auto (64×64) | Pre-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| SAM 2 | Auto (64×64) | Post-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **SAM 2 + DINOv2 (Ours)** | Semantic | Pre-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **SAM 2 + DINOv2 (Ours)** | Semantic | Post-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

## 4.4 Experiment 3: 3D Reconstruction Quality

**Objective:** Compare reconstruction methods across conditions using completeness, noise, and boll retention.

**Table 3. 3D reconstruction evaluation across methods and defoliation conditions.** RC = Reconstruction Completeness, BRR = Boll Retention Rate, NN = Mean Nearest-Neighbor distance (noise proxy). CD = Chamfer Distance to post-defoliation reference (lower is better).

| Category | Method | Condition | # Registered↑ | # Points (K) | RC↑ | BRR↑ | NN↓ | CD↓ |
|----------|--------|-----------|---------------|-------------|-----|------|-----|-----|
| Classical SfM | COLMAP (SIFT) | Pre-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| | COLMAP (SIFT) | Post-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | — |
| | COLMAP (SP+SG) | Pre-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| | COLMAP (SP+SG) | Post-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | — |
| Foundation | DUSt3R | Pre-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Model | DUSt3R | Post-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | — |
| Ours | Semantic SfM (DINOv2-only) | Pre-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| | Semantic SfM (DINOv2-only) | Post-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | — |
| | **Hybrid (COLMAP + Semantic)** | Pre-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| | **Hybrid (COLMAP + Semantic)** | Post-def | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | — |

## 4.5 Experiment 4: Morphological Measurement Accuracy

**Objective:** Evaluate consistency and reliability of extracted boll morphological measurements across conditions.

**Table 4. Boll morphology statistics under pre- vs post-defoliation conditions.** Cohen's d > 0.8 indicates large effect size. p-values Bonferroni-corrected for 6 comparisons.

| Metric | Pre-def (μ±σ) | Post-def (μ±σ) | Δ(%) | Cohen's d | p-value | Sig. |
|--------|--------------|----------------|------|-----------|---------|------|
| Diameter (mm) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Girth (mm) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Volume (mm³) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Curvature | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Visibility | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Compactness | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

## 4.6 Experiment 5: LLM Agronomic Reasoning Quality

**Objective:** Evaluate vision-language and language models for structured agronomic reasoning from cotton morphology data. Not constrained to edge — cloud-hosted frontier models are included.

**Table 5. LLM reasoning evaluation on 20 cotton morphology test cases.** Consistency = inter-run agreement (3 runs). Schema = JSON output compliance. Hallu. = 1 − hallucination rate. Expert = agreement with agronomist ground truth (n=20). Cost is per-inference.

| Category | Model | Params | Modality | Consistency↑ | Schema↑ | Hallu.-Free↑ | Expert↑ | Latency↓ | Cost↓ |
|----------|-------|--------|----------|-------------|---------|-------------|---------|----------|-------|
| Frontier | Gemini 2.5 Pro | — | F, T | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| (Cloud) | GPT-4.1 | — | T | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| | Claude Opus 4 | — | T | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Open-weight | GLM-4-9B | 9B | T | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | $0 |
| (Local) | Gemma 3 12B | 12B | T | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | $0 |
| | Qwen 2.5-VL 7B | 7B | F, T | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | $0 |
| | Phi-4 | 14B | T | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | $0 |
| | LLaVA-Video 7B | 7B | F, T | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | $0 |
| Ours | **EGAgent (Gemini 2.5 Pro)** | — | F, C, T | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

F = frames (UAV images), C = morphology context graph, T = structured text report.

**Table 6. Ablation: LLM reasoning with different input modalities.** EGX = Entity Graph Extraction from 3D morphology.

| Method | VLM | EGX | # Frames | Input | Accuracy↑ | Gain (%) |
|--------|-----|-----|----------|-------|-----------|----------|
| Direct text | GPT-4.1 | — | 0 | T | [TBD] | baseline |
| Direct text | Gemini 2.5 Pro | — | 0 | T | [TBD] | [TBD] |
| + UAV frames | Gemini 2.5 Pro | — | 50 | F, T | [TBD] | [TBD] |
| + Morph. context | Gemini 2.5 Pro | ✓ | 0 | C, T | [TBD] | [TBD] |
| **+ All modalities** | Gemini 2.5 Pro | ✓ | 50 | F, C, T | [TBD] | [TBD] |
| Direct text | GLM-4-9B | — | 0 | T | [TBD] | [TBD] |
| + Morph. context | GLM-4-9B | ✓ | 0 | C, T | [TBD] | [TBD] |

## 4.7 Experiment 6: End-to-End Pipeline Evaluation

**Objective:** Evaluate the complete pipeline from raw UAV images to management recommendations.

**Table 7. End-to-end pipeline performance under different reconstruction × LLM configurations.**

| Reconstruction | Segmentation | LLM | BRR↑ | Morph. CV↓ | Expert Agree↑ | Total Time |
|---------------|-------------|-----|------|-----------|--------------|------------|
| COLMAP | SAM 2 Auto | GPT-4.1 | [TBD] | [TBD] | [TBD] | [TBD] |
| COLMAP | SAM 2 + DINO | GPT-4.1 | [TBD] | [TBD] | [TBD] | [TBD] |
| Semantic SfM | SAM 2 Auto | GPT-4.1 | [TBD] | [TBD] | [TBD] | [TBD] |
| **Hybrid** | **SAM 2 + DINO** | **Gemini 2.5 Pro** | [TBD] | [TBD] | [TBD] | [TBD] |
| Hybrid | SAM 2 + DINO | GLM-4-9B | [TBD] | [TBD] | [TBD] | [TBD] |

## 4.8 Ablation Studies

**Table 8. Ablation on reconstruction components.** Evaluated on post-defoliation subset (200 images).

| ID | Ablation Variable | Configuration | RC↑ | BRR↑ | NN↓ |
|----|-------------------|---------------|-----|------|-----|
| A1 | DINOv2 backbone | ViT-S/14 | [TBD] | [TBD] | [TBD] |
| | | ViT-B/14 | [TBD] | [TBD] | [TBD] |
| | | **ViT-L/14** | [TBD] | [TBD] | [TBD] |
| A2 | Matching threshold τ | 0.5 | [TBD] | [TBD] | [TBD] |
| | | 0.6 | [TBD] | [TBD] | [TBD] |
| | | **0.7** | [TBD] | [TBD] | [TBD] |
| | | 0.8 | [TBD] | [TBD] | [TBD] |
| | | 0.9 | [TBD] | [TBD] | [TBD] |
| A3 | SAM 2 grid density | 16×16 | [TBD] | [TBD] | [TBD] |
| | | **32×32** | [TBD] | [TBD] | [TBD] |
| | | 64×64 | [TBD] | [TBD] | [TBD] |
| A4 | Semantic loss weight λ_sem | 0.0 (geometric only) | [TBD] | [TBD] | [TBD] |
| | | 0.1 | [TBD] | [TBD] | [TBD] |
| | | **0.3** | [TBD] | [TBD] | [TBD] |
| | | 0.5 | [TBD] | [TBD] | [TBD] |
| | | 1.0 (semantic only) | [TBD] | [TBD] | [TBD] |
| A5 | # Input views | 50 | [TBD] | [TBD] | [TBD] |
| | | 100 | [TBD] | [TBD] | [TBD] |
| | | 200 | [TBD] | [TBD] | [TBD] |
| | | **340** | [TBD] | [TBD] | [TBD] |

**Table 9. Ablation on LLM modality contribution.** Using Gemini 2.5 Pro on 20 test cases.

| Input Configuration | Text | Frames | Morph. Graph | Expert Agree↑ | Gain vs. Text-only |
|--------------------|------|--------|-------------|--------------|-------------------|
| Text-only | ✓ | ✗ | ✗ | [TBD] | baseline |
| + Frames | ✓ | ✓ | ✗ | [TBD] | [TBD] |
| + Morphology graph | ✓ | ✗ | ✓ | [TBD] | [TBD] |
| **Full multimodal** | ✓ | ✓ | ✓ | [TBD] | [TBD] |
