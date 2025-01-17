# 5. Discussion

## 5.1 Semantic Features vs. Photometric Features on Cotton

[TBD — Analysis of DINOv2 vs SuperPoint+SuperGlue results. Expected: semantic features significantly outperform on textureless boll surfaces but may underperform on textured soil/leaf regions. Discuss the domain-specific advantage of foundation model features for agricultural targets that violate classical photogrammetry assumptions.]

## 5.2 Defoliation as a Controlled Visibility Experiment

[TBD — Quantification of occlusion's impact on reconstruction quality. Expected: post-defoliation yields substantially higher BRR and lower morphological variance. Discuss implications: defoliation timing should consider not just agronomic factors but also phenotyping window optimization. This finding has practical significance — UAV flights timed immediately post-defoliation may provide the highest-quality morphological data before harvest.]

## 5.3 Reconstruction Completeness and the SfM Failure Mode

[TBD — Document where and why COLMAP fails. Expected failure modes: (1) insufficient correspondences on cotton lint → degenerate pose estimation, (2) photometric ambiguity across boll surfaces → noisy depth estimates, (3) wind-displaced bolls → temporal inconsistency in multi-view matching. Contrast with semantic reconstruction's robustness to these modes.]

## 5.4 Practical Implications for Precision Cotton Management

The end-to-end pipeline from UAV imagery to management recommendations has several practical implications:

- **PGR timing:** Boll diameter distributions and growth stagnation metrics can inform mepiquat chloride application timing.
- **Harvest aid scheduling:** Maturity index and open boll percentage guide defoliant and boll opener application.
- **Stress detection:** Anomalous size variance across canopy heights may indicate water stress or nutrient deficiency zones.
- **Yield estimation:** Boll count and volume measurements provide early yield projections.

## 5.5 Edge Deployment Considerations

[TBD — Discuss LLM benchmarking results. Expected: GLM-4-9B offers best reasoning quality at acceptable latency; smaller models (Phi-4) offer faster inference with some accuracy tradeoff. Discuss quantization impact and practical deployment on NVIDIA Jetson or Apple M-series hardware for in-field processing.]

## 5.6 Limitations

1. **Single-field evaluation.** Results are demonstrated on one field captured on one date. Generalization across cultivars, growth stages, and environmental conditions requires further validation.
2. **No ground-truth 3D geometry.** Without LiDAR ground truth, reconstruction quality is evaluated via proxy metrics (completeness, noise) rather than direct geometric error.
3. **Manual annotations limited.** Segmentation ground truth covers only 50 images per condition; larger annotation sets would strengthen evaluation.
4. **LLM reasoning evaluation.** Expert agronomist validation is limited in scope; comprehensive evaluation would require multi-expert inter-rater reliability study.
5. **Wind effects not explicitly modeled.** Wind-induced boll displacement is characterized indirectly through correspondence stability rather than explicit motion modeling.

## 5.7 Future Work

- Extension to multi-temporal monitoring across the growing season
- Integration of multispectral (NDVI, thermal) data for physiological measurements
- 3D Gaussian Splatting with semantic feature distillation for real-time visualization
- Federated learning across multiple farm sites for domain adaptation
- Integration with autonomous sprayer systems for closed-loop PGR application
