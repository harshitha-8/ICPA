# 6. Conclusion

We have presented a semantic 3D reconstruction pipeline for cotton boll morphology analysis that addresses fundamental limitations of classical Structure-from-Motion on textureless, occluded, and wind-affected agricultural canopies. By replacing pixel-level feature matching with dense semantic correspondence fields derived from the DINOv2 vision foundation model, our approach achieves robust multi-view matching on surfaces where SIFT, SuperPoint, and other classical descriptors fail.

Our key findings are:

1. **DINOv2 semantic correspondences outperform classical feature matching on cotton.** On boll-containing image regions, DINOv2-based matching achieves [TBD]% higher inlier ratio than SuperPoint+SuperGlue, with correspondence stability persisting even under partial leaf occlusion.

2. **Defoliation serves as a controlled visibility intervention.** Post-defoliation imagery yields [TBD]% higher boll retention in 3D reconstruction and [TBD]% lower morphological measurement variance, establishing defoliation timing as a critical factor for phenotyping data quality.

3. **SAM 2 enables zero-shot boll segmentation with temporal consistency.** Instance retention rates of [TBD]% across multi-view sequences demonstrate that foundation model segmentation generalizes to cotton boll imagery without domain-specific training.

4. **Edge-deployable LLM reasoning closes the phenotyping-to-decision gap.** A 9B-parameter model achieves [TBD]% agreement with expert agronomist recommendations at sub-5-second inference latency, making field-level deployment feasible.

5. **The complete pipeline operates from raw UAV imagery to management recommendations,** establishing a practical template for agronomist-in-the-loop precision agriculture systems.

This work demonstrates that vision foundation models can unlock 3D phenotyping capabilities on crop targets that have historically resisted photogrammetric analysis, and that semantic priors offer a principled approach to reconstruction under the biological and environmental variability inherent to field agriculture. Future work will extend this paradigm to multi-temporal monitoring, multispectral integration, and closed-loop autonomous management systems.
