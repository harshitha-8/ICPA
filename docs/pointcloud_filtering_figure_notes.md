# Point-Cloud Filtering Figure Notes

## Inspiration

The peanut reconstruction paper shows a clear visual progression:

1. original 3D color point cloud;
2. PassThrough filtering to retain the plant/plot region;
3. statistical filtering to remove scattered noise;
4. multi-view fusion after rotation/translation.

That figure style is useful for our ICPA paper because cotton reviewers need to
see not only the final map, but also how the noisy UAV-derived point set becomes
a measurement-ready boll map.

## Cotton Version

The local figure generator creates:

- original UAV color point-cloud proxy;
- PassThrough study-site filtering;
- statistical + cotton-lint filtering;
- source UAV frame;
- plot-grid mapping of measurement-ready boll candidates;
- two-frame fused point-cloud proxy.

Regenerate:

```bash
/Users/harshu/miniconda3/bin/python tools/build_cotton_pointcloud_filtering_figure.py \
  --manifest outputs/reconstruction_inputs/icml_dataset_sample/reconstruction_images.csv \
  --phase post \
  --out outputs/figures/cotton_pointcloud_filtering_process.png
```

## Accuracy Boundary

This figure is a **UAV 2.5D scaffold** generated from real cotton images. It is
useful for method planning, paper drafts, and visual inspection. For final
metric claims, the same panels should be regenerated from calibrated COLMAP,
VGGT/MASt3R, or 3D Gaussian Splatting geometry.

## Source

- Peanut point-cloud reconstruction article: https://pmc.ncbi.nlm.nih.gov/articles/PMC12924431/
