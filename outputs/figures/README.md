# Figure Outputs

Generated figures are kept out of git so the repository stays lightweight.

Regenerate the Utonia-style cotton representation montage with:

```bash
/Users/harshu/miniconda3/bin/python tools/build_utonia_style_figure.py \
  --manifest outputs/reconstruction_inputs/icml_dataset_sample/reconstruction_images.csv \
  --out outputs/figures/utonia_style_cotton_representation.png \
  --image-width 1000
```

The current figure is an illustrative scaffold built from real pre/post UAV
frames. It uses pseudo-depth and semantic proxy panels until calibrated camera
poses, depth, or Gaussian splatting outputs are available.

Regenerate the point-cloud filtering and mapping scaffold with:

```bash
/Users/harshu/miniconda3/bin/python tools/build_cotton_pointcloud_filtering_figure.py \
  --manifest outputs/reconstruction_inputs/icml_dataset_sample/reconstruction_images.csv \
  --phase post \
  --out outputs/figures/cotton_pointcloud_filtering_process.png
```
