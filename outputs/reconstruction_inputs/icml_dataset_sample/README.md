# ICML Dataset Reconstruction Input Smoke Run

Dataset root: `/Volumes/T9/ICML`

Command:

```bash
/Users/harshu/miniconda3/bin/python algorithms/prepare_reconstruction_inputs.py \
  --dataset-root /Volumes/T9/ICML \
  --out-dir outputs/reconstruction_inputs/icml_dataset_sample \
  --per-group 4 \
  --copy-images
```

This run selected 24 sharp frames: four images from each detected pre/post folder. The copied image files are ignored by Git to avoid committing large dataset derivatives, while the CSV manifests are committed.

Outputs:

- `reconstruction_images.csv`: selected source images, phase, folder, and local viewer-image path.
- `camera_path_scaffold.csv`: placeholder path coordinates for the viewer contract.
- `empty_boll_anchors.csv`: empty anchor table with the expected viewer schema.

Generated PLY, MP4, preview images, copied input images, and scene manifests are not kept in Git because they can be recreated from the scripts. `colmap` was not available on this machine at run time, so the camera path is a scaffold. The next reconstruction step is to replace the scaffold with real poses from COLMAP, VGGT, MASt3R, or another pose/depth estimator.
