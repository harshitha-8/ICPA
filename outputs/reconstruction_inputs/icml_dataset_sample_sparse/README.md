# Sparse-View Reconstruction Manifests

These manifests are generated from
`outputs/reconstruction_inputs/icml_dataset_sample/reconstruction_images.csv`.

They support MegaDepth-X-inspired sparse-view robustness experiments:

- `stride_1.csv`: dense selected sample;
- `stride_2.csv`: every second frame within each phase/folder group;
- `stride_4.csv`: every fourth frame within each phase/folder group;
- `stride_8.csv`: ultra-sparse stress test;
- `balanced_4_per_group.csv`: equal frame budget for each pre/post folder.

Regenerate with:

```bash
/Users/harshu/miniconda3/bin/python algorithms/sparse_view_sampler.py \
  --manifest outputs/reconstruction_inputs/icml_dataset_sample/reconstruction_images.csv \
  --out-dir outputs/reconstruction_inputs/icml_dataset_sample_sparse \
  --strides 1 2 4 8 \
  --per-group-budget 4
```
