# ICML Dataset Counter Smoke Run

Dataset root: `/Volumes/T9/ICML`

Runtime: `/Users/harshu/miniconda3/bin/python`

Command:

```bash
/Users/harshu/miniconda3/bin/python algorithms/run_dataset_counter.py \
  --dataset-root /Volumes/T9/ICML \
  --out-dir outputs/counts/icml_dataset_sample \
  --max-images 40 \
  --save-annotated-limit 8
```

Observed real image folders, excluding macOS `._*` sidecar files:

| Folder | Images | Phase |
|---|---:|---|
| `205_Post_Def_rgb` | 203 | post |
| `Post_def_rgb_part1` | 300 | post |
| `part3_post_def_rgb` | 340 | post |
| `part4_post_def_rgb` | 26 | post |
| `Part_one_pre_def_rgb` | 340 | pre |
| `part 2_pre_def_rgb` | 340 | pre |

The smoke run intentionally used the first 40 images only. All 40 came from the first sorted post-defoliation folder, producing a mean count of 3307.475 bolls/image. This is a detector sanity check, not the final experiment table.

Next full run should use the same script without `--max-images`, ideally without annotated image export:

```bash
/Users/harshu/miniconda3/bin/python algorithms/run_dataset_counter.py \
  --dataset-root /Volumes/T9/ICML \
  --out-dir outputs/counts/icml_dataset_full \
  --save-annotated-limit 0
```
