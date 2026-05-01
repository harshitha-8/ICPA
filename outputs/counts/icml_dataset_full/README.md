# ICML Dataset Full Counter Run

Dataset root: `/Volumes/T9/ICML`

Runtime: `/Users/harshu/miniconda3/bin/python`

Command:

```bash
/Users/harshu/miniconda3/bin/python algorithms/run_dataset_counter.py \
  --dataset-root /Volumes/T9/ICML \
  --out-dir outputs/counts/icml_dataset_full \
  --save-annotated-limit 0
```

Outputs:

- `phase_summary.csv`: pre/post totals.
- `folder_summary.csv`: folder-level phase summaries.

The detailed `counts_by_image.csv` file is not kept in Git because it is a regenerable experiment artifact. Use the command above to recreate it locally when needed.

Important note: the current inherited detector applies a `1.6x` multiplier to pre-defoliation counts. Therefore, `total_count` for pre-defoliation is adjusted count, while `total_raw_candidates` is the raw detector candidate count. This is useful for continuity with prior counting work, but the 3D paper should report both raw and adjusted values until the detector is recalibrated.

Phase-level result:

| Phase | Images | Total count | Raw candidates | Mean count |
|---|---:|---:|---:|---:|
| pre | 680 | 2914620 | 1821639 | 4286.206 |
| post | 869 | 2698919 | 2698919 | 3105.776 |

Folder-level result:

| Phase | Folder | Images | Mean count | Mean raw candidates |
|---|---|---:|---:|---:|
| post | `205_Post_Def_rgb` | 203 | 3203.148 | 3203.148 |
| post | `Post_def_rgb_part1` | 300 | 3159.483 | 3159.483 |
| post | `part3_post_def_rgb` | 340 | 3017.965 | 3017.965 |
| post | `part4_post_def_rgb` | 26 | 2874.115 | 2874.115 |
| pre | `Part_one_pre_def_rgb` | 340 | 4123.015 | 2576.900 |
| pre | `part 2_pre_def_rgb` | 340 | 4449.397 | 2780.862 |
