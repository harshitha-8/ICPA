# ICPA Paper Experiment Package

This package reports reproducible values from the current UAV cotton pipeline. It does not claim physical diameter, volume, or calibrated 3D accuracy unless manual measurements, camera calibration, GSD, or independent 3D reference data are added.

## Core Equations

Let candidate i have mask major axis a_i in pixels, minor axis b_i in pixels, area A_i, box area B_i, lint fraction l_i, green fraction g_i, brightness fraction q_i, and ground sampling distance s cm/px.

- Visibility proxy: V_i = A_i / B_i.
- Length proxy: L_i = s a_i.
- Width proxy: W_i = s b_i.
- Diameter proxy: D_i = (L_i + W_i) / 2.
- Ellipsoid volume proxy: U_i = (4 pi / 3)(L_i/2)(W_i/2)(W_i/2).
- Aspect score: R_i = max(0, 1 - (rho_i - 1) / 3.5), where rho_i = max(w_i,h_i)/min(w_i,h_i).
- Size score: S_i = min(sqrt(A_i)/20, 1).
- Readiness score: M_i = 0.32 l_i + 0.20 V_i + 0.16 q_i + 0.14 R_i + 0.12 S_i + 0.06(1-g_i) + 0.08 I[post].
- Score penalties: M_i <- 0.5 M_i if rho_i > 4.2; M_i <- 0.45 M_i if g_i > 0.55 and l_i < 0.20.
- Phase contrast: Delta_mu(x) = mean_post(x) - mean_pre(x).
- Relative phase contrast: Delta_%(x) = 100 (mean_post(x) - mean_pre(x)) / max(|mean_pre(x)|, eps).
- Bootstrap-style normal 95% CI used here: mean(x) +/- 1.96 std(x)/sqrt(n).
- Robust volume mutation uses U_i^99 = min(U_i, percentile_99(U)) before sorting, so the plot is not dominated by a few extreme proxy masks.
- First volume mutation threshold: D_thr = 5 mean(diff(sort(U^99))).

## Generated Tables

- `table_1_phase_count_summary.csv`: 2 rows, 5 columns.
- `table_2_folder_count_summary.csv`: 6 rows, 7 columns.
- `table_3_candidate_phase_summary.csv`: 2 rows, 11 columns.
- `table_4_phase_contrast.csv`: 6 rows, 5 columns.
- `table_5_candidate_score_ablation.csv`: 8 rows, 4 columns.
- `table_6_plot_grid_proxy_summary.csv`: 344 rows, 6 columns.
- `table_7_proxy_volume_mutation.csv`: 2 rows, 9 columns.
- `table_8_local_2p5d_summary.csv`: 12 rows, 11 columns.
- `table_9_phase_confidence_intervals.csv`: 8 rows, 4 columns.
- `table_10_image_candidate_summary.csv`: 160 rows, 7 columns.

## Generated Figures

- `readiness_distribution.png`: pre/post readiness distribution.
- `proxy_trait_boxplots.png`: diameter, volume, and visibility proxy distributions.
- `readiness_ablation.png`: ablation of readiness-score terms.
- `plot_grid_candidate_heatmaps.png`: 4 x 43 spatial candidate-density proxy maps.
- `volume_mutation_proxy.png`: sorted volume proxy and first-difference threshold analysis.
- `local_2p5d_quality_scatter.png`: local 2.5D target quality.

## Boundary For Paper Writing

Use these results as current MVP/proxy evidence. Detection counts and candidate-mining statistics are pipeline outputs, not manually verified ground truth. Diameter and volume are scale-dependent proxies under the current GSD assumption. For final agronomic claims, add expert annotations, physical boll measurements, or calibrated multi-view geometry.

## Manifest

```json
{
  "artifact_type": "icpa paper experiment package",
  "scientific_boundary": "Proxy evaluation package; no manual ground-truth accuracy claims.",
  "gsd_cm_per_px": 0.25,
  "grid_rows": 4,
  "grid_cols": 43,
  "inputs": {
    "phase_summary": "outputs/counts/icml_dataset_full/phase_summary.csv",
    "folder_summary": "outputs/counts/icml_dataset_full/folder_summary.csv",
    "image_candidate_summary": "outputs/metrics/measurement_ready_bolls/image_candidate_summary.csv",
    "candidates": "outputs/metrics/measurement_ready_bolls/measurement_ready_candidates.csv",
    "local_2p5d_summary": "outputs/metrics/local_boll_2p5d_reconstruction/local_boll_2p5d_summary.csv"
  },
  "outputs": {
    "tables": "outputs/experiments/icpa_paper_metrics/tables",
    "figures": "outputs/experiments/icpa_paper_metrics/figures",
    "report": "outputs/experiments/icpa_paper_metrics/experiment_report.md"
  }
}
```
