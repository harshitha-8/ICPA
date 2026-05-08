#!/usr/bin/env python3
"""Build paper-ready experiment tables and figures for the ICPA cotton study.

The package is intentionally honest about validation boundaries. It computes
fully reproducible statistics from the current repository artifacts, but labels
measurements as proxy values whenever no physical ground truth is available.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUT = Path("outputs/experiments/icpa_paper_metrics")
GSD_CM_PER_PX = 0.25
GRID_ROWS = 4
GRID_COLS = 43


def ci95(values: pd.Series) -> tuple[float, float]:
    arr = values.dropna().astype(float).to_numpy()
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return (mean, mean)
    sem = float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
    return (mean - 1.96 * sem, mean + 1.96 * sem)


def add_proxy_traits(candidates: pd.DataFrame, gsd_cm_per_px: float) -> pd.DataFrame:
    df = candidates.copy()
    df["length_cm_proxy"] = df["mask_major_axis_px"] * gsd_cm_per_px
    df["width_cm_proxy"] = df["mask_minor_axis_px"] * gsd_cm_per_px
    df["diameter_cm_proxy"] = (df["length_cm_proxy"] + df["width_cm_proxy"]) / 2.0
    df["ellipsoid_volume_cm3_proxy"] = (
        4.0
        / 3.0
        * math.pi
        * (df["length_cm_proxy"] / 2.0)
        * (df["width_cm_proxy"] / 2.0)
        * (df["width_cm_proxy"] / 2.0)
    )
    df["aspect_ratio"] = df[["width", "height"]].max(axis=1) / df[["width", "height"]].min(axis=1).clip(lower=1)
    df["plot_row"] = np.floor(df["center_y"] / df.groupby("image")["center_y"].transform("max").clip(lower=1) * GRID_ROWS).clip(0, GRID_ROWS - 1).astype(int) + 1
    df["plot_col"] = np.floor(df["center_x"] / df.groupby("image")["center_x"].transform("max").clip(lower=1) * GRID_COLS).clip(0, GRID_COLS - 1).astype(int) + 1
    return df


def recompute_readiness(df: pd.DataFrame, drop: str | None = None) -> pd.Series:
    aspect_score = (1.0 - ((df["aspect_ratio"] - 1.0).clip(lower=0) / 3.5)).clip(lower=0)
    size_score = np.sqrt(df["area_px"]).clip(upper=20.0) / 20.0
    phase_bonus = np.where(df["phase"] == "post", 0.08, 0.0)
    terms = {
        "lint": 0.32 * df["lint_fraction"],
        "visibility": 0.20 * df["visibility"],
        "brightness": 0.16 * df["brightness"],
        "shape": 0.14 * aspect_score,
        "size": 0.12 * size_score,
        "green_penalty": 0.06 * (1.0 - df["green_fraction"]),
        "phase_bonus": phase_bonus,
    }
    if drop is not None:
        terms[drop] = 0.0
    score = sum(terms.values())
    score = np.where(df["aspect_ratio"] > 4.2, score * 0.50, score)
    score = np.where((df["green_fraction"] > 0.55) & (df["lint_fraction"] < 0.20), score * 0.45, score)
    return pd.Series(np.clip(score, 0.0, 1.0), index=df.index)


def top_overlap(reference: pd.Series, alternative: pd.Series, frac: float = 0.05) -> float:
    k = max(1, int(round(len(reference) * frac)))
    ref_top = set(reference.nlargest(k).index)
    alt_top = set(alternative.nlargest(k).index)
    return len(ref_top & alt_top) / k


def write_tables(out_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(table_dir / f"{name}.csv", index=False)


def plot_readiness_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=180)
    for phase, color in [("pre", "#4C78A8"), ("post", "#F58518")]:
        vals = df.loc[df["phase"] == phase, "measurement_ready_score"].to_numpy()
        ax.hist(vals, bins=28, alpha=0.58, density=True, label=phase, color=color)
    ax.set_xlabel("Measurement-readiness score")
    ax.set_ylabel("Density")
    ax.set_title("Pre/post measurement-ready candidate score distribution")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "readiness_distribution.png")
    plt.close(fig)


def plot_trait_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6), dpi=180)
    metrics = [
        ("diameter_cm_proxy", "Diameter proxy (cm)"),
        ("ellipsoid_volume_cm3_proxy", "Ellipsoid volume proxy (cm3)"),
        ("visibility", "Visibility proxy"),
    ]
    for ax, (col, label) in zip(axes, metrics):
        data = [df.loc[df["phase"] == phase, col].dropna().to_numpy() for phase in ["pre", "post"]]
        ax.boxplot(data, tick_labels=["pre", "post"], showfliers=False, patch_artist=True)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Proxy trait distributions by defoliation phase")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "proxy_trait_boxplots.png")
    plt.close(fig)


def plot_ablation(ablation: pd.DataFrame, out_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9.0, 4.5), dpi=180)
    x = np.arange(len(ablation))
    ax1.bar(x - 0.18, ablation["mean_score"], width=0.36, color="#72B7B2", label="mean score")
    ax1.set_ylabel("Mean readiness score")
    ax1.set_ylim(0, max(0.9, float(ablation["mean_score"].max()) + 0.05))
    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, ablation["top5_overlap_with_full"], marker="o", color="#E45756", label="top-5% overlap")
    ax2.set_ylabel("Top-5% overlap with full model")
    ax2.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ablation["variant"], rotation=35, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")
    ax1.set_title("Ablation of candidate-readiness scoring terms")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "readiness_ablation.png")
    plt.close(fig)


def plot_spatial_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 4.8), dpi=180)
    for ax, phase in zip(axes, ["pre", "post"]):
        sub = df[df["phase"] == phase]
        grid = (
            sub.groupby(["plot_row", "plot_col"]).size().unstack(fill_value=0).reindex(index=range(1, GRID_ROWS + 1), columns=range(1, GRID_COLS + 1), fill_value=0)
        )
        im = ax.imshow(grid.to_numpy(), aspect="auto", cmap="inferno")
        ax.set_title(f"{phase} candidate density proxy, {GRID_ROWS} x {GRID_COLS} grid")
        ax.set_ylabel("Row")
        ax.set_xlabel("Column")
        fig.colorbar(im, ax=ax, fraction=0.016, pad=0.01, label="candidate count")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "plot_grid_candidate_heatmaps.png")
    plt.close(fig)


def plot_volume_mutation(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), dpi=180)
    for col, phase in enumerate(["pre", "post"]):
        raw_vals = np.sort(df.loc[df["phase"] == phase, "ellipsoid_volume_cm3_proxy"].dropna().to_numpy())
        cap = float(np.percentile(raw_vals, 99)) if len(raw_vals) else float("nan")
        vals = np.clip(raw_vals, None, cap)
        diffs = np.diff(vals, prepend=vals[0])
        threshold = float(np.mean(diffs) * 5.0) if len(diffs) else float("nan")
        mutation_idx = int(np.argmax(diffs > threshold)) if np.any(diffs > threshold) else -1
        axes[0, col].plot(np.arange(len(vals)), vals, lw=1.5)
        if mutation_idx >= 0:
            axes[0, col].axvline(mutation_idx, color="#4C78A8", ls="--", lw=1.1)
        axes[0, col].set_title(f"{phase}: p99-capped sorted volume proxy")
        axes[0, col].set_ylabel("volume proxy (cm3)")
        axes[1, col].plot(np.arange(len(diffs)), diffs, lw=1.2)
        axes[1, col].axhline(threshold, color="#E45756", ls="--", lw=1.1)
        axes[1, col].set_title(f"{phase}: first-difference threshold")
        axes[1, col].set_ylabel("difference")
        axes[1, col].set_xlabel("candidate index")
        rows.append(
            {
                "phase": phase,
                "candidates": len(raw_vals),
                "raw_mean_volume_proxy": float(np.mean(raw_vals)),
                "median_volume_proxy": float(np.median(raw_vals)),
                "p95_volume_proxy": float(np.percentile(raw_vals, 95)),
                "p99_volume_proxy": cap,
                "p99_capped_mean_volume_proxy": float(np.mean(vals)),
                "threshold_D_p99_capped": threshold,
                "first_mutation_index_p99_capped": mutation_idx,
            }
        )
    fig.suptitle("Robust proxy ellipsoid-volume mutation analysis")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "volume_mutation_proxy.png")
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_local_2p5d(local: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=180)
    sc = ax.scatter(local["lint_mean"], local["height_mean"], s=70, c=local["measurement_ready_score"], cmap="viridis")
    for _, row in local.iterrows():
        ax.text(row["lint_mean"], row["height_mean"], str(int(row["local_rank"])), fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("Mean lint proxy")
    ax.set_ylabel("Mean 2.5D height proxy")
    ax.set_title("Local 2.5D reconstruction target quality")
    fig.colorbar(sc, ax=ax, label="readiness score")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "local_2p5d_quality_scatter.png")
    plt.close(fig)


def build_report(out_dir: Path, tables: dict[str, pd.DataFrame], manifest: dict[str, object]) -> None:
    lines = [
        "# ICPA Paper Experiment Package",
        "",
        "This package reports reproducible values from the current UAV cotton pipeline. It does not claim physical diameter, volume, or calibrated 3D accuracy unless manual measurements, camera calibration, GSD, or independent 3D reference data are added.",
        "",
        "## Core Equations",
        "",
        "Let candidate i have mask major axis a_i in pixels, minor axis b_i in pixels, area A_i, box area B_i, lint fraction l_i, green fraction g_i, brightness fraction q_i, and ground sampling distance s cm/px.",
        "",
        "- Visibility proxy: V_i = A_i / B_i.",
        "- Length proxy: L_i = s a_i.",
        "- Width proxy: W_i = s b_i.",
        "- Diameter proxy: D_i = (L_i + W_i) / 2.",
        "- Ellipsoid volume proxy: U_i = (4 pi / 3)(L_i/2)(W_i/2)(W_i/2).",
        "- Aspect score: R_i = max(0, 1 - (rho_i - 1) / 3.5), where rho_i = max(w_i,h_i)/min(w_i,h_i).",
        "- Size score: S_i = min(sqrt(A_i)/20, 1).",
        "- Readiness score: M_i = 0.32 l_i + 0.20 V_i + 0.16 q_i + 0.14 R_i + 0.12 S_i + 0.06(1-g_i) + 0.08 I[post].",
        "- Score penalties: M_i <- 0.5 M_i if rho_i > 4.2; M_i <- 0.45 M_i if g_i > 0.55 and l_i < 0.20.",
        "- Phase contrast: Delta_mu(x) = mean_post(x) - mean_pre(x).",
        "- Relative phase contrast: Delta_%(x) = 100 (mean_post(x) - mean_pre(x)) / max(|mean_pre(x)|, eps).",
        "- Bootstrap-style normal 95% CI used here: mean(x) +/- 1.96 std(x)/sqrt(n).",
        "- Robust volume mutation uses U_i^99 = min(U_i, percentile_99(U)) before sorting, so the plot is not dominated by a few extreme proxy masks.",
        "- First volume mutation threshold: D_thr = 5 mean(diff(sort(U^99))).",
        "",
        "## Generated Tables",
        "",
    ]
    for name, df in tables.items():
        lines.append(f"- `{name}.csv`: {len(df)} rows, {len(df.columns)} columns.")
    lines += [
        "",
        "## Generated Figures",
        "",
        "- `readiness_distribution.png`: pre/post readiness distribution.",
        "- `proxy_trait_boxplots.png`: diameter, volume, and visibility proxy distributions.",
        "- `readiness_ablation.png`: ablation of readiness-score terms.",
        "- `plot_grid_candidate_heatmaps.png`: 4 x 43 spatial candidate-density proxy maps.",
        "- `volume_mutation_proxy.png`: sorted volume proxy and first-difference threshold analysis.",
        "- `local_2p5d_quality_scatter.png`: local 2.5D target quality.",
        "",
        "## Boundary For Paper Writing",
        "",
        "Use these results as current MVP/proxy evidence. Detection counts and candidate-mining statistics are pipeline outputs, not manually verified ground truth. Diameter and volume are scale-dependent proxies under the current GSD assumption. For final agronomic claims, add expert annotations, physical boll measurements, or calibrated multi-view geometry.",
        "",
        "## Manifest",
        "",
        "```json",
        json.dumps(manifest, indent=2),
        "```",
        "",
    ]
    (out_dir / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    phase_summary = pd.read_csv(args.phase_summary)
    folder_summary = pd.read_csv(args.folder_summary)
    image_summary = pd.read_csv(args.image_candidate_summary)
    candidates = add_proxy_traits(pd.read_csv(args.candidates), args.gsd_cm_per_px)
    local = pd.read_csv(args.local_2p5d_summary)

    phase_candidate = candidates.groupby("phase").agg(
        candidates=("rank", "count"),
        mean_readiness=("measurement_ready_score", "mean"),
        median_readiness=("measurement_ready_score", "median"),
        mean_lint=("lint_fraction", "mean"),
        mean_green=("green_fraction", "mean"),
        mean_visibility=("visibility", "mean"),
        mean_diameter_cm_proxy=("diameter_cm_proxy", "mean"),
        mean_volume_cm3_proxy=("ellipsoid_volume_cm3_proxy", "mean"),
        high_score_ge_055=("measurement_ready_score", lambda s: float(np.mean(s >= 0.55))),
        high_score_ge_075=("measurement_ready_score", lambda s: float(np.mean(s >= 0.75))),
    ).reset_index()
    cis = []
    for phase, sub in candidates.groupby("phase"):
        for metric in ["measurement_ready_score", "visibility", "diameter_cm_proxy", "ellipsoid_volume_cm3_proxy"]:
            lo, hi = ci95(sub[metric])
            cis.append({"phase": phase, "metric": metric, "ci95_low": lo, "ci95_high": hi})
    phase_ci = pd.DataFrame(cis)

    phase_contrast_rows = []
    for metric in ["measurement_ready_score", "lint_fraction", "green_fraction", "visibility", "diameter_cm_proxy", "ellipsoid_volume_cm3_proxy"]:
        means = candidates.groupby("phase")[metric].mean()
        pre = float(means.get("pre", np.nan))
        post = float(means.get("post", np.nan))
        phase_contrast_rows.append(
            {
                "metric": metric,
                "pre_mean": pre,
                "post_mean": post,
                "post_minus_pre": post - pre,
                "relative_change_pct": 100.0 * (post - pre) / max(abs(pre), 1e-12),
            }
        )
    phase_contrast = pd.DataFrame(phase_contrast_rows)

    full = recompute_readiness(candidates)
    ablation_rows = [{"variant": "full", "mean_score": float(full.mean()), "spearman_with_full": 1.0, "top5_overlap_with_full": 1.0}]
    for drop in ["lint", "visibility", "brightness", "shape", "size", "green_penalty", "phase_bonus"]:
        alt = recompute_readiness(candidates, drop=drop)
        ablation_rows.append(
            {
                "variant": f"minus_{drop}",
                "mean_score": float(alt.mean()),
                "spearman_with_full": float(full.corr(alt, method="spearman")),
                "top5_overlap_with_full": top_overlap(full, alt, frac=0.05),
            }
        )
    ablation = pd.DataFrame(ablation_rows)

    plot_grid = candidates.groupby(["phase", "plot_row", "plot_col"]).agg(
        candidate_count=("rank", "count"),
        mean_readiness=("measurement_ready_score", "mean"),
        mean_volume_cm3_proxy=("ellipsoid_volume_cm3_proxy", "mean"),
    ).reset_index()

    volume_mutation = plot_volume_mutation(candidates, out_dir)
    plot_readiness_distribution(candidates, out_dir)
    plot_trait_boxplots(candidates, out_dir)
    plot_ablation(ablation, out_dir)
    plot_spatial_heatmaps(candidates, out_dir)
    plot_local_2p5d(local, out_dir)

    tables = {
        "table_1_phase_count_summary": phase_summary,
        "table_2_folder_count_summary": folder_summary,
        "table_3_candidate_phase_summary": phase_candidate.round(5),
        "table_4_phase_contrast": phase_contrast.round(5),
        "table_5_candidate_score_ablation": ablation.round(5),
        "table_6_plot_grid_proxy_summary": plot_grid.round(5),
        "table_7_proxy_volume_mutation": volume_mutation.round(5),
        "table_8_local_2p5d_summary": local.round(5),
        "table_9_phase_confidence_intervals": phase_ci.round(5),
        "table_10_image_candidate_summary": image_summary,
    }
    write_tables(out_dir, tables)
    manifest = {
        "artifact_type": "icpa paper experiment package",
        "scientific_boundary": "Proxy evaluation package; no manual ground-truth accuracy claims.",
        "gsd_cm_per_px": args.gsd_cm_per_px,
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
        "inputs": {
            "phase_summary": str(args.phase_summary),
            "folder_summary": str(args.folder_summary),
            "image_candidate_summary": str(args.image_candidate_summary),
            "candidates": str(args.candidates),
            "local_2p5d_summary": str(args.local_2p5d_summary),
        },
        "outputs": {
            "tables": str(out_dir / "tables"),
            "figures": str(out_dir / "figures"),
            "report": str(out_dir / "experiment_report.md"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    build_report(out_dir, tables, manifest)
    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--phase-summary", type=Path, default=Path("outputs/counts/icml_dataset_full/phase_summary.csv"))
    parser.add_argument("--folder-summary", type=Path, default=Path("outputs/counts/icml_dataset_full/folder_summary.csv"))
    parser.add_argument("--image-candidate-summary", type=Path, default=Path("outputs/metrics/measurement_ready_bolls/image_candidate_summary.csv"))
    parser.add_argument("--candidates", type=Path, default=Path("outputs/metrics/measurement_ready_bolls/measurement_ready_candidates.csv"))
    parser.add_argument("--local-2p5d-summary", type=Path, default=Path("outputs/metrics/local_boll_2p5d_reconstruction/local_boll_2p5d_summary.csv"))
    parser.add_argument("--gsd-cm-per-px", type=float, default=GSD_CM_PER_PX)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
