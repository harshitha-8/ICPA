#!/usr/bin/env python3
"""Build peanut-paper-inspired cotton point-cloud filtering figures.

The figure is generated from the cotton UAV dataset and mirrors the structure of
plant point-cloud preprocessing figures: original color point cloud,
PassThrough/study-site crop, statistical/semantic filtering, and a simple
two-view fusion schematic. The geometry is a current UAV 2.5D proxy until true
camera poses or Gaussian splats are available.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from App.reconstruction_core import (
    compute_measurements,
    create_plot_grid_map,
    depth_to_points,
    estimate_depth,
    load_rgb,
    robust_subset,
)
from algorithms.cotton_boll_detector import detect_cotton_bolls


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def choose_images(manifest: Path, phase: str) -> tuple[Path, Path]:
    rows = [row for row in read_manifest(manifest) if row["phase"] == phase]
    if len(rows) < 2:
        raise RuntimeError(f"Need at least two {phase} images in {manifest}")
    return Path(rows[0]["source_image"]), Path(rows[1]["source_image"])


def point_proxy(rgb: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = estimate_depth(rgb)
    points, colors = depth_to_points(rgb, depth, max_points=max_points)
    return points, colors / 255.0, depth


def passthrough_mask(points: np.ndarray) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return (
        (x >= np.quantile(x, 0.06))
        & (x <= np.quantile(x, 0.94))
        & (y >= np.quantile(y, 0.08))
        & (y <= np.quantile(y, 0.92))
        & (z >= np.quantile(z, 0.08))
    )


def statistical_filter(points: np.ndarray, colors: np.ndarray) -> np.ndarray:
    brightness = colors.mean(axis=1)
    sat = colors.max(axis=1) - colors.min(axis=1)
    lint_like = (brightness > np.quantile(brightness, 0.58)) & (sat < np.quantile(sat, 0.72))
    return lint_like


def scatter_panel(ax, points: np.ndarray, colors: np.ndarray, title: str, s: float = 1.0) -> None:
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=s, linewidths=0, alpha=0.88)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("X proxy")
    ax.set_ylabel("Y proxy")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)


def build_figure(image_a: Path, image_b: Path, out_path: Path, max_points: int) -> None:
    rgb_a = load_rgb(image_a, long_edge=900)
    rgb_b = load_rgb(image_b, long_edge=900)
    points_a, colors_a, depth_a = point_proxy(rgb_a, max_points=max_points)
    points_b, colors_b, _ = point_proxy(rgb_b, max_points=max_points)

    pass_mask = passthrough_mask(points_a)
    stat_mask = pass_mask & statistical_filter(points_a, colors_a)

    _, _, candidates = detect_cotton_bolls(rgb_a, "post")
    measurements = compute_measurements(
        candidates=candidates,
        original_rgb=rgb_a,
        depth=depth_a,
        original_shape=rgb_a.shape[:2],
        resized_shape=rgb_a.shape[:2],
        gsd_cm_per_px=0.25,
    )
    ready = robust_subset(measurements)
    plot_map, _ = create_plot_grid_map(rgb_a, ready)

    fused_points = np.vstack([points_a[::2], points_b[::2] + np.array([0.035, 0.025, 0.0])])
    fused_colors = np.vstack([colors_a[::2], colors_b[::2]])
    fused_mask = passthrough_mask(fused_points)

    fig = plt.figure(figsize=(14, 8), dpi=180)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.28, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    scatter_panel(ax1, points_a, colors_a, "(a) Original UAV color point-cloud proxy", s=1.0)

    ax2 = fig.add_subplot(gs[0, 1])
    scatter_panel(ax2, points_a[pass_mask], colors_a[pass_mask], "(b) PassThrough study-site filtering", s=1.2)

    ax3 = fig.add_subplot(gs[0, 2])
    scatter_panel(ax3, points_a[stat_mask], colors_a[stat_mask], "(c) Statistical + cotton-lint filtering", s=1.5)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(rgb_a)
    ax4.set_title("(d) Source UAV frame", fontsize=10, loc="left")
    ax4.axis("off")

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(plot_map)
    ax5.set_title("(e) Plot-grid mapping of measurement-ready bolls", fontsize=10, loc="left")
    ax5.axis("off")

    ax6 = fig.add_subplot(gs[1, 2])
    scatter_panel(ax6, fused_points[fused_mask], fused_colors[fused_mask], "(f) Two-frame fused point-cloud proxy", s=1.1)

    fig.suptitle(
        "Cotton UAV point-cloud preprocessing and mapping scaffold",
        fontsize=14,
        y=0.985,
    )
    fig.text(
        0.5,
        0.015,
        "Generated from real cotton UAV frames. Point cloud is a current 2.5D proxy; replace with calibrated COLMAP/VGGT/3DGS geometry for final metric claims.",
        ha="center",
        fontsize=9,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("outputs/reconstruction_inputs/icml_dataset_sample/reconstruction_images.csv"))
    parser.add_argument("--phase", choices=["pre", "post"], default="post")
    parser.add_argument("--out", type=Path, default=Path("outputs/figures/cotton_pointcloud_filtering_process.png"))
    parser.add_argument("--max-points", type=int, default=16000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_a, image_b = choose_images(args.manifest, args.phase)
    build_figure(image_a, image_b, args.out, args.max_points)


if __name__ == "__main__":
    main()
