#!/usr/bin/env python3
"""Create HD pre/post defoliation UAV panels and clean 2.5D map views.

The exported figures are data-grounded visual assets for the ICPA paper/app.
They use real UAV RGB frames and an image-derived canopy/lint visibility height
proxy. They must not be described as calibrated metric 3D reconstruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm

from build_uav_orthomap_3d_figure import (
    DEFAULT_POST,
    DEFAULT_PRE,
    OrthomapResult,
    build_result,
    plot_surface,
)


DEFAULT_OUT = Path("/Volumes/T9/ICPA/outputs/figures/hd_prepost_3d_maps")


def configure_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        }
    )


def save_panel(pre: OrthomapResult, post: OrthomapResult, out_dir: Path) -> Path:
    """Save a manuscript-style panel with RGB, proxy height, and 3D map."""
    fig = plt.figure(figsize=(18.0, 11.0), dpi=220)
    fig.patch.set_facecolor("white")
    rows = [(pre, "Pre-defoliation"), (post, "Post-defoliation")]

    for row_idx, (result, title) in enumerate(rows):
        ax_rgb = fig.add_subplot(2, 3, row_idx * 3 + 1)
        ax_rgb.imshow(result.rgb_display)
        ax_rgb.set_title(f"{title} UAV RGB", fontsize=13, fontweight="bold", pad=8)
        ax_rgb.axis("off")

        ax_height = fig.add_subplot(2, 3, row_idx * 3 + 2)
        ax_height.imshow(cm.viridis(result.height)[..., :3])
        ax_height.set_title("Image-derived visibility height", fontsize=13, fontweight="bold", pad=8)
        ax_height.axis("off")

        ax_map = fig.add_subplot(2, 3, row_idx * 3 + 3, projection="3d")
        plot_surface(ax_map, result)
        ax_map.set_title(f"{title} 2.5D map", fontsize=13, fontweight="bold", pad=5)

    fig.suptitle(
        "Pre/Post Defoliation Cotton UAV Orthomap-to-2.5D Map",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0.020, 1, 0.930), h_pad=1.35, w_pad=0.95)
    out_path = out_dir / "pre_post_defoliation_hd_orthomap_panel.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out_path


def save_clean_map(result: OrthomapResult, out_dir: Path) -> Path:
    """Save a clean HD 3D map view for one phase."""
    fig = plt.figure(figsize=(12.0, 8.5), dpi=260)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    plot_surface(ax, result)
    ax.view_init(elev=50, azim=-55)
    ax.set_title(
        f"{result.phase.title()} Cotton UAV 2.5D Map\n"
        "Real RGB texture with image-derived canopy/lint visibility height",
        fontsize=14,
        fontweight="bold",
        pad=8,
    )
    out_path = out_dir / f"{result.phase}_clean_2p5d_map_hd.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    return out_path


def save_rgb_height(result: OrthomapResult, out_dir: Path) -> tuple[Path, Path]:
    rgb_path = out_dir / f"{result.phase}_rgb_hd.png"
    height_path = out_dir / f"{result.phase}_visibility_height_hd.png"

    fig_rgb = plt.figure(figsize=(10, 7), dpi=240)
    ax_rgb = fig_rgb.add_subplot(1, 1, 1)
    ax_rgb.imshow(result.rgb_display)
    ax_rgb.set_title(f"{result.phase.title()} UAV RGB", fontsize=13, fontweight="bold")
    ax_rgb.axis("off")
    fig_rgb.savefig(rgb_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig_rgb)

    fig_h = plt.figure(figsize=(10, 7), dpi=240)
    ax_h = fig_h.add_subplot(1, 1, 1)
    ax_h.imshow(cm.viridis(result.height)[..., :3])
    ax_h.set_title(f"{result.phase.title()} image-derived visibility height", fontsize=13, fontweight="bold")
    ax_h.axis("off")
    fig_h.savefig(height_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig_h)
    return rgb_path, height_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-image", type=Path, default=DEFAULT_PRE)
    parser.add_argument("--post-image", type=Path, default=DEFAULT_POST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--display-edge", type=int, default=1800)
    parser.add_argument("--surface-edge", type=int, default=360)
    return parser.parse_args()


def main() -> None:
    configure_fonts()
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pre = build_result("pre-defoliation", args.pre_image, args.out_dir, args.display_edge, args.surface_edge)
    post = build_result("post-defoliation", args.post_image, args.out_dir, args.display_edge, args.surface_edge)

    panel = save_panel(pre, post, args.out_dir)
    pre_map = save_clean_map(pre, args.out_dir)
    post_map = save_clean_map(post, args.out_dir)
    pre_rgb, pre_height = save_rgb_height(pre, args.out_dir)
    post_rgb, post_height = save_rgb_height(post, args.out_dir)

    manifest = {
        "scientific_boundary": "Real UAV RGB texture plus image-derived 2.5D visibility surface; not calibrated metric 3D.",
        "main_panel": str(panel),
        "clean_3d_maps": {"pre": str(pre_map), "post": str(post_map)},
        "rgb": {"pre": str(pre_rgb), "post": str(post_rgb)},
        "height_proxy": {"pre": str(pre_height), "post": str(post_height)},
        "ply": {"pre": str(pre.ply_path), "post": str(post.ply_path)},
        "sources": {"pre": str(pre.source), "post": str(post.source)},
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"main_panel={panel}")
    print(f"pre_3d_map={pre_map}")
    print(f"post_3d_map={post_map}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
