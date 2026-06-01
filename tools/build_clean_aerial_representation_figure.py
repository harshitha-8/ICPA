#!/usr/bin/env python3
"""Build clean aerial pre/post cotton representation figures.

This replaces the earlier busy Utonia-style collage with a paper-friendly
layout: aerial RGB evidence, morphology/visibility proxy, semantic proxy, and
clean 2.5D map views. All panels are generated from real UAV frames. The 2.5D
maps are image-derived proxy surfaces, not calibrated metric 3D.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from PIL import Image

from build_uav_orthomap_3d_figure import (
    DEFAULT_POST,
    DEFAULT_PRE,
    OrthomapResult,
    build_result,
    plot_surface,
)


OUT_DIR = Path("/Volumes/T9/ICPA/outputs/figures/clean_aerial_cotton_representation")


def configure_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "figure.titlesize": 17,
        }
    )


def semantic_proxy(rgb: np.ndarray, phase: str) -> np.ndarray:
    """Return a restrained semantic proxy for canopy, soil, and exposed lint."""
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    value = np.max(rgb_f, axis=2)
    saturation = np.max(rgb_f, axis=2) - np.min(rgb_f, axis=2)
    exg = 2.0 * g - r - b

    soil = ((r > g * 0.95) & (value < 0.68)) | (value < 0.30)
    canopy = exg > 0.04
    lint = (value > 0.58) & (saturation < 0.22)

    out = np.full_like(rgb, (231, 232, 224))
    out[soil] = (158, 134, 108)
    out[canopy] = (78, 143, 86) if phase == "pre" else (112, 132, 105)
    out[lint] = (238, 238, 232) if phase == "post" else (214, 219, 205)

    # Softly blend with original luminance so the proxy still feels aerial.
    lum = (0.299 * r + 0.587 * g + 0.114 * b)[..., None]
    blended = 0.72 * out.astype(np.float32) + 0.28 * (lum * 255.0)
    return np.clip(blended, 0, 255).astype(np.uint8)


def add_panel_label(ax: plt.Axes, label: str, caption: str) -> None:
    ax.set_title(label, loc="left", pad=7)
    ax.text(
        0.0,
        -0.055,
        caption,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4E554C",
    )


def save_surface_only(result: OrthomapResult, out_dir: Path) -> Path:
    fig = plt.figure(figsize=(10.5, 7.0), dpi=260)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    plot_surface(ax, result)
    ax.view_init(elev=49, azim=-56)
    ax.set_title(
        f"Aerial 2.5D Map - {result.phase.title()}",
        fontsize=15,
        fontweight="bold",
        pad=8,
    )
    fig.text(
        0.5,
        0.035,
        "Real UAV RGB texture with image-derived canopy/lint visibility height; not calibrated metric 3D.",
        ha="center",
        fontsize=8.8,
        color="#4E554C",
    )
    path = out_dir / f"{result.phase}_aerial_2p5d_map_clean.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def save_main_figure(pre: OrthomapResult, post: OrthomapResult, out_dir: Path) -> Path:
    fig = plt.figure(figsize=(16.0, 9.8), dpi=230)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.28, wspace=0.075)

    panels = [
        ("Aerial UAV RGB - Pre-defoliation", pre.rgb_display, "Canopy-dominant rows before defoliation.", None),
        ("Aerial UAV RGB - Post-defoliation", post.rgb_display, "Exposed lint and row structure after defoliation.", None),
        ("Aerial Morphology Field Proxy", cm.viridis(post.height)[..., :3], "Image-derived relief emphasizes exposed lint and row texture.", None),
        ("Pre Semantic Proxy", semantic_proxy(pre.rgb_surface, "pre"), "Muted classes: canopy, soil/residue, and visible lint.", None),
        ("Post Semantic Proxy", semantic_proxy(post.rgb_surface, "post"), "Post-defoliation proxy isolates stronger exposed-boll evidence.", None),
        ("Aerial 2.5D Map View", None, "Real RGB texture lifted by visibility-height proxy.", "surface"),
    ]

    for idx, (title, image, caption, kind) in enumerate(panels):
        if kind == "surface":
            ax = fig.add_subplot(gs[idx // 3, idx % 3], projection="3d")
            plot_surface(ax, post)
            ax.view_init(elev=48, azim=-57)
            ax.set_title("")
            ax.text2D(0.0, 1.025, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold")
            ax.text2D(0.0, -0.06, caption, transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#4E554C")
        else:
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            ax.imshow(image)
            ax.axis("off")
            add_panel_label(ax, title, caption)

    fig.suptitle(
        "Aerial Cotton Representation for Pre/Post-Defoliation Phenotyping",
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        "Real UAV imagery with morphology, semantic, and 2.5D proxy views for scouting and reconstruction target selection.",
        ha="center",
        fontsize=10.5,
        color="#394238",
    )
    fig.text(
        0.018,
        0.015,
        "Scientific boundary: this figure is an aerial 2.5D proxy scaffold. Metric boll diameter/volume requires calibrated scale, camera poses, or physical validation.",
        fontsize=8.7,
        color="#4E554C",
    )
    path = out_dir / "aerial_cotton_representation_clean_hd.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-image", type=Path, default=DEFAULT_PRE)
    parser.add_argument("--post-image", type=Path, default=DEFAULT_POST)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--display-edge", type=int, default=1700)
    parser.add_argument("--surface-edge", type=int, default=360)
    return parser.parse_args()


def main() -> None:
    configure_fonts()
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pre = build_result("pre-defoliation", args.pre_image, args.out_dir, args.display_edge, args.surface_edge)
    post = build_result("post-defoliation", args.post_image, args.out_dir, args.display_edge, args.surface_edge)

    main_path = save_main_figure(pre, post, args.out_dir)
    pre_map = save_surface_only(pre, args.out_dir)
    post_map = save_surface_only(post, args.out_dir)

    manifest = {
        "main_figure": str(main_path),
        "clean_2p5d_maps": {"pre": str(pre_map), "post": str(post_map)},
        "ply": {"pre": str(pre.ply_path), "post": str(post.ply_path)},
        "sources": {"pre": str(pre.source), "post": str(post.source)},
        "scientific_boundary": "Aerial RGB plus morphology/semantic/2.5D proxy views; not calibrated metric 3D.",
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"main_figure={main_path}")
    print(f"pre_map={pre_map}")
    print(f"post_map={post_map}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
