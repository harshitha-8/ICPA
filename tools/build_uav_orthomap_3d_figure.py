#!/usr/bin/env python3
"""Build data-grounded 2.5D UAV orthomap figures for pre/post cotton imagery.

This script does not claim calibrated 3D reconstruction. It converts real UAV
RGB images into a reproducible, image-derived canopy/visibility height surface
for inspection and figure planning. The exported PLY files are suitable for
viewer tests, while the PNG figure is meant to replace synthetic/AI-looking
mockups with a scientifically honest map-like view.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from PIL import Image, ImageFilter


DATASET_ROOT = Path("/Volumes/T9/ICML")
DEFAULT_PRE = DATASET_ROOT / "Part_one_pre_def_rgb" / "DJI_20250929095743_0311_D.JPG"
DEFAULT_POST = DATASET_ROOT / "205_Post_Def_rgb" / "DJI_20250929124505_0127_D.JPG"
OUTPUT_DIR = Path("/Volumes/T9/ICPA/outputs/figures/uav_orthomap_3d")


@dataclass(frozen=True)
class OrthomapResult:
    phase: str
    source: Path
    rgb_display: np.ndarray
    rgb_surface: np.ndarray
    height: np.ndarray
    vegetation: np.ndarray
    lint: np.ndarray
    ply_path: Path


def resize_long_edge(image: Image.Image, long_edge: int, resample: int = Image.Resampling.LANCZOS) -> Image.Image:
    width, height = image.size
    scale = long_edge / max(width, height)
    if scale >= 1:
        return image.copy()
    return image.resize((int(width * scale), int(height * scale)), resample)


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    lo = float(np.percentile(values, 1))
    hi = float(np.percentile(values, 99))
    if hi <= lo + 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def load_rgb(path: Path, long_edge: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = resize_long_edge(image, long_edge)
    return np.asarray(image, dtype=np.uint8)


def smooth01(values: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray((normalize(values) * 255).astype(np.uint8), mode="L")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def estimate_canopy_visibility_height(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a deterministic image-derived height proxy.

    The proxy intentionally combines vegetation, exposed cotton lint, and local
    texture. It is an inspection surface, not a metric canopy-height model.
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    value = np.max(rgb_f, axis=2)
    saturation = np.max(rgb_f, axis=2) - np.min(rgb_f, axis=2)
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    vegetation = normalize(2.0 * g - r - b)
    lint = np.clip((value - 0.47) / 0.40, 0.0, 1.0) * np.clip((0.55 - saturation) / 0.55, 0.0, 1.0)

    gy, gx = np.gradient(gray)
    texture = normalize(np.sqrt(gx * gx + gy * gy))

    row_prior = np.linspace(0.12, 0.0, rgb.shape[0], dtype=np.float32)[:, None]
    height = 0.42 * smooth01(vegetation, 2.0)
    height += 0.38 * smooth01(lint, 1.4)
    height += 0.16 * smooth01(texture, 1.0)
    height += 0.04 * row_prior
    height = smooth01(height, 1.2)
    return normalize(height), normalize(vegetation), normalize(lint)


def write_ply(path: Path, rgb: np.ndarray, height: np.ndarray, xy_scale: float = 1.0, z_scale: float = 0.22) -> None:
    h, w = height.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx.astype(np.float32) / max(w - 1, 1) - 0.5) * xy_scale
    y = (0.5 - yy.astype(np.float32) / max(h - 1, 1)) * xy_scale * h / max(w, 1)
    z = height.astype(np.float32) * z_scale

    points = np.column_stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)])
    colors = rgb.reshape(-1, 3)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for (px, py, pz), (cr, cg, cb) in zip(points, colors):
            f.write(f"{px:.6f} {py:.6f} {pz:.6f} {int(cr)} {int(cg)} {int(cb)}\n")


def build_result(phase: str, source: Path, output_dir: Path, display_edge: int, surface_edge: int) -> OrthomapResult:
    rgb_display = load_rgb(source, display_edge)
    rgb_surface = load_rgb(source, surface_edge)
    height, vegetation, lint = estimate_canopy_visibility_height(rgb_surface)
    ply_path = output_dir / f"{phase}_uav_2p5d_orthomap.ply"
    write_ply(ply_path, rgb_surface, height)
    return OrthomapResult(
        phase=phase,
        source=source,
        rgb_display=rgb_display,
        rgb_surface=rgb_surface,
        height=height,
        vegetation=vegetation,
        lint=lint,
        ply_path=ply_path,
    )


def plot_surface(ax: plt.Axes, result: OrthomapResult) -> None:
    h, w = result.height.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = xx / max(w - 1, 1)
    y = 1.0 - yy / max(h - 1, 1)
    z = result.height * 0.28
    facecolors = result.rgb_surface.astype(np.float32) / 255.0
    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        facecolors=facecolors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax.view_init(elev=54, azim=-58)
    ax.set_box_aspect((1.0, 0.72, 0.23))
    ax.set_axis_off()
    ax.set_title(f"{result.phase.title()} 2.5D UAV map", fontsize=12, fontweight="bold", pad=2)


def build_figure(pre: OrthomapResult, post: OrthomapResult, output_dir: Path) -> Path:
    fig = plt.figure(figsize=(15.5, 9.2), dpi=190)
    fig.patch.set_facecolor("white")
    results = [pre, post]

    for row, result in enumerate(results):
        ax_rgb = fig.add_subplot(2, 3, row * 3 + 1)
        ax_rgb.imshow(result.rgb_display)
        ax_rgb.set_title(f"{result.phase.title()} UAV RGB", fontsize=12, fontweight="bold")
        ax_rgb.axis("off")

        ax_height = fig.add_subplot(2, 3, row * 3 + 2)
        height_rgb = cm.viridis(result.height)[..., :3]
        ax_height.imshow(height_rgb)
        ax_height.set_title("Image-derived canopy/visibility height", fontsize=12, fontweight="bold")
        ax_height.axis("off")

        ax_surface = fig.add_subplot(2, 3, row * 3 + 3, projection="3d")
        plot_surface(ax_surface, result)

    fig.suptitle(
        "Pre/Post Defoliation Cotton UAV Orthomap-To-2.5D Map\n"
        "Generated from real UAV RGB frames; height is a canopy/lint visibility proxy, not calibrated metric 3D.",
        fontsize=15,
        fontweight="bold",
        y=0.982,
    )
    fig.text(
        0.02,
        0.015,
        f"Pre source: {pre.source.name} | Post source: {post.source.name} | "
        "Next scientific step: COLMAP/NAS3R/3DGS on overlapping image subsets for calibrated local boll geometry.",
        fontsize=8.5,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    output_path = output_dir / "pre_post_uav_2p5d_orthomap.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_single_surface_figure(result: OrthomapResult, output_dir: Path) -> Path:
    fig = plt.figure(figsize=(9.6, 7.2), dpi=220)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    plot_surface(ax, result)
    ax.set_title(
        f"{result.phase.title()} Cotton UAV 2.5D Orthomap\n"
        "Real RGB texture with image-derived canopy/lint visibility height",
        fontsize=12,
        fontweight="bold",
        pad=4,
    )
    output_path = output_dir / f"{result.phase}_uav_2p5d_map_view.png"
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output_path


def write_manifest(
    output_dir: Path,
    pre: OrthomapResult,
    post: OrthomapResult,
    figure_path: Path,
    pre_map_path: Path,
    post_map_path: Path,
) -> Path:
    manifest = {
        "artifact_type": "image-derived 2.5D UAV orthomap",
        "scientific_boundary": (
            "This is not calibrated 3D reconstruction. It is a deterministic "
            "canopy/lint visibility surface from real UAV RGB images."
        ),
        "figure": str(figure_path),
        "map_views": {
            "pre": str(pre_map_path),
            "post": str(post_map_path),
        },
        "outputs": {
            "pre_ply": str(pre.ply_path),
            "post_ply": str(post.ply_path),
        },
        "sources": {
            "pre": str(pre.source),
            "post": str(post.source),
        },
        "next_step": (
            "Run COLMAP/SfM-MVS, NAS3R, or local 3DGS on overlapping pre/post "
            "subsets before claiming metric boll diameter or volume."
        ),
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-image", type=Path, default=DEFAULT_PRE)
    parser.add_argument("--post-image", type=Path, default=DEFAULT_POST)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--display-edge", type=int, default=1150)
    parser.add_argument("--surface-edge", type=int, default=260)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pre = build_result("pre-defoliation", args.pre_image, args.out_dir, args.display_edge, args.surface_edge)
    post = build_result("post-defoliation", args.post_image, args.out_dir, args.display_edge, args.surface_edge)
    figure_path = build_figure(pre, post, args.out_dir)
    pre_map_path = build_single_surface_figure(pre, args.out_dir)
    post_map_path = build_single_surface_figure(post, args.out_dir)
    manifest_path = write_manifest(args.out_dir, pre, post, figure_path, pre_map_path, post_map_path)
    print(f"figure={figure_path}")
    print(f"pre_map={pre_map_path}")
    print(f"post_map={post_map_path}")
    print(f"pre_ply={pre.ply_path}")
    print(f"post_ply={post.ply_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
