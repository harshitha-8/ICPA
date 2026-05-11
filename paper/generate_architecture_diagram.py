#!/usr/bin/env python3
"""Generate a paper-ready architecture overview for the ICPA cotton project.

The figure is intentionally honest about the current system: calibrated 3D
metrology is shown as the validation path, while the implemented MVP is shown
as a mask-guided 2.5D/proxy review pipeline.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageEnhance, ImageOps


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


PALETTE = {
    "ink": "#1d1f23",
    "muted": "#5f6670",
    "line": "#2f3b46",
    "input": "#eef7f1",
    "detect": "#fff2cc",
    "mask": "#f3e8ff",
    "geom": "#e8f1ff",
    "trait": "#e8fbf8",
    "eval": "#fff0f0",
    "paper": "#ffffff",
    "accent": "#1f77b4",
    "post": "#2a9d8f",
    "pre": "#8ab17d",
    "warn": "#c65f3a",
}


def crop_center(path: Path, size: tuple[int, int]) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    return ImageOps.fit(im, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def load_panel(path: Path, size: tuple[int, int], boost: float = 1.0) -> Image.Image:
    if not path.exists():
        im = Image.new("RGB", size, "#f4f4f4")
        return im
    im = crop_center(path, size)
    if boost != 1.0:
        im = ImageEnhance.Contrast(im).enhance(boost)
        im = ImageEnhance.Sharpness(im).enhance(1.15)
    return im


def add_image(ax, path: Path, xy: tuple[float, float], zoom: float, size: tuple[int, int], label: str | None = None) -> None:
    im = load_panel(path, size, boost=1.05)
    oi = OffsetImage(im, zoom=zoom)
    ab = AnnotationBbox(oi, xy, frameon=True, bboxprops={"edgecolor": "#d0d4d8", "linewidth": 0.8})
    ax.add_artist(ab)
    if label:
        ax.text(xy[0], xy[1] - 0.062, label, ha="center", va="top", fontsize=8.5, color=PALETTE["muted"])


def box(ax, xy, wh, title: str, body: str, fc: str, ec: str = "#293241", lw: float = 1.15) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        mutation_aspect=1.0,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + 0.012, y + h - 0.026, title, ha="left", va="top", fontsize=10.5, weight="bold", color=PALETTE["ink"])
    wrapped = "\n".join("\n".join(textwrap.wrap(part, width=36)) for part in body.split("\n"))
    ax.text(x + 0.012, y + h - 0.065, wrapped, ha="left", va="top", fontsize=8.1, color=PALETTE["ink"], linespacing=1.08)


def arrow(ax, start, end, color: str = "#34495e", style: str = "solid", rad: float = 0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            linestyle=style,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=3,
        )
    )


def mini_mask(ax, x: float, y: float) -> None:
    ax.add_patch(Rectangle((x, y), 0.105, 0.080, facecolor="#313131", edgecolor="#1a1a1a", lw=0.9))
    blobs = [
        (0.020, 0.046, 0.017, 0.014),
        (0.044, 0.053, 0.018, 0.015),
        (0.070, 0.038, 0.020, 0.016),
        (0.034, 0.026, 0.014, 0.012),
    ]
    for bx, by, bw, bh in blobs:
        ax.add_patch(FancyBboxPatch((x + bx, y + by), bw, bh, boxstyle="round,pad=0.003,rounding_size=0.014", fc="#f8fbff", ec="#c2c9d0", lw=0.4))
    ax.text(x + 0.052, y - 0.010, "lint mask", ha="center", va="top", fontsize=7.5, color=PALETTE["muted"])


def mini_point_cloud(ax, x: float, y: float) -> None:
    ax.add_patch(Rectangle((x, y), 0.118, 0.082, facecolor="#f8fafc", edgecolor="#d0d4d8", lw=0.8))
    pts = [
        (0.016, 0.030, "#735f45"),
        (0.030, 0.050, "#ffffff"),
        (0.045, 0.046, "#dfe8ec"),
        (0.058, 0.032, "#3f6d39"),
        (0.072, 0.056, "#ffffff"),
        (0.088, 0.035, "#b9c8c8"),
        (0.100, 0.046, "#6c6559"),
    ]
    for px, py, c in pts:
        ax.scatter([x + px], [y + py], s=28, c=c, edgecolors="#46515b", linewidths=0.35, zorder=5)
    ax.plot([x + 0.015, x + 0.104], [y + 0.020, y + 0.025], color="#9aa2aa", lw=0.9)
    ax.text(x + 0.059, y - 0.010, "mask-to-3D review", ha="center", va="top", fontsize=7.5, color=PALETTE["muted"])


def main() -> None:
    fig, ax = plt.subplots(figsize=(15.2, 8.2), dpi=160)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.015, 0.965, "Mask-Guided 3D Cotton Boll Phenotyping", fontsize=17, weight="bold", color=PALETTE["ink"])
    ax.text(
        0.015,
        0.928,
        "Pre/post defoliation is modeled as a visibility intervention; calibrated 3D metrology remains the validation path.",
        fontsize=10.3,
        color=PALETTE["muted"],
    )

    # Panel rails
    ax.text(0.018, 0.872, "A  UAV evidence", fontsize=11, weight="bold")
    ax.text(0.300, 0.872, "B  Perception", fontsize=11, weight="bold")
    ax.text(0.534, 0.872, "C  Geometry + traits", fontsize=11, weight="bold")
    ax.text(0.792, 0.872, "D  Evaluation + reporting", fontsize=11, weight="bold")
    ax.plot([0.015, 0.985], [0.852, 0.852], color="#222222", lw=1.1)

    pre = REPO / "outputs/metrics/measurement_ready_bolls/top_crops/0001_pre_DJI_20250929095941_0370_D_cand371.jpg"
    post = REPO / "outputs/metrics/measurement_ready_bolls/top_crops/0002_post_DJI_20250929124623_0166_D_cand333.jpg"
    crop1 = REPO / "outputs/metrics/measurement_ready_bolls/top_crops/0002_post_DJI_20250929124623_0166_D_cand333.jpg"
    crop2 = REPO / "outputs/metrics/measurement_ready_bolls/top_crops/0001_pre_DJI_20250929095941_0370_D_cand371.jpg"
    grid_fig = REPO / "outputs/experiments/icpa_paper_metrics/figures/plot_grid_candidate_heatmaps.png"
    p25d_fig = REPO / "outputs/figures/uav_orthomap_3d/pre_post_uav_2p5d_orthomap.png"

    add_image(ax, pre, (0.075, 0.755), 0.115, (360, 245), "pre-defoliation")
    add_image(ax, post, (0.205, 0.755), 0.115, (360, 245), "post-defoliation")
    ax.text(0.140, 0.655, "phase-aware UAV frames", fontsize=8.5, ha="center", color=PALETTE["muted"])

    box(ax, (0.305, 0.690), (0.145, 0.128), "Candidate detector", "CLAHE + top-hat filters\nOtsu + contour/color gates", PALETTE["detect"])
    box(ax, (0.470, 0.690), (0.145, 0.128), "SAM-style masks", "box prompts isolate lint\nSAM/SAM2 can replace this slot", PALETTE["mask"])
    box(ax, (0.635, 0.690), (0.145, 0.128), "Proxy 3D review", "mask pixels enter 2.5D scene\ncalibrated SfM/3DGS is metric path", PALETTE["geom"])
    box(ax, (0.800, 0.690), (0.165, 0.128), "Structured outputs", "count + visibility\nproxy diameter/volume\nplot-cell summaries", PALETTE["trait"])

    arrow(ax, (0.255, 0.755), (0.305, 0.760))
    arrow(ax, (0.450, 0.760), (0.470, 0.760))
    arrow(ax, (0.615, 0.760), (0.635, 0.760))
    arrow(ax, (0.780, 0.760), (0.800, 0.760))

    # Real crop strip and mask/3D icons
    add_image(ax, crop1, (0.360, 0.575), 0.115, (170, 170), "ranked post crop")
    add_image(ax, crop2, (0.480, 0.575), 0.115, (170, 170), "ranked pre crop")
    mini_mask(ax, 0.555, 0.535)
    mini_point_cloud(ax, 0.675, 0.535)
    arrow(ax, (0.505, 0.575), (0.555, 0.575))
    arrow(ax, (0.660, 0.575), (0.675, 0.575))

    # Middle architecture formula block
    ax.add_patch(FancyBboxPatch((0.030, 0.430), 0.930, 0.070, boxstyle="round,pad=0.012,rounding_size=0.012", fc="#fbfbfb", ec="#cfd5dc", lw=0.9))
    ax.text(0.050, 0.474, "Measurement record", fontsize=10.5, weight="bold", color=PALETTE["ink"])
    ax.text(0.220, 0.474, "r_i = {box, mask, visibility, length, width, diameter, volume, readiness, plot cell}", fontsize=11.2, color=PALETTE["ink"])
    ax.text(
        0.220,
        0.444,
        "where dimensions and volume are reported as proxy traits until scale, pose, or physical measurements are validated.",
        fontsize=9.1,
        color=PALETTE["muted"],
    )

    # Bottom panels: plot mapping, local 3D, evaluation.
    add_image(ax, p25d_fig, (0.180, 0.235), 0.165, (520, 300), "UAV orthomap → 2.5D review")
    add_image(ax, grid_fig, (0.485, 0.235), 0.150, (560, 310), "row/column plot-grid aggregation")

    box(ax, (0.710, 0.250), (0.265, 0.155), "Evaluation protocol", "Count: P/R/F1, MAE, RMSE\nMask: IoU, boundary F1\n3D: reproj. + completeness\nTraits: size/volume error\nDecision: schema + expert agreement", PALETTE["eval"])
    box(ax, (0.710, 0.078), (0.265, 0.132), "Agronomist-in-the-loop", "Consumes measured traits only.\nDoes not perform reconstruction.\nCompare open-source LLMs for faithful summaries.", "#f6f7ff")
    arrow(ax, (0.606, 0.235), (0.720, 0.310))
    arrow(ax, (0.842, 0.250), (0.842, 0.210))

    # Calibration boundary.
    ax.add_patch(FancyBboxPatch((0.030, 0.046), 0.630, 0.066, boxstyle="round,pad=0.010,rounding_size=0.012", fc="#fffdf7", ec=PALETTE["warn"], lw=1.0, linestyle="--"))
    ax.text(0.045, 0.090, "Claim boundary", fontsize=10.3, weight="bold", color=PALETTE["warn"])
    ax.text(
        0.170,
        0.096,
        "Current MVP: proxy mask-to-3D review.\nMetric 3D requires GSD/GCP/camera pose, multi-view reconstruction, RGB-D, or physical boll measurements.",
        fontsize=8.2,
        color=PALETTE["ink"],
        va="top",
    )

    # Tiny legend.
    legend = [("input", PALETTE["input"]), ("detector", PALETTE["detect"]), ("mask", PALETTE["mask"]), ("geometry", PALETTE["geom"]), ("trait/eval", PALETTE["trait"])]
    lx = 0.720
    for i, (name, color) in enumerate(legend):
        ax.add_patch(Rectangle((lx + 0.048 * i, 0.034), 0.014, 0.014, fc=color, ec="#9aa2aa", lw=0.4))
        ax.text(lx + 0.017 + 0.048 * i, 0.041, name, fontsize=7.2, va="center", color=PALETTE["muted"])

    for ext in ["png", "pdf", "svg"]:
        path = OUT / f"icpa_cotton_architecture_overview.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        print(path)


if __name__ == "__main__":
    main()
