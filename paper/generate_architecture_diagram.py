#!/usr/bin/env python3
"""Generate a crisp CVPR/NeurIPS-style architecture figure for the ICPA paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageEnhance, ImageOps


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "ink": "#15181d",
    "muted": "#5c6470",
    "line": "#2c3440",
    "data": "#f0f7f1",
    "detect": "#fff4cf",
    "mask": "#f1e7ff",
    "geo": "#e9f2ff",
    "trait": "#e7fbf6",
    "eval": "#fff0f0",
    "llm": "#f3f5ff",
    "warn": "#c65f3a",
}


def im(path: Path, size: tuple[int, int], contrast: float = 1.04) -> Image.Image:
    if not path.exists():
        return Image.new("RGB", size, "#f3f4f6")
    img = Image.open(path).convert("RGB")
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(1.12)
    return img


def place_img(ax, path: Path, xy: tuple[float, float], size=(240, 170), zoom=0.22, ec="#cfd5dd") -> None:
    artist = OffsetImage(im(path, size), zoom=zoom)
    ab = AnnotationBbox(artist, xy, frameon=True, bboxprops={"edgecolor": ec, "linewidth": 0.85, "facecolor": "white"})
    ax.add_artist(ab)


def panel(ax, xy, wh, title: str, color: str, subtitle: str | None = None, title_size: float = 9.6) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.014",
            fc=color,
            ec=COLORS["line"],
            lw=1.25,
            zorder=1,
        )
    )
    ax.text(x + 0.014, y + h - 0.030, title, fontsize=title_size, weight="bold", va="top", color=COLORS["ink"], zorder=5, linespacing=1.0)
    if subtitle:
        ax.text(x + 0.014, y + 0.024, subtitle, fontsize=7.1, va="bottom", color=COLORS["muted"], zorder=5)


def arrow(ax, a, b, label: str | None = None, dashed: bool = False) -> None:
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=13,
            lw=1.35,
            color=COLORS["line"],
            linestyle="--" if dashed else "-",
            zorder=10,
        )
    )
    if label:
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ax.text(mx, my + 0.025, label, fontsize=7.6, ha="center", color=COLORS["muted"])


def step(ax, x: float, y: float, n: int, text: str) -> None:
    ax.add_patch(Circle((x, y), 0.014, fc=COLORS["ink"], ec="white", lw=0.8, zorder=20))
    ax.text(x, y - 0.001, str(n), fontsize=7.8, color="white", weight="bold", ha="center", va="center", zorder=21)
    ax.text(x + 0.019, y, text, fontsize=8.1, color=COLORS["ink"], va="center")


def mask_icon(ax, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(Rectangle((x, y), w, h, fc="#272727", ec="#111111", lw=0.75, zorder=3))
    blobs = [(0.18, 0.55, 0.18, 0.16), (0.42, 0.65, 0.22, 0.18), (0.68, 0.50, 0.20, 0.17), (0.36, 0.30, 0.16, 0.15)]
    for bx, by, bw, bh in blobs:
        ax.add_patch(FancyBboxPatch((x + bx * w, y + by * h), bw * w, bh * h, boxstyle="round,pad=0.004,rounding_size=0.016", fc="#f7fbff", ec="#c7cdd4", lw=0.45, zorder=4))
    ax.plot([x + 0.31 * w, x + 0.31 * w], [y + 0.25 * h, y + 0.50 * h], color="white", lw=0.7, zorder=5)
    ax.plot([x + 0.29 * w, x + 0.33 * w], [y + 0.25 * h, y + 0.25 * h], color="white", lw=0.7, zorder=5)


def pointcloud_icon(ax, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(Rectangle((x, y), w, h, fc="#fbfcfd", ec="#cfd5dd", lw=0.75, zorder=3))
    ax.plot([x + 0.12 * w, x + 0.88 * w], [y + 0.28 * h, y + 0.34 * h], color="#8e98a3", lw=0.9, zorder=4)
    pts = [(0.15, 0.38, "#735f45"), (0.28, 0.62, "#f8fbff"), (0.42, 0.55, "#dde9ec"), (0.53, 0.40, "#3b703c"), (0.66, 0.68, "#f8fbff"), (0.80, 0.48, "#b8c7c8"), (0.90, 0.60, "#686159")]
    for px, py, c in pts:
        ax.scatter([x + px * w], [y + py * h], s=24, c=c, edgecolors="#47515c", linewidths=0.35, zorder=6)


def mini_table(ax, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="#c8ced6", lw=0.75, zorder=3))
    cols = ["count", "vis.", "diam.", "vol."]
    vals = ["3106", ".64", "211", "16.1"]
    for i, col in enumerate(cols):
        cx = x + (i + 0.5) * w / 4
        ax.text(cx, y + 0.66 * h, col, fontsize=5.8, ha="center", color=COLORS["muted"], zorder=5)
        ax.text(cx, y + 0.34 * h, vals[i], fontsize=7.4, weight="bold", ha="center", color=COLORS["ink"], zorder=5)
        if i:
            ax.plot([x + i * w / 4, x + i * w / 4], [y + 0.16 * h, y + 0.86 * h], color="#d5d9df", lw=0.6, zorder=4)


def metric_strip(ax, x: float, y: float, w: float, h: float) -> None:
    names = ["Detect", "Mask", "3D", "Trait", "LLM"]
    metrics = ["P/R/F1", "IoU", "reproj.", "MAE", "schema"]
    colors = [COLORS["detect"], COLORS["mask"], COLORS["geo"], COLORS["trait"], COLORS["llm"]]
    for i, (name, metric, color) in enumerate(zip(names, metrics, colors)):
        xi = x + i * w / 5
        ax.add_patch(Rectangle((xi, y), w / 5 - 0.004, h, fc=color, ec="#c7ced7", lw=0.7, zorder=2))
        ax.text(xi + w / 10, y + 0.61 * h, name, fontsize=7.3, weight="bold", ha="center", color=COLORS["ink"])
        ax.text(xi + w / 10, y + 0.28 * h, metric, fontsize=7.0, ha="center", color=COLORS["muted"])


def main() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 5.7), dpi=240)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.018, 0.958, "Mask-Guided 3D Cotton Boll Phenotyping", fontsize=13.5, weight="bold", color=COLORS["ink"])
    ax.text(0.018, 0.920, "A compact pipeline for pre/post-defoliation UAV imagery, proxy 3D review, and measurement-ready trait evaluation.", fontsize=8.9, color=COLORS["muted"])

    # Main panels.
    y, h = 0.545, 0.265
    xs = [0.035, 0.225, 0.405, 0.585, 0.765]
    ws = [0.145, 0.135, 0.135, 0.135, 0.165]
    titles = ["UAV phase pair", "Candidate\nlocalization", "Prompted lint\nmask", "Mask-to-3D\nreview", "Trait record"]
    fills = [COLORS["data"], COLORS["detect"], COLORS["mask"], COLORS["geo"], COLORS["trait"]]
    subtitles = ["pre vs post", "bright lint proposals", "box-prompted extraction", "proxy now; metric later", "per boll + plot cell"]
    for x, w, title, fill, sub in zip(xs, ws, titles, fills, subtitles):
        panel(ax, (x, y), (w, h), title, fill, sub)

    pre = REPO / "outputs/metrics/measurement_ready_bolls/top_crops/0001_pre_DJI_20250929095941_0370_D_cand371.jpg"
    post = REPO / "outputs/metrics/measurement_ready_bolls/top_crops/0002_post_DJI_20250929124623_0166_D_cand333.jpg"
    p25d = REPO / "outputs/figures/uav_orthomap_3d/pre_post_uav_2p5d_orthomap.png"
    grid = REPO / "outputs/experiments/icpa_paper_metrics/figures/plot_grid_candidate_heatmaps.png"
    vol = REPO / "outputs/experiments/icpa_paper_metrics/figures/volume_mutation_proxy.png"

    place_img(ax, pre, (0.078, 0.658), (150, 115), 0.20)
    place_img(ax, post, (0.137, 0.658), (150, 115), 0.20)
    ax.add_patch(Rectangle((0.256, 0.595), 0.060, 0.062, fill=False, ec="#f4d03f", lw=1.1, zorder=5))
    place_img(ax, post, (0.292, 0.640), (150, 115), 0.225)
    ax.text(0.253, 0.706, "ranked crop", fontsize=7.0, color=COLORS["muted"])
    mask_icon(ax, 0.432, 0.625, 0.080, 0.084)
    pointcloud_icon(ax, 0.612, 0.625, 0.088, 0.084)
    mini_table(ax, 0.790, 0.628, 0.108, 0.080)

    for i in range(4):
        arrow(ax, (xs[i] + ws[i] + 0.008, y + 0.145), (xs[i + 1] - 0.010, y + 0.145), label=str(i + 1))

    # Lower evidence row.
    ax.text(0.035, 0.445, "Evidence used in the paper", fontsize=10.5, weight="bold", color=COLORS["ink"])
    ax.plot([0.035, 0.952], [0.429, 0.429], color="#232323", lw=0.85)
    place_img(ax, p25d, (0.170, 0.300), (520, 300), 0.185)
    place_img(ax, grid, (0.430, 0.300), (560, 310), 0.164)
    place_img(ax, vol, (0.675, 0.300), (420, 280), 0.150)
    ax.text(0.170, 0.190, "(a) UAV orthomap to 2.5D review", fontsize=8.0, ha="center", color=COLORS["muted"])
    ax.text(0.430, 0.190, "(b) row/column aggregation", fontsize=8.0, ha="center", color=COLORS["muted"])
    ax.text(0.675, 0.190, "(c) proxy volume mutation", fontsize=8.0, ha="center", color=COLORS["muted"])

    panel(ax, (0.785, 0.238), (0.178, 0.120), "Evaluation", COLORS["eval"], None, title_size=8.8)
    ax.text(0.802, 0.294, "P/R/F1  |  IoU  |  MAE", fontsize=7.0, color=COLORS["ink"], weight="bold", va="center")
    ax.text(0.802, 0.270, "reprojection + completeness", fontsize=6.6, color=COLORS["muted"], va="center")
    ax.text(0.802, 0.248, "schema + expert agreement", fontsize=6.6, color=COLORS["muted"], va="center")
    arrow(ax, (0.740, 0.300), (0.785, 0.304), label="5")

    # Boundary strip.
    ax.add_patch(FancyBboxPatch((0.035, 0.070), 0.920, 0.070, boxstyle="round,pad=0.008,rounding_size=0.010", fc="#fffdf8", ec=COLORS["warn"], lw=0.9, linestyle="--"))
    ax.text(0.052, 0.110, "Claim boundary", fontsize=9.2, weight="bold", color=COLORS["warn"], va="center")
    ax.text(0.178, 0.110, "Current system produces measurement-ready masks and proxy 2.5D review. Calibrated 3D boll metrology requires GSD/GCP/camera pose, multi-view geometry, RGB-D, or physical measurements.", fontsize=6.7, color=COLORS["ink"], va="center")

    # Tiny notation strip.
    step(ax, 0.050, 0.842, 1, "phase")
    step(ax, 0.145, 0.842, 2, "detect")
    step(ax, 0.250, 0.842, 3, "mask")
    step(ax, 0.348, 0.842, 4, "project")
    step(ax, 0.458, 0.842, 5, "evaluate")
    ax.text(0.585, 0.842, "record: {box, mask, visibility, diameter, volume, readiness, plot cell}", fontsize=8.2, color=COLORS["muted"], va="center")

    for ext in ("png", "pdf", "svg"):
        out = OUT / f"icpa_cotton_architecture_overview.{ext}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        print(out)


if __name__ == "__main__":
    main()
