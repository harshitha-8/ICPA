#!/usr/bin/env python3
"""
Build a Utonia-style multimodal 3D representation figure for cotton imagery.

The figure is inspired by unified 3D foundation representation visuals:
real RGB texture, pseudo-depth, semantic/phase features, and point-cloud-like
views shown as a single composite. It is generated from the project imagery so
it remains grounded in the cotton dataset.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_image(row: dict[str, str]) -> Path:
    viewer = Path(row["viewer_image"]) if row.get("viewer_image") else None
    if viewer is not None and viewer.exists():
        return viewer
    return Path(row["source_image"])


def choose_rows(rows: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    pre = next(row for row in rows if row["phase"] == "pre")
    post = next(row for row in rows if row["phase"] == "post")
    return pre, post


def load_rgb(path: Path, width: int) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Could not read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = width / w
    return cv2.resize(rgb, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def fit(img: np.ndarray, width: int, height: int, bg=(255, 255, 255)) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def cover(img: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(width / w, height / h)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    y = max(0, (resized.shape[0] - height) // 2)
    x = max(0, (resized.shape[1] - width) // 2)
    return resized[y : y + height, x : x + width]


def pseudo_height(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    value = np.max(rgb_f, axis=2)
    sat = np.max(rgb_f, axis=2) - np.min(rgb_f, axis=2)
    white = np.clip((value - 0.48) * 2.2, 0, 1) * np.clip((0.55 - sat) * 2.2, 0, 1)
    green = np.clip((g - r) * 2.3, 0, 1)
    return cv2.GaussianBlur(0.25 + 1.25 * white + 0.45 * green, (0, 0), sigmaX=1.3)


def depth_colormap(height: np.ndarray) -> np.ndarray:
    norm = cv2.normalize(height, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cmap = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)


def semantic_map(rgb: np.ndarray, phase: str) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    rgb_f = rgb.astype(np.float32)
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    exg = 2 * g - r - b

    out = np.full_like(rgb, 245)
    soil = (val < 135) | ((r > g) & (r > b))
    canopy = exg > 10
    lint = (sat < 90) & (val > 145)
    out[soil] = (156, 120, 92)
    out[canopy] = (54, 170, 92) if phase == "pre" else (128, 150, 120)
    out[lint] = (246, 85, 150) if phase == "pre" else (80, 210, 235)
    return cv2.medianBlur(out, 5)


def point_cloud_panel(rgb: np.ndarray, height: np.ndarray, width: int, panel_h: int, phase: str) -> np.ndarray:
    h, w = rgb.shape[:2]
    stride = 5
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    z = height[yy, xx]
    x2 = ((xx - w / 2) * 0.85 + (yy - h / 2) * 0.22).reshape(-1)
    y2 = ((yy - h / 2) * 0.42 - z * 110).reshape(-1)
    colors = rgb[yy, xx].reshape(-1, 3)
    x2 = (x2 - x2.min()) / (x2.max() - x2.min() + 1e-6)
    y2 = (y2 - y2.min()) / (y2.max() - y2.min() + 1e-6)
    canvas = np.full((panel_h, width, 3), 255, dtype=np.uint8)
    px = (x2 * (width - 80) + 40).astype(np.int32)
    py = (y2 * (panel_h - 70) + 35).astype(np.int32)
    tint = np.array([1.0, 1.0, 1.0])
    if phase == "pre":
        tint = np.array([0.92, 1.04, 0.92])
    else:
        tint = np.array([1.05, 0.98, 0.92])
    colors = np.clip(colors.astype(np.float32) * tint, 0, 255).astype(np.uint8)
    order = np.argsort(py)
    for pxi, pyi, col in zip(px[order], py[order], colors[order]):
        cv2.circle(canvas, (int(pxi), int(pyi)), 1, tuple(int(c) for c in col[::-1]), -1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def label(panel: np.ndarray, text: str) -> np.ndarray:
    out = panel.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], 34), (255, 255, 255), -1)
    out = cv2.addWeighted(overlay, 0.88, out, 0.12, 0)
    cv2.putText(out, text, (13, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA)
    return out


def build_figure(pre: np.ndarray, post: np.ndarray, out_path: Path) -> None:
    pre_h = pseudo_height(pre)
    post_h = pseudo_height(post)

    rgb_blend = np.concatenate([cover(pre, 450, 360), cover(post, 450, 360)], axis=1)
    depth_blend = np.concatenate([cover(depth_colormap(pre_h), 450, 360), cover(depth_colormap(post_h), 450, 360)], axis=1)
    top_left = label(rgb_blend, "UAV RGB: pre-defoliation to post-defoliation")
    top_right = label(depth_blend, "morphology field proxy: canopy relief and exposed lint")

    mid_left = label(cover(semantic_map(pre, "pre"), 900, 300), "pre semantic feature proxy: canopy, soil, lint candidates")
    mid_right = label(cover(semantic_map(post, "post"), 900, 300), "post semantic feature proxy: exposed boll structure")

    bottom_left = label(fit(pre, 430, 300), "source crop")
    bottom_mid_left = label(fit(depth_colormap(post_h), 430, 300), "pseudo-depth crop")
    bottom_mid_right = label(point_cloud_panel(pre, pre_h, 430, 300, "pre"), "pre point-cloud view")
    bottom_right = label(point_cloud_panel(post, post_h, 430, 300, "post"), "post point-cloud view")

    canvas = np.full((1080, 1800, 3), 255, dtype=np.uint8)
    canvas[0:360, 0:900] = top_left
    canvas[0:360, 900:1800] = top_right
    canvas[374:674, 0:900] = mid_left
    canvas[374:674, 900:1800] = mid_right
    canvas[700:1000, 20:450] = bottom_left
    canvas[700:1000, 470:900] = bottom_mid_left
    canvas[700:1000, 920:1350] = bottom_mid_right
    canvas[700:1000, 1370:1800] = bottom_right

    cv2.putText(canvas, "Unified cotton 3D representation scaffold: RGB, morphology, semantic proxy, and point-cloud views", (26, 1038), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Illustrative project figure generated from real pre/post UAV frames; calibrated depth and Gaussian splats replace proxies after reconstruction.", (26, 1064), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (75, 75, 75), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-width", type=int, default=1000)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    pre_row, post_row = choose_rows(rows)
    pre = load_rgb(resolve_image(pre_row), args.image_width)
    post = load_rgb(resolve_image(post_row), args.image_width)
    build_figure(pre, post, args.out)


if __name__ == "__main__":
    main()
