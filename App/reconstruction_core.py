#!/usr/bin/env python3
"""Lightweight cotton 3D reconstruction utilities for the local app.

This module ports the useful ideas from the earlier Cotton-3D-Reconstruction
prototype into the current ICPA repo: dataset indexing, monocular depth
fallbacks, point-cloud export, detector-guided cotton candidates, and proxy
diameter/volume estimates. The measurements are explicitly proxy values until
camera calibration or ground control provides a trusted cm-per-pixel scale.
"""

from __future__ import annotations

import base64
import csv
import io
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.cotton_boll_detector import detect_cotton_bolls, infer_phase_from_greenness

DATASET_ROOT = Path("/Volumes/T9/ICML")
OUTPUT_ROOT = REPO_ROOT / "outputs" / "app_assets"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

PHASE_DIRECTORIES = {
    "pre": [
        DATASET_ROOT / "Part_one_pre_def_rgb",
        DATASET_ROOT / "part 2_pre_def_rgb",
    ],
    "post": [
        DATASET_ROOT / "205_Post_Def_rgb",
        DATASET_ROOT / "Post_def_rgb_part1",
        DATASET_ROOT / "part3_post_def_rgb",
        DATASET_ROOT / "part4_post_def_rgb",
    ],
}


@dataclass(frozen=True)
class DatasetImage:
    phase: str
    folder: str
    path: Path

    @property
    def label(self) -> str:
        return f"{self.folder} :: {self.path.name}"


def list_dataset_images(phase: str) -> list[DatasetImage]:
    items: list[DatasetImage] = []
    for folder in PHASE_DIRECTORIES.get(phase, []):
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS and not path.name.startswith("._"):
                items.append(DatasetImage(phase=phase, folder=folder.name, path=path))
    return items


def resolve_dataset_image(phase: str, label: str | None) -> Path:
    if not label:
        images = list_dataset_images(phase)
        if not images:
            raise FileNotFoundError(f"No {phase} images found under {DATASET_ROOT}")
        return images[0].path

    for item in list_dataset_images(phase):
        if item.label == label:
            return item.path
    raise FileNotFoundError(f"Could not resolve dataset image: {label}")


def load_rgb(path: Path, long_edge: int = 960) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = long_edge / max(h, w)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return rgb


def estimate_depth(rgb: np.ndarray) -> np.ndarray:
    """A deterministic morphology-aware depth proxy for fast MVP inspection."""
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    value = np.max(rgb_f, axis=2)
    sat = np.max(rgb_f, axis=2) - np.min(rgb_f, axis=2)
    white_lint = np.clip((value - 0.45) * 2.4, 0, 1) * np.clip((0.58 - sat) * 2.0, 0, 1)
    green_canopy = np.clip((2.0 * g - r - b) * 1.7, 0, 1)
    texture = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    texture = normalize(np.abs(texture))
    row_prior = np.linspace(1.0, 0.15, rgb.shape[0], dtype=np.float32)[:, None]
    depth = 0.30 * row_prior + 0.35 * white_lint + 0.20 * green_canopy + 0.15 * texture
    return normalize(cv2.GaussianBlur(depth, (0, 0), sigmaX=1.1))


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - lo) / (hi - lo)


def depth_preview(depth: np.ndarray) -> np.ndarray:
    depth_u8 = (normalize(depth) * 255).clip(0, 255).astype(np.uint8)
    color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def depth_to_points(
    rgb: np.ndarray,
    depth: np.ndarray,
    max_points: int,
    z_scale: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx.astype(np.float32) - w / 2.0) / max(w, h)
    y = (h / 2.0 - yy.astype(np.float32)) / max(w, h)
    z = depth.astype(np.float32) * (z_scale / max(w, h))
    points = np.column_stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)])
    colors = rgb.reshape(-1, 3)
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
        points = points[idx]
        colors = colors[idx]
    return points, colors


def mask_rows_to_points(
    rgb: np.ndarray,
    depth: np.ndarray,
    candidates: list[Any],
    ready_rows: list[dict[str, Any]],
    original_shape: tuple[int, int],
    resized_shape: tuple[int, int],
    max_points: int = 9000,
) -> tuple[np.ndarray, np.ndarray]:
    """Project mask-selected boll pixels into the same proxy 3D space as depth."""
    oh, ow = original_shape
    rh, rw = resized_shape
    selected: list[tuple[int, int, int]] = []

    for row in ready_rows:
        cand_index = int(row["id"]) - 1
        if cand_index < 0 or cand_index >= len(candidates):
            continue
        cand = candidates[cand_index]
        pad = max(2, int(0.10 * max(cand.width, cand.height)))
        x0 = int(np.clip(cand.x - pad, 0, ow - 1))
        y0 = int(np.clip(cand.y - pad, 0, oh - 1))
        x1 = int(np.clip(cand.x + cand.width + pad, x0 + 1, ow))
        y1 = int(np.clip(cand.y + cand.height + pad, y0 + 1, oh))
        mask = candidate_lint_mask(rgb, cand)
        if mask.shape[:2] != (y1 - y0, x1 - x0):
            continue
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        stride = max(1, len(xs) // 80)
        for x_local, y_local in zip(xs[::stride], ys[::stride]):
            selected.append((x0 + int(x_local), y0 + int(y_local), int(row["id"])))

    if not selected:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    if len(selected) > max_points:
        indices = np.linspace(0, len(selected) - 1, max_points, dtype=np.int32)
        selected = [selected[i] for i in indices]

    coords = np.array(selected, dtype=np.float32)
    ox = coords[:, 0]
    oy = coords[:, 1]
    rx = np.clip(np.round(ox * (rw / max(ow, 1))).astype(np.int32), 0, rw - 1)
    ry = np.clip(np.round(oy * (rh / max(oh, 1))).astype(np.int32), 0, rh - 1)

    x = (rx.astype(np.float32) - rw / 2.0) / max(rw, rh)
    y = (rh / 2.0 - ry.astype(np.float32)) / max(rw, rh)
    z = depth[ry, rx].astype(np.float32) * (80.0 / max(rw, rh))
    points = np.column_stack([x, y, z])

    palette = np.array(
        [
            [255, 226, 84],
            [31, 190, 215],
            [95, 220, 140],
            [245, 125, 92],
            [175, 130, 230],
        ],
        dtype=np.uint8,
    )
    ids = coords[:, 2].astype(np.int32)
    colors = palette[ids % len(palette)]
    return points, colors


def encode_image(rgb: np.ndarray, max_width: int = 980) -> str:
    h, w = rgb.shape[:2]
    if w > max_width:
        scale = max_width / w
        rgb = cv2.resize(rgb, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def compute_measurements(
    candidates: list[Any],
    original_rgb: np.ndarray,
    depth: np.ndarray,
    original_shape: tuple[int, int],
    resized_shape: tuple[int, int],
    gsd_cm_per_px: float,
) -> list[dict[str, Any]]:
    oh, ow = original_shape
    rh, rw = resized_shape
    sx = rw / max(ow, 1)
    sy = rh / max(oh, 1)
    rows: list[dict[str, Any]] = []
    for idx, cand in enumerate(candidates, start=1):
        x0 = int(np.clip(cand.x * sx, 0, rw - 1))
        y0 = int(np.clip(cand.y * sy, 0, rh - 1))
        x1 = int(np.clip((cand.x + cand.width) * sx, x0 + 1, rw))
        y1 = int(np.clip((cand.y + cand.height) * sy, y0 + 1, rh))
        local_depth = depth[y0:y1, x0:x1]
        depth_score = float(local_depth.mean()) if local_depth.size else 0.0
        diameter_px = 0.5 * (cand.width + cand.height)
        diameter_cm = diameter_px * gsd_cm_per_px
        radius_cm = 0.5 * diameter_cm
        volume_cm3 = (4.0 / 3.0) * np.pi * radius_cm**3
        visibility = float(np.clip(cand.area / max(cand.width * cand.height, 1), 0.0, 1.0))
        lint_fraction, green_fraction, brightness_score = candidate_color_scores(original_rgb, cand)
        mask = candidate_lint_mask(original_rgb, cand)
        mask_metrics = lint_mask_metrics(mask)
        mask_length_cm = mask_metrics["length_px"] * gsd_cm_per_px
        mask_width_cm = mask_metrics["width_px"] * gsd_cm_per_px
        ellipsoid_volume_cm3 = (
            (4.0 / 3.0)
            * np.pi
            * max(mask_length_cm, 0.0)
            * max(mask_width_cm, 0.0)
            * max(mask_width_cm, 0.0)
            / 8.0
        )
        rows.append(
            {
                "id": idx,
                "x": cand.x,
                "y": cand.y,
                "width": cand.width,
                "height": cand.height,
                "diameter_px": round(float(diameter_px), 2),
                "diameter_cm_proxy": round(float(diameter_cm), 3),
                "volume_cm3_proxy": round(float(volume_cm3), 3),
                "mask_area_px": int(mask_metrics["area_px"]),
                "mask_length_px": round(float(mask_metrics["length_px"]), 2),
                "mask_width_px": round(float(mask_metrics["width_px"]), 2),
                "length_cm_proxy": round(float(mask_length_cm), 3),
                "width_cm_proxy": round(float(mask_width_cm), 3),
                "ellipsoid_volume_cm3_proxy": round(float(ellipsoid_volume_cm3), 3),
                "visibility_proxy": round(visibility, 3),
                "depth_score": round(depth_score, 3),
                "lint_fraction": round(lint_fraction, 3),
                "green_fraction": round(green_fraction, 3),
                "brightness_score": round(brightness_score, 3),
            }
        )
    return add_extraction_quality(rows)


def candidate_lint_mask(rgb: np.ndarray, cand: Any) -> np.ndarray:
    """Return the largest lint-like connected component inside one candidate box.

    This is a deterministic SAM-style placeholder: detector boxes act like
    prompts, and the mask is the extracted low-saturation bright cotton region.
    Replace this function with SAM/SAM2 inference when model weights are wired in.
    """
    h, w = rgb.shape[:2]
    pad = max(2, int(0.10 * max(cand.width, cand.height)))
    x0 = int(np.clip(cand.x - pad, 0, w - 1))
    y0 = int(np.clip(cand.y - pad, 0, h - 1))
    x1 = int(np.clip(cand.x + cand.width + pad, x0 + 1, w))
    y1 = int(np.clip(cand.y + cand.height + pad, y0 + 1, h))
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32)
    val = hsv[..., 2].astype(np.float32)
    crop_f = crop.astype(np.float32)
    r, g, b = crop_f[..., 0], crop_f[..., 1], crop_f[..., 2]
    exg = 2.0 * g - r - b

    lint = (((sat < 112) & (val > 132)) | ((sat < 155) & (val > 185))) & (exg < 52)
    mask = (lint.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == largest).astype(np.uint8) * 255)


def lint_mask_metrics(mask: np.ndarray) -> dict[str, float]:
    area_px = int(np.count_nonzero(mask))
    if area_px < 4:
        return {"area_px": 0, "length_px": 0.0, "width_px": 0.0}

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"area_px": area_px, "length_px": 0.0, "width_px": 0.0}
    contour = max(contours, key=cv2.contourArea)
    (_, _), (a, b), _ = cv2.minAreaRect(contour)
    length_px = max(float(a), float(b))
    width_px = min(float(a), float(b))
    if width_px <= 0.0:
        ys, xs = np.where(mask > 0)
        length_px = float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
        width_px = float(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    return {
        "area_px": area_px,
        "length_px": length_px,
        "width_px": width_px,
    }


def candidate_color_scores(rgb: np.ndarray, cand: Any) -> tuple[float, float, float]:
    h, w = rgb.shape[:2]
    x0 = int(np.clip(cand.x, 0, w - 1))
    y0 = int(np.clip(cand.y, 0, h - 1))
    x1 = int(np.clip(cand.x + cand.width, x0 + 1, w))
    y1 = int(np.clip(cand.y + cand.height, y0 + 1, h))
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0, 1.0, 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32)
    val = hsv[..., 2].astype(np.float32)
    crop_f = crop.astype(np.float32)
    r, g, b = crop_f[..., 0], crop_f[..., 1], crop_f[..., 2]
    exg = 2.0 * g - r - b

    lint = ((sat < 110) & (val > 135)) | ((sat < 145) & (val > 178))
    green = (exg > 18) & (g > r) & (g > b)
    bright = val > 150
    return (
        float(np.mean(lint)),
        float(np.mean(green)),
        float(np.mean(bright)),
    )


def add_extraction_quality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    diameters = np.array([float(row["diameter_px"]) for row in rows], dtype=np.float32)
    lo, median, hi = np.quantile(diameters, [0.08, 0.50, 0.92])
    span = max(float(hi - lo), 1e-6)
    for row in rows:
        diameter = float(row["diameter_px"])
        size_score = 1.0 - min(abs(diameter - float(median)) / span, 1.0)
        lint = float(row["lint_fraction"])
        green = float(row["green_fraction"])
        visibility = float(row["visibility_proxy"])
        depth = float(row["depth_score"])
        brightness = float(row["brightness_score"])
        quality = (
            0.34 * lint
            + 0.18 * visibility
            + 0.16 * depth
            + 0.14 * size_score
            + 0.10 * brightness
            + 0.08 * (1.0 - green)
        )
        if diameter < lo or diameter > hi:
            quality *= 0.72
        if green > 0.55 and lint < 0.18:
            quality *= 0.55
        row["size_score"] = round(float(size_score), 3)
        row["extraction_quality"] = round(float(np.clip(quality, 0.0, 1.0)), 3)
    return rows


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            r, g, b = [int(v) for v in color]
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r} {g} {b}\n")


def save_measurements(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "x",
        "y",
        "width",
        "height",
        "diameter_px",
        "diameter_cm_proxy",
        "volume_cm3_proxy",
        "mask_area_px",
        "mask_length_px",
        "mask_width_px",
        "length_cm_proxy",
        "width_cm_proxy",
        "ellipsoid_volume_cm3_proxy",
        "visibility_proxy",
        "depth_score",
        "lint_fraction",
        "green_fraction",
        "brightness_score",
        "size_score",
        "extraction_quality",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_boll_crops(
    rgb: np.ndarray,
    candidates: list[Any],
    measurement_rows: list[dict[str, Any]],
    out_dir: Path,
    limit: int = 36,
) -> list[dict[str, Any]]:
    """Crop high-confidence cotton-boll candidates for UI inspection."""
    crop_dir = out_dir / "boll_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    h, w = rgb.shape[:2]
    gallery: list[dict[str, Any]] = []

    for row in measurement_rows[:limit]:
        cand_index = int(row["id"]) - 1
        if cand_index < 0 or cand_index >= len(candidates):
            continue
        cand = candidates[cand_index]
        half_window = max(56, int(1.7 * max(cand.width, cand.height)))
        cx = cand.x + cand.width // 2
        cy = cand.y + cand.height // 2
        x0 = max(0, cx - half_window)
        y0 = max(0, cy - half_window)
        x1 = min(w, cx + half_window)
        y1 = min(h, cy + half_window)
        if x1 <= x0 or y1 <= y0:
            continue

        crop = rgb[y0:y1, x0:x1].copy()
        local_x0 = cand.x - x0
        local_y0 = cand.y - y0
        local_x1 = local_x0 + cand.width
        local_y1 = local_y0 + cand.height
        crop = emphasize_candidate_lint(crop, local_x0, local_y0, local_x1, local_y1)
        cv2.rectangle(crop, (local_x0, local_y0), (local_x1, local_y1), (250, 218, 72), max(1, crop.shape[0] // 70))
        crop_path = crop_dir / f"boll_{int(row['id']):04d}.jpg"
        cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        gallery.append(
            {
                **row,
                "crop_image": encode_image(crop, max_width=240),
                "crop_path": str(crop_path),
            }
        )
    return gallery


def emphasize_candidate_lint(crop: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    out = (crop.astype(np.float32) * 0.58).astype(np.uint8)
    h, w = crop.shape[:2]
    x0 = int(np.clip(x0, 0, w - 1))
    y0 = int(np.clip(y0, 0, h - 1))
    x1 = int(np.clip(x1, x0 + 1, w))
    y1 = int(np.clip(y1, y0 + 1, h))
    roi = crop[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    lint = ((sat < 115) & (val > 135)) | ((sat < 150) & (val > 180))
    highlighted = roi.copy()
    highlighted[lint] = (
        0.45 * highlighted[lint].astype(np.float32)
        + 0.55 * np.array([255, 245, 190], dtype=np.float32)
    ).astype(np.uint8)
    out[y0:y1, x0:x1] = highlighted
    return out


def create_mask_overlay(
    rgb: np.ndarray,
    candidates: list[Any],
    ready_rows: list[dict[str, Any]],
    limit: int = 120,
) -> np.ndarray:
    """Draw SAM-style cotton boll masks on top of the real image."""
    overlay = rgb.copy()
    mask_layer = rgb.copy()
    palette = [
        (22, 154, 180),
        (248, 196, 65),
        (58, 145, 95),
        (218, 104, 76),
        (122, 104, 188),
    ]
    for pos, row in enumerate(ready_rows[:limit]):
        cand_index = int(row["id"]) - 1
        if cand_index < 0 or cand_index >= len(candidates):
            continue
        cand = candidates[cand_index]
        pad = max(2, int(0.10 * max(cand.width, cand.height)))
        h, w = rgb.shape[:2]
        x0 = int(np.clip(cand.x - pad, 0, w - 1))
        y0 = int(np.clip(cand.y - pad, 0, h - 1))
        x1 = int(np.clip(cand.x + cand.width + pad, x0 + 1, w))
        y1 = int(np.clip(cand.y + cand.height + pad, y0 + 1, h))
        mask = candidate_lint_mask(rgb, cand)
        if mask.shape[:2] != (y1 - y0, x1 - x0) or np.count_nonzero(mask) < 4:
            continue

        color = np.array(palette[pos % len(palette)], dtype=np.float32)
        roi = mask_layer[y0:y1, x0:x1]
        pix = mask > 0
        roi[pix] = (0.48 * roi[pix].astype(np.float32) + 0.52 * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shifted = [cnt + np.array([[[x0, y0]]], dtype=np.int32) for cnt in contours]
        cv2.drawContours(overlay, shifted, -1, tuple(int(v) for v in color), max(1, min(h, w) // 650), cv2.LINE_AA)
        if pos < 45:
            cv2.putText(
                overlay,
                str(row["id"]),
                (cand.x, max(14, cand.y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    overlay = cv2.addWeighted(overlay, 0.62, mask_layer, 0.38, 0)
    badge = f"SAM-style lint masks: {min(len(ready_rows), limit)} candidates | proxy until SAM/SAM2 validation"
    cv2.rectangle(overlay, (0, 0), (min(rgb.shape[1], max(620, 9 * len(badge))), 44), (255, 255, 255), -1)
    cv2.putText(overlay, badge, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (20, 32, 25), 2, cv2.LINE_AA)
    return overlay


def create_extraction_overlay(
    rgb: np.ndarray,
    candidates: list[Any],
    measurement_rows: list[dict[str, Any]],
    ready_rows: list[dict[str, Any]],
) -> np.ndarray:
    overlay = rgb.copy()
    ready_ids = {int(row["id"]) for row in ready_rows}
    display_rows = sorted(
        measurement_rows,
        key=lambda row: float(row["extraction_quality"]),
        reverse=True,
    )[:180]
    for row in display_rows:
        cand = candidates[int(row["id"]) - 1]
        if int(row["id"]) in ready_ids:
            color = (250, 218, 72)
            thickness = max(2, min(rgb.shape[:2]) // 600)
        else:
            color = (140, 140, 140)
            thickness = 1
        cv2.rectangle(
            overlay,
            (cand.x, cand.y),
            (cand.x + cand.width, cand.y + cand.height),
            color,
            thickness,
        )
    badge = f"measurement-ready: {len(ready_rows)} / raw: {len(candidates)}"
    cv2.rectangle(overlay, (0, 0), (max(380, 12 * len(badge)), 42), (255, 255, 255), -1)
    cv2.putText(overlay, badge, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 32, 25), 2, cv2.LINE_AA)
    return overlay


def create_plot_grid_map(
    rgb: np.ndarray,
    ready_rows: list[dict[str, Any]],
    rows_count: int = 4,
    cols_count: int = 43,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Create a row/column plot map from measurement-ready image candidates."""
    overlay = rgb.copy()
    h, w = rgb.shape[:2]
    margin_x = int(w * 0.045)
    margin_y = int(h * 0.06)
    x0, y0 = margin_x, margin_y
    x1, y1 = w - margin_x, h - margin_y

    cell_w = (x1 - x0) / cols_count
    cell_h = (y1 - y0) / rows_count
    cell_stats: dict[tuple[int, int], list[dict[str, Any]]] = {}

    for row in ready_rows:
        cx = float(row["x"]) + 0.5 * float(row["width"])
        cy = float(row["y"]) + 0.5 * float(row["height"])
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        col = int(np.clip((cx - x0) / cell_w, 0, cols_count - 1))
        grid_row = int(np.clip((cy - y0) / cell_h, 0, rows_count - 1))
        cell_stats.setdefault((grid_row, col), []).append(row)

    tint = overlay.copy()
    tint[y0:y1, x0:x1] = (
        0.70 * tint[y0:y1, x0:x1].astype(np.float32)
        + 0.30 * np.array([34, 178, 190], dtype=np.float32)
    ).astype(np.uint8)
    overlay = cv2.addWeighted(overlay, 0.72, tint, 0.28, 0)

    cv2.rectangle(overlay, (x0, y0), (x1, y1), (230, 40, 40), max(3, min(h, w) // 450))
    for r in range(1, rows_count):
        y = int(round(y0 + r * cell_h))
        cv2.line(overlay, (x0, y), (x1, y), (35, 35, 35), max(1, min(h, w) // 900))
    for c in range(1, cols_count):
        x = int(round(x0 + c * cell_w))
        cv2.line(overlay, (x, y0), (x, y1), (235, 205, 55), max(1, min(h, w) // 900))

    cell_rows: list[dict[str, Any]] = []
    for (grid_row, col), values in sorted(cell_stats.items()):
        count = len(values)
        diameters = [float(item["diameter_cm_proxy"]) for item in values]
        volumes = [float(item["volume_cm3_proxy"]) for item in values]
        qualities = [float(item["extraction_quality"]) for item in values]
        cx = int(round(x0 + (col + 0.5) * cell_w))
        cy = int(round(y0 + (grid_row + 0.5) * cell_h))
        radius = int(np.clip(3 + np.sqrt(count) * 1.4, 4, 14))
        cv2.circle(overlay, (cx, cy), radius, (20, 210, 230), -1, cv2.LINE_AA)
        if count >= 2:
            cv2.putText(
                overlay,
                str(count),
                (cx + radius + 2, cy + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cell_rows.append(
            {
                "row": grid_row + 1,
                "column": col + 1,
                "boll_count": count,
                "mean_diameter_cm_proxy": round(float(np.mean(diameters)), 3),
                "mean_volume_cm3_proxy": round(float(np.mean(volumes)), 3),
                "mean_extraction_quality": round(float(np.mean(qualities)), 3),
            }
        )

    legend = f"Plot-grid proxy: {rows_count} rows x {cols_count} columns | mapped candidates: {sum(v['boll_count'] for v in cell_rows)}"
    cv2.rectangle(overlay, (0, 0), (min(w, max(540, 10 * len(legend))), 44), (255, 255, 255), -1)
    cv2.putText(overlay, legend, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 32, 25), 2, cv2.LINE_AA)
    return overlay, sorted(cell_rows, key=lambda item: item["boll_count"], reverse=True)


def reconstruct_dataset_image(
    phase: str,
    label: str | None,
    max_points: int,
    gsd_cm_per_px: float,
) -> dict[str, Any]:
    image_path = resolve_dataset_image(phase, label)
    original_bgr = cv2.imread(str(image_path))
    if original_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    resolved_phase = infer_phase_from_greenness(original_rgb) if phase == "auto" else phase
    annotated, adjusted_count, candidates = detect_cotton_bolls(original_rgb, resolved_phase)

    rgb = load_rgb(image_path)
    depth = estimate_depth(rgb)
    points, colors = depth_to_points(rgb, depth, max_points=max_points)
    measurements = compute_measurements(
        candidates=candidates,
        original_rgb=original_rgb,
        depth=depth,
        original_shape=original_rgb.shape[:2],
        resized_shape=rgb.shape[:2],
        gsd_cm_per_px=gsd_cm_per_px,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{image_path.stem}_{stamp}"
    ply_path = out_dir / "scene_point_cloud.ply"
    boll_ply_path = out_dir / "boll_mask_point_cloud.ply"
    csv_path = out_dir / "boll_measurement_proxy.csv"
    save_ply(ply_path, points, colors)

    robust_measurements = robust_subset(measurements)
    robust_sorted = sorted(
        robust_measurements,
        key=lambda row: float(row["extraction_quality"]),
        reverse=True,
    )
    visible = robust_sorted[:75]
    boll_points, boll_colors = mask_rows_to_points(
        rgb=original_rgb,
        depth=depth,
        candidates=candidates,
        ready_rows=robust_sorted[:160],
        original_shape=original_rgb.shape[:2],
        resized_shape=rgb.shape[:2],
    )
    save_ply(boll_ply_path, boll_points, boll_colors)
    save_measurements(csv_path, measurements)
    boll_crops = extract_boll_crops(original_rgb, candidates, robust_sorted, out_dir)
    extraction_overlay = create_extraction_overlay(original_rgb, candidates, measurements, robust_sorted)
    mask_overlay = create_mask_overlay(original_rgb, candidates, robust_sorted)
    plot_map, plot_cells = create_plot_grid_map(original_rgb, robust_sorted)
    summary = {
        "image": str(image_path),
        "phase": resolved_phase,
        "adjusted_count": adjusted_count,
        "raw_candidates": len(candidates),
        "measurement_candidates": len(robust_measurements),
        "point_count": len(points),
        "boll_mask_point_count": len(boll_points),
        "gsd_cm_per_px": gsd_cm_per_px,
        "median_diameter_cm_proxy": round(float(np.median([r["diameter_cm_proxy"] for r in robust_measurements])) if robust_measurements else 0.0, 3),
        "median_volume_cm3_proxy": round(float(np.median([r["volume_cm3_proxy"] for r in robust_measurements])) if robust_measurements else 0.0, 3),
        "median_length_cm_proxy": round(float(np.median([r["length_cm_proxy"] for r in robust_measurements])) if robust_measurements else 0.0, 3),
        "median_width_cm_proxy": round(float(np.median([r["width_cm_proxy"] for r in robust_measurements])) if robust_measurements else 0.0, 3),
        "median_ellipsoid_volume_cm3_proxy": round(float(np.median([r["ellipsoid_volume_cm3_proxy"] for r in robust_measurements])) if robust_measurements else 0.0, 3),
        "ply": str(ply_path),
        "boll_mask_ply": str(boll_ply_path),
        "measurements_csv": str(csv_path),
    }
    return {
        "summary": summary,
        "input_image": encode_image(rgb),
        "annotated_image": encode_image(annotated),
        "depth_image": encode_image(depth_preview(depth)),
        "extraction_overlay_image": encode_image(extraction_overlay),
        "mask_overlay_image": encode_image(mask_overlay),
        "plot_map_image": encode_image(plot_map),
        "measurements": visible,
        "boll_crops": boll_crops,
        "plot_cells": plot_cells[:120],
        "points": [
            [round(float(x), 5), round(float(y), 5), round(float(z), 5), int(r), int(g), int(b)]
            for (x, y, z), (r, g, b) in zip(points, colors)
        ],
        "boll_points": [
            [round(float(x), 5), round(float(y), 5), round(float(z), 5), int(r), int(g), int(b)]
            for (x, y, z), (r, g, b) in zip(boll_points, boll_colors)
        ],
    }


def dataset_payload() -> dict[str, Any]:
    return {
        phase: [{"label": item.label, "path": str(item.path)} for item in list_dataset_images(phase)]
        for phase in ("pre", "post")
    }


def robust_subset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) < 8:
        return rows
    diameters = np.array([float(row["diameter_px"]) for row in rows], dtype=np.float32)
    lo, hi = np.quantile(diameters, [0.05, 0.90])
    qualities = np.array([float(row.get("extraction_quality", 0.0)) for row in rows], dtype=np.float32)
    quality_floor = max(0.22, float(np.quantile(qualities, 0.62)))
    return [
        row
        for row in rows
        if lo <= float(row["diameter_px"]) <= hi
        and float(row["visibility_proxy"]) >= 0.05
        and float(row.get("lint_fraction", 0.0)) >= 0.05
        and float(row.get("green_fraction", 1.0)) <= 0.70
        and float(row.get("extraction_quality", 0.0)) >= quality_floor
    ]
