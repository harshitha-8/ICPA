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
        rows.append(
            {
                "id": idx,
                "x": cand.x,
                "y": cand.y,
                "diameter_px": round(float(diameter_px), 2),
                "diameter_cm_proxy": round(float(diameter_cm), 3),
                "volume_cm3_proxy": round(float(volume_cm3), 3),
                "visibility_proxy": round(visibility, 3),
                "depth_score": round(depth_score, 3),
            }
        )
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
        "diameter_px",
        "diameter_cm_proxy",
        "volume_cm3_proxy",
        "visibility_proxy",
        "depth_score",
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
        pad = max(8, int(0.35 * max(cand.width, cand.height)))
        x0 = max(0, cand.x - pad)
        y0 = max(0, cand.y - pad)
        x1 = min(w, cand.x + cand.width + pad)
        y1 = min(h, cand.y + cand.height + pad)
        if x1 <= x0 or y1 <= y0:
            continue

        crop = rgb[y0:y1, x0:x1].copy()
        local_x0 = cand.x - x0
        local_y0 = cand.y - y0
        local_x1 = local_x0 + cand.width
        local_y1 = local_y0 + cand.height
        cv2.rectangle(crop, (local_x0, local_y0), (local_x1, local_y1), (46, 170, 97), max(1, crop.shape[0] // 80))
        cv2.putText(
            crop,
            f"#{row['id']}",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
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
        depth=depth,
        original_shape=original_rgb.shape[:2],
        resized_shape=rgb.shape[:2],
        gsd_cm_per_px=gsd_cm_per_px,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{image_path.stem}_{stamp}"
    ply_path = out_dir / "scene_point_cloud.ply"
    csv_path = out_dir / "boll_measurement_proxy.csv"
    save_ply(ply_path, points, colors)
    save_measurements(csv_path, measurements)

    visible = sorted(
        measurements,
        key=lambda row: (row["visibility_proxy"], row["depth_score"]),
        reverse=True,
    )[:75]
    robust_measurements = robust_subset(measurements)
    boll_crops = extract_boll_crops(original_rgb, candidates, visible, out_dir)
    summary = {
        "image": str(image_path),
        "phase": resolved_phase,
        "adjusted_count": adjusted_count,
        "raw_candidates": len(candidates),
        "measurement_candidates": len(robust_measurements),
        "point_count": len(points),
        "gsd_cm_per_px": gsd_cm_per_px,
        "median_diameter_cm_proxy": round(float(np.median([r["diameter_cm_proxy"] for r in robust_measurements])) if robust_measurements else 0.0, 3),
        "median_volume_cm3_proxy": round(float(np.median([r["volume_cm3_proxy"] for r in robust_measurements])) if robust_measurements else 0.0, 3),
        "ply": str(ply_path),
        "measurements_csv": str(csv_path),
    }
    return {
        "summary": summary,
        "input_image": encode_image(rgb),
        "annotated_image": encode_image(annotated),
        "depth_image": encode_image(depth_preview(depth)),
        "measurements": visible,
        "boll_crops": boll_crops,
        "points": [
            [round(float(x), 5), round(float(y), 5), round(float(z), 5), int(r), int(g), int(b)]
            for (x, y, z), (r, g, b) in zip(points, colors)
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
    return [
        row
        for row in rows
        if lo <= float(row["diameter_px"]) <= hi and float(row["visibility_proxy"]) >= 0.05
    ]
