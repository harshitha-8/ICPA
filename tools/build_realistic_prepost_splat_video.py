#!/usr/bin/env python3
"""
Build a realistic pre/post cotton splat-style video from real UAV imagery.

This version avoids black backgrounds, oversized anchors, and artificial UI.
It uses full-color image-derived splats with gentle camera motion so the video
looks like a reconstruction artifact rather than a synthetic dashboard.

Scientific boundary: this is still a visual 2.5D splat scaffold. True metric
Gaussian Splatting requires camera poses from COLMAP/VGGT/MASt3R and 3DGS
training. The goal here is a realistic MVP teaser using the current dataset.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class SplatScene:
    image: np.ndarray
    height: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    phase: str
    source_name: str


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_image(row: dict[str, str]) -> Path:
    viewer = Path(row["viewer_image"]) if row.get("viewer_image") else None
    if viewer is not None and viewer.exists():
        return viewer
    return Path(row["source_image"])


def choose_rows(rows: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    pre_rows = [row for row in rows if row["phase"] == "pre"]
    post_rows = [row for row in rows if row["phase"] == "post"]
    if not pre_rows or not post_rows:
        raise RuntimeError("Manifest must contain at least one pre and one post image.")
    return pre_rows[0], post_rows[0]


def resize_rgb(path: Path, width: int) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = width / w
    return cv2.resize(rgb, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def estimate_height(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    value = np.max(rgb_f, axis=2)
    saturation = np.max(rgb_f, axis=2) - np.min(rgb_f, axis=2)
    white_lint = np.clip((value - 0.50) * 2.0, 0.0, 1.0) * np.clip((0.55 - saturation) * 2.5, 0.0, 1.0)
    green_vegetation = np.clip((g - r) * 2.3, 0.0, 1.0)
    soil = np.clip((r - b) * 1.4, 0.0, 0.35)
    height = 0.10 + 1.25 * white_lint + 0.45 * green_vegetation + 0.10 * soil
    return cv2.GaussianBlur(height, (0, 0), sigmaX=1.4)


def make_scene(row: dict[str, str], width: int, stride: int) -> SplatScene:
    path = resolve_image(row)
    rgb = resize_rgb(path, width)
    h, w = rgb.shape[:2]
    height = estimate_height(rgb)
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    scale = 0.018
    x = (xx - w / 2.0) * scale
    z = (yy - h / 2.0) * scale
    y = height[yy, xx]
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    colors = rgb[yy, xx].reshape(-1, 3).astype(np.uint8)
    return SplatScene(image=rgb, height=height, points=points, colors=colors, phase=row["phase"], source_name=path.name)


def look_at(camera: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - camera
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, np.array([0, 1, 0], dtype=np.float32))
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    return np.stack([right, up, forward], axis=0)


def project(points: np.ndarray, camera: np.ndarray, target: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = look_at(camera, target)
    cam = (points - camera) @ rot.T
    z = cam[:, 2]
    valid = z > 0.08
    focal = 0.5 * width / math.tan(math.radians(52.0) / 2.0)
    x = cam[:, 0] * focal / z + width / 2.0
    y = height / 2.0 - cam[:, 1] * focal / z
    valid &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x[valid].astype(np.int32), y[valid].astype(np.int32), valid


def fit_height(scene: SplatScene, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.cvtColor(scene.image, cv2.COLOR_RGB2BGR)
    ih, iw = bgr.shape[:2]
    scale = min(width / iw, height / ih)
    rw, rh = int(iw * scale), int(ih * scale)
    x = (width - rw) // 2
    y = (height - rh) // 2
    image_canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    height_canvas = np.zeros((height, width), dtype=np.float32)
    image_canvas[y : y + rh, x : x + rw] = cv2.resize(bgr, (rw, rh), interpolation=cv2.INTER_AREA)
    height_canvas[y : y + rh, x : x + rw] = cv2.resize(scene.height, (rw, rh), interpolation=cv2.INTER_AREA)
    height_canvas = cv2.GaussianBlur(height_canvas, (0, 0), sigmaX=2.0)
    return image_canvas, height_canvas


def fit(img: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def draw_label(canvas: np.ndarray, text: str, x: int, y: int, scale: float = 0.72) -> None:
    cv2.putText(canvas, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (25, 25, 25), 2, cv2.LINE_AA)


def render_splat(scene: SplatScene, t: float, width: int, height: int) -> np.ndarray:
    image, depth = fit_height(scene, width, height)
    depth = np.clip(depth / (float(depth.max()) + 1e-6), 0.0, 1.0)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    yaw = math.sin(t * math.pi * 2.0)
    push = math.sin(t * math.pi)
    drift = (t - 0.5) * 26.0
    map_x = xx - yaw * depth * 80.0 - drift
    map_y = yy + push * depth * 46.0 - depth * 30.0
    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    focus = cv2.GaussianBlur(warped, (0, 0), sigmaX=0.7 + 1.2 * (1.0 - depth.mean()))
    canvas = cv2.addWeighted(warped, 0.88, focus, 0.12, 0)
    vignette = np.clip(1.06 - (((xx - width / 2) / width) ** 2 + ((yy - height / 2) / height) ** 2) * 1.25, 0.72, 1.0)
    canvas = np.clip(canvas.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)
    phase = "Pre-defoliation" if scene.phase == "pre" else "Post-defoliation"
    cv2.rectangle(canvas, (0, 0), (width, 118), (245, 245, 245), -1)
    draw_label(canvas, f"{phase} image-based splat reconstruction", 44, 55, 0.82)
    draw_label(canvas, "real UAV texture with depth-parallax; metric 3DGS still requires calibrated poses", 44, 92, 0.55)
    return canvas


def compose_original(scene: SplatScene, alpha: float, width: int, height: int) -> np.ndarray:
    bgr = cv2.cvtColor(scene.image, cv2.COLOR_RGB2BGR)
    frame = fit(bgr, width, height)
    phase = "Pre-defoliation UAV frame" if scene.phase == "pre" else "Post-defoliation UAV frame"
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 125), (255, 255, 255), -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    draw_label(frame, phase, 44, 58, 0.86)
    draw_label(frame, scene.source_name, 44, 96, 0.55)
    if alpha < 1.0:
        frame = cv2.addWeighted(frame, alpha, np.full_like(frame, 245), 1.0 - alpha, 0)
    return frame


def build_frames(pre: SplatScene, post: SplatScene, width: int, height: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for i in range(35):
        frames.append(compose_original(pre, 1.0, width, height))
    for i in range(75):
        frames.append(render_splat(pre, i / 74.0, width, height))
    for i in range(25):
        a = i / 24.0
        left = render_splat(pre, 0.95, width, height)
        right = compose_original(post, 1.0, width, height)
        frames.append(cv2.addWeighted(left, 1.0 - a, right, a, 0))
    for i in range(35):
        frames.append(compose_original(post, 1.0, width, height))
    for i in range(75):
        frames.append(render_splat(post, i / 74.0, width, height))
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a realistic pre/post splat-style cotton video.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-video", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--image-width", type=int, default=1100)
    parser.add_argument("--stride", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    pre_row, post_row = choose_rows(rows)
    pre = make_scene(pre_row, args.image_width, args.stride)
    post = make_scene(post_row, args.image_width, args.stride)
    frames = build_frames(pre, post, 1920, 1080)
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(args.frames_dir / f"frame_{i:05d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"frames: {len(frames)}")
    print(f"pre points: {len(pre.points):,}; post points: {len(post.points):,}")
    print(f"encode: ffmpeg -y -framerate {args.fps} -i {args.frames_dir}/frame_%05d.jpg -c:v libx264 -pix_fmt yuv420p -movflags +faststart {args.out_video}")


if __name__ == "__main__":
    main()
