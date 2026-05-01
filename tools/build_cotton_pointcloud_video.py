#!/usr/bin/env python3
"""
Build a cotton-field point-cloud flythrough MVP from selected images.

This is a visual reconstruction scaffold, not a metric SfM result. It converts
selected UAV frames into colored 2.5D point-cloud tiles, adds simple cotton-boll
anchors from bright lint regions, and renders a camera flythrough. Real COLMAP,
VGGT, MASt3R, or Gaussian-splat geometry can later replace this scaffold while
keeping the same video language.
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
class SceneCloud:
    points: np.ndarray
    colors: np.ndarray
    anchors: np.ndarray
    anchor_phases: list[str]


def read_manifest(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit > 0:
        post = [r for r in rows if r["phase"] == "post"]
        pre = [r for r in rows if r["phase"] == "pre"]
        paired = []
        for idx in range(max(len(post), len(pre))):
            if idx < len(post):
                paired.append(post[idx])
            if idx < len(pre):
                paired.append(pre[idx])
        rows = paired[:limit]
    return rows


def image_path(row: dict[str, str]) -> Path:
    viewer = Path(row["viewer_image"]) if row.get("viewer_image") else None
    if viewer is not None and viewer.exists():
        return viewer
    return Path(row["source_image"])


def resize_for_scene(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = width / w
    return cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def cotton_height(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    value = np.max(rgb_f, axis=2)
    sat = np.max(rgb_f, axis=2) - np.min(rgb_f, axis=2)
    white_lint = np.clip((value - 0.55) * 2.2, 0.0, 1.0) * np.clip((0.45 - sat) * 3.0, 0.0, 1.0)
    green_canopy = np.clip((g - r) * 2.5, 0.0, 1.0)
    height = 0.20 + 1.25 * white_lint + 0.55 * green_canopy
    return cv2.GaussianBlur(height, (0, 0), sigmaX=1.2)


def detect_anchor_pixels(rgb: np.ndarray, max_anchors: int) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    mask = ((sat < 75) & (val > 175)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    comps = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if 4 <= area <= 700:
            comps.append((area, centroids[i]))
    comps.sort(reverse=True, key=lambda item: item[0])
    return np.array([centroid for _, centroid in comps[:max_anchors]], dtype=np.float32)


def build_scene(rows: list[dict[str, str]], image_width: int, stride: int, anchors_per_tile: int) -> SceneCloud:
    all_points = []
    all_colors = []
    all_anchors = []
    anchor_phases: list[str] = []

    tile_gap = 7.5
    scale = 0.028
    for tile_idx, row in enumerate(rows):
        bgr = cv2.imread(str(image_path(row)))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(resize_for_scene(bgr, image_width), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        height = cotton_height(rgb)

        yy, xx = np.mgrid[0:h:stride, 0:w:stride]
        z_world = tile_idx * tile_gap + (yy - h / 2.0) * scale
        x_world = (xx - w / 2.0) * scale
        y_world = height[yy, xx] * (1.0 if row["phase"] == "post" else 1.15)
        pts = np.stack([x_world, y_world, z_world], axis=-1).reshape(-1, 3)
        cols = rgb[yy, xx].reshape(-1, 3)

        all_points.append(pts.astype(np.float32))
        all_colors.append(cols.astype(np.uint8))

        anchors = detect_anchor_pixels(rgb, anchors_per_tile)
        for cx, cy in anchors:
            ax = (cx - w / 2.0) * scale
            az = tile_idx * tile_gap + (cy - h / 2.0) * scale
            ay = float(height[int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))]) + 0.35
            all_anchors.append([ax, ay, az])
            anchor_phases.append(row["phase"])

    if not all_points:
        raise RuntimeError("No readable images found for scene.")
    return SceneCloud(
        points=np.concatenate(all_points, axis=0),
        colors=np.concatenate(all_colors, axis=0),
        anchors=np.array(all_anchors, dtype=np.float32),
        anchor_phases=anchor_phases,
    )


def look_at(camera: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - camera
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    return np.stack([right, up, forward], axis=0)


def project(points: np.ndarray, camera: np.ndarray, target: np.ndarray, width: int, height: int, fov: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = look_at(camera, target)
    rel = points - camera
    cam = rel @ rot.T
    z = cam[:, 2]
    valid = z > 0.2
    focal = 0.5 * width / math.tan(math.radians(fov) / 2.0)
    x = (cam[:, 0] * focal / z) + width / 2.0
    y = height / 2.0 - (cam[:, 1] * focal / z)
    valid &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x[valid].astype(np.int32), y[valid].astype(np.int32), valid


def draw_projected_points(canvas: np.ndarray, scene: SceneCloud, camera: np.ndarray, target: np.ndarray) -> None:
    h, w = canvas.shape[:2]
    x, y, valid = project(scene.points, camera, target, w, h, fov=58.0)
    colors = scene.colors[valid]
    if len(x) == 0:
        return
    depth = np.linalg.norm(scene.points[valid] - camera, axis=1)
    order = np.argsort(depth)[::-1]
    x, y, colors = x[order], y[order], colors[order]
    canvas[y, x] = colors[:, ::-1]
    canvas[:] = cv2.dilate(canvas, np.ones((2, 2), np.uint8), iterations=1)


def draw_anchors(canvas: np.ndarray, scene: SceneCloud, camera: np.ndarray, target: np.ndarray) -> None:
    if len(scene.anchors) == 0:
        return
    h, w = canvas.shape[:2]
    x, y, valid = project(scene.anchors, camera, target, w, h, fov=58.0)
    phases = [p for p, keep in zip(scene.anchor_phases, valid) if keep]
    for px, py, phase in zip(x, y, phases):
        color = (0, 230, 120) if phase == "post" else (0, 185, 255)
        cv2.circle(canvas, (int(px), int(py)), 5, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(px), int(py)), 9, (255, 255, 255), 1, cv2.LINE_AA)


def draw_overlay(canvas: np.ndarray, frame_idx: int, total_frames: int, scene: SceneCloud) -> None:
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 88), (10, 14, 22), -1)
    cv2.putText(canvas, "Cotton nursery / field 3D reconstruction MVP", (32, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "2.5D colored point cloud + camera flythrough + candidate boll anchors", (32, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (170, 220, 255), 2, cv2.LINE_AA)
    progress = int((canvas.shape[1] - 64) * frame_idx / max(1, total_frames - 1))
    cv2.rectangle(canvas, (32, canvas.shape[0] - 38), (32 + progress, canvas.shape[0] - 28), (0, 210, 130), -1)
    cv2.putText(canvas, f"points={len(scene.points):,} anchors={len(scene.anchors):,}", (32, canvas.shape[0] - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def render_video(scene: SceneCloud, out_video: Path, frames_dir: Path, num_frames: int, fps: int) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    z_min = float(np.min(scene.points[:, 2]))
    z_max = float(np.max(scene.points[:, 2]))
    center_x = float(np.mean(scene.points[:, 0]))

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        target = np.array([center_x, 0.65, z_min + t * (z_max - z_min)], dtype=np.float32)
        side = math.sin(t * math.pi * 2.0) * 5.0
        camera = np.array([center_x + side, 5.0, target[2] - 11.0], dtype=np.float32)
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        canvas[:] = (8, 11, 18)
        draw_projected_points(canvas, scene, camera, target)
        draw_anchors(canvas, scene, camera, target)
        draw_overlay(canvas, i, num_frames, scene)
        cv2.imwrite(str(frames_dir / f"frame_{i:05d}.jpg"), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%05d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_video),
    ]
    print(" ".join(cmd))


def write_ply(scene: SceneCloud, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(scene.points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(scene.points, scene.colors):
            f.write(f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f} {int(color[0])} {int(color[1])} {int(color[2])}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cotton point-cloud flythrough MVP.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-video", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--out-ply", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--image-width", type=int, default=560)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--anchors-per-tile", type=int, default=80)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest, args.limit)
    scene = build_scene(rows, args.image_width, args.stride, args.anchors_per_tile)
    write_ply(scene, args.out_ply)
    render_video(scene, args.out_video, args.frames_dir, args.frames, args.fps)
    print(f"scene points: {len(scene.points):,}")
    print(f"anchors: {len(scene.anchors):,}")
    print(f"ply: {args.out_ply}")
    print(f"frames: {args.frames_dir}")


if __name__ == "__main__":
    main()
