#!/usr/bin/env python3
"""
Build a pre/post-defoliation 3D reconstruction teaser video.

The video uses real cotton imagery, detection-derived boll candidates, a
monochrome 2.5D reconstruction scaffold, and proxy diameter/volume statistics.
True centimeter measurements require GSD or ground-control scale calibration;
until then, the video labels measurement values as proxy quantities.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "algorithms"))

from cotton_boll_detector import detect_candidates  # noqa: E402


@dataclass(frozen=True)
class DetectionStats:
    phase: str
    raw_count: int
    shown_count: int
    median_diameter_px: float
    median_volume_px3: float


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def image_path(row: dict[str, str]) -> Path:
    viewer = Path(row["viewer_image"]) if row.get("viewer_image") else None
    if viewer is not None and viewer.exists():
        return viewer
    return Path(row["source_image"])


def choose_pair(rows: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    pre = next(row for row in rows if row["phase"] == "pre")
    post = next(row for row in rows if row["phase"] == "post")
    return pre, post


def fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    resized = cv2.resize(img, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y = (h - resized.shape[0]) // 2
    x = (w - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def mono(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def load_rgb(row: dict[str, str], width: int) -> np.ndarray:
    bgr = cv2.imread(str(image_path(row)))
    if bgr is None:
        raise RuntimeError(f"Could not read {image_path(row)}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = width / w
    return cv2.resize(rgb, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def measure_candidates(rgb: np.ndarray, phase: str, max_show: int) -> tuple[list, DetectionStats]:
    candidates = detect_candidates(rgb, phase)
    candidates = sorted(candidates, key=lambda c: c.area, reverse=True)
    shown = candidates[:max_show]
    diameters = []
    volumes = []
    for cand in shown:
        diameter = math.sqrt(float(cand.width * cand.height))
        a = max(cand.width, cand.height) / 2.0
        b = min(cand.width, cand.height) / 2.0
        volume = 4.0 / 3.0 * math.pi * a * b * b
        diameters.append(diameter)
        volumes.append(volume)
    stats = DetectionStats(
        phase=phase,
        raw_count=len(candidates),
        shown_count=len(shown),
        median_diameter_px=float(np.median(diameters)) if diameters else 0.0,
        median_volume_px3=float(np.median(volumes)) if volumes else 0.0,
    )
    return shown, stats


def draw_text(img: np.ndarray, text: str, pos: tuple[int, int], scale: float = 0.7, color=(245, 245, 245)) -> None:
    x, y = pos
    cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_detections(panel: np.ndarray, candidates: list, source_shape: tuple[int, int], color: tuple[int, int, int]) -> None:
    ph, pw = panel.shape[:2]
    sh, sw = source_shape
    scale = min(pw / sw, ph / sh)
    ox = (pw - sw * scale) / 2.0
    oy = (ph - sh * scale) / 2.0
    for cand in candidates:
        x = int(ox + cand.x * scale)
        y = int(oy + cand.y * scale)
        w = max(2, int(cand.width * scale))
        h = max(2, int(cand.height * scale))
        cx = x + w // 2
        cy = y + h // 2
        cv2.rectangle(panel, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
        cv2.circle(panel, (cx, cy), 3, color, -1, cv2.LINE_AA)


def cotton_height(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32) / 255.0
    val = np.max(rgb_f, axis=2)
    sat = np.max(rgb_f, axis=2) - np.min(rgb_f, axis=2)
    white_lint = np.clip((val - 0.55) * 2.4, 0, 1) * np.clip((0.50 - sat) * 3.0, 0, 1)
    return cv2.GaussianBlur(0.20 + 1.4 * white_lint, (0, 0), sigmaX=1.1)


def build_cloud(rows: list[dict[str, str]], width: int, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    pts, cols, anchors, phases = [], [], [], []
    selected = []
    pre = [r for r in rows if r["phase"] == "pre"][:3]
    post = [r for r in rows if r["phase"] == "post"][:3]
    for i in range(max(len(pre), len(post))):
        if i < len(post):
            selected.append(post[i])
        if i < len(pre):
            selected.append(pre[i])
    selected = selected[:6]
    for idx, row in enumerate(selected):
        rgb = load_rgb(row, width)
        h, w = rgb.shape[:2]
        height = cotton_height(rgb)
        yy, xx = np.mgrid[0:h:stride, 0:w:stride]
        x = (xx - w / 2.0) * 0.035
        z = idx * 8.0 + (yy - h / 2.0) * 0.035
        y = height[yy, xx] * (1.15 if row["phase"] == "pre" else 1.0)
        pts.append(np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[yy, xx].reshape(-1)
        cols.append(np.stack([gray, gray, gray], axis=1).astype(np.uint8))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = ((hsv[..., 1] < 80) & (hsv[..., 2] > 180)).astype(np.uint8) * 255
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        comps = []
        for comp in range(1, n):
            area = stats[comp, cv2.CC_STAT_AREA]
            if 4 <= area <= 700:
                comps.append((area, centroids[comp]))
        for _, (cx, cy) in sorted(comps, key=lambda item: item[0], reverse=True)[:35]:
            anchors.append([(cx - w / 2.0) * 0.035, float(height[int(cy), int(cx)]) + 0.35, idx * 8.0 + (cy - h / 2.0) * 0.035])
            phases.append(row["phase"])
    return np.concatenate(pts), np.concatenate(cols), np.array(anchors, dtype=np.float32), phases


def look_at(camera: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - camera
    forward /= np.linalg.norm(forward) + 1e-8
    right = np.cross(forward, np.array([0, 1, 0], dtype=np.float32))
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, forward)
    return np.stack([right, up, forward], axis=0)


def project(points: np.ndarray, camera: np.ndarray, target: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = look_at(camera, target)
    cam = (points - camera) @ rot.T
    z = cam[:, 2]
    valid = z > 0.2
    focal = 0.5 * w / math.tan(math.radians(56) / 2)
    x = cam[:, 0] * focal / z + w / 2
    y = h / 2 - cam[:, 1] * focal / z
    valid &= (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return x[valid].astype(np.int32), y[valid].astype(np.int32), valid


def render_cloud_frame(points: np.ndarray, colors: np.ndarray, anchors: np.ndarray, phases: list[str], t: float, w: int, h: int) -> np.ndarray:
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (6, 6, 8)
    zmin, zmax = float(points[:, 2].min()), float(points[:, 2].max())
    target = np.array([0, 0.7, zmin + t * (zmax - zmin)], dtype=np.float32)
    camera = np.array([math.sin(t * math.pi * 2) * 8.0, 5.5, target[2] - 13], dtype=np.float32)
    px, py, valid = project(points, camera, target, w, h)
    if len(px):
        depth = np.linalg.norm(points[valid] - camera, axis=1)
        order = np.argsort(depth)[::-1]
        canvas[py[order], px[order]] = colors[valid][order]
        canvas[:] = cv2.dilate(canvas, np.ones((2, 2), np.uint8), iterations=1)
    if len(anchors):
        ax, ay, av = project(anchors, camera, target, w, h)
        shown_phases = [p for p, keep in zip(phases, av) if keep]
        for x, y, phase in zip(ax, ay, shown_phases):
            color = (50, 255, 50) if phase == "post" else (245, 245, 245)
            cv2.circle(canvas, (int(x), int(y)), 6, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(x), int(y)), 11, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def compose_intro(pre_img, post_img, pre_cands, post_cands, pre_stats, post_stats, frame_i: int, total: int) -> np.ndarray:
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    canvas[:] = (8, 8, 10)
    left = fit(mono(pre_img), 860, 650)
    right = fit(mono(post_img), 860, 650)
    draw_detections(left, pre_cands, pre_img.shape[:2], (240, 240, 240))
    draw_detections(right, post_cands, post_img.shape[:2], (40, 255, 90))
    canvas[140:790, 70:930] = left
    canvas[140:790, 990:1850] = right
    draw_text(canvas, "PRE-DEFOLIATION", (90, 115), 0.9)
    draw_text(canvas, "POST-DEFOLIATION", (1010, 115), 0.9, (80, 255, 120))
    draw_text(canvas, "Detection-guided 3D cotton boll phenotyping", (70, 55), 1.15)
    y0 = 850
    draw_text(canvas, f"pre raw candidates shown: {pre_stats.shown_count}/{pre_stats.raw_count}", (90, y0), 0.7)
    draw_text(canvas, f"post raw candidates shown: {post_stats.shown_count}/{post_stats.raw_count}", (1010, y0), 0.7, (80, 255, 120))
    draw_text(canvas, "diameter/volume below are proxy until scale calibration", (520, 1005), 0.65, (220, 220, 220))
    progress = int(1780 * frame_i / max(1, total - 1))
    cv2.rectangle(canvas, (70, 1030), (70 + progress, 1042), (80, 255, 120), -1)
    return canvas


def compose_measurement(cloud: np.ndarray, pre_stats: DetectionStats, post_stats: DetectionStats, frame_i: int, total: int) -> np.ndarray:
    canvas = cv2.resize(cloud, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (1920, 170), (0, 0, 0), -1)
    cv2.rectangle(overlay, (1120, 210), (1850, 770), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0)
    draw_text(canvas, "3D reconstruction scaffold: monochrome point cloud + semantic boll anchors", (45, 55), 0.95)
    draw_text(canvas, "Next scientific step: replace 2.5D scaffold with COLMAP/VGGT/MASt3R/3DGS metric geometry", (45, 105), 0.64, (220, 220, 220))
    draw_text(canvas, "PROXY MEASUREMENT PANEL", (1160, 270), 0.78, (80, 255, 120))
    draw_text(canvas, f"Pre median diameter:  {pre_stats.median_diameter_px:.1f} px", (1160, 340), 0.66)
    draw_text(canvas, f"Post median diameter: {post_stats.median_diameter_px:.1f} px", (1160, 390), 0.66, (80, 255, 120))
    draw_text(canvas, f"Pre median volume:    {pre_stats.median_volume_px3/1000:.1f}k px^3", (1160, 465), 0.66)
    draw_text(canvas, f"Post median volume:   {post_stats.median_volume_px3/1000:.1f}k px^3", (1160, 515), 0.66, (80, 255, 120))
    draw_text(canvas, "cm/mm output requires GSD or field scale", (1160, 610), 0.62, (220, 220, 220))
    draw_text(canvas, "USP: pre/post visibility + semantic 3D morphology", (1160, 665), 0.62, (220, 220, 220))
    progress = int(1780 * frame_i / max(1, total - 1))
    cv2.rectangle(canvas, (70, 1030), (70 + progress, 1042), (80, 255, 120), -1)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-video", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    pre_row, post_row = choose_pair(rows)
    pre_img = load_rgb(pre_row, 1200)
    post_img = load_rgb(post_row, 1200)
    pre_cands, pre_stats = measure_candidates(pre_img, "pre", max_show=120)
    post_cands, post_stats = measure_candidates(post_img, "post", max_show=120)
    points, colors, anchors, phases = build_cloud(rows, width=520, stride=4)

    args.frames_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(45):
        frames.append(compose_intro(pre_img, post_img, pre_cands, post_cands, pre_stats, post_stats, i, 45))
    for i in range(120):
        t = i / 119
        cloud = render_cloud_frame(points, colors, anchors, phases, t, 1280, 720)
        frames.append(compose_measurement(cloud, pre_stats, post_stats, i, 120))

    for idx, frame in enumerate(frames):
        cv2.imwrite(str(args.frames_dir / f"frame_{idx:05d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    print(f"frames: {len(frames)}")
    print(f"pre: {pre_stats}")
    print(f"post: {post_stats}")
    print(f"encode: ffmpeg -y -framerate {args.fps} -i {args.frames_dir}/frame_%05d.jpg -c:v libx264 -pix_fmt yuv420p -movflags +faststart {args.out_video}")


if __name__ == "__main__":
    main()
