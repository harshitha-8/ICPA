#!/usr/bin/env python3
"""
Build a lightweight MVP video for the cotton 3D viewer story.

The video is intentionally honest: it visualizes selected reconstruction frames,
phase labels, folder provenance, and a scaffolded camera path. It does not claim
that metric 3D reconstruction has already been solved.
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


def fit_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def draw_label(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.8) -> None:
    cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def draw_path_panel(canvas: np.ndarray, idx: int, total: int, phase: str) -> None:
    x0, y0, w, h = 1280, 130, 520, 760
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (22, 27, 35), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (95, 140, 180), 2)
    draw_label(canvas, "Camera path scaffold", x0 + 35, y0 + 55, 0.8)

    pad = 70
    xs = np.linspace(x0 + pad, x0 + w - pad, total)
    ys = y0 + h // 2 + 120 * np.sin(np.linspace(0, 2.4 * np.pi, total))
    pts = np.column_stack([xs, ys]).astype(np.int32)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(canvas, tuple(a), tuple(b), (70, 150, 255), 3, cv2.LINE_AA)

    for p_i, p in enumerate(pts):
        color = (80, 180, 255) if p_i <= idx else (70, 75, 85)
        cv2.circle(canvas, tuple(p), 6, color, -1, cv2.LINE_AA)

    current_color = (0, 220, 120) if phase == "post" else (255, 185, 70)
    cv2.circle(canvas, tuple(pts[idx]), 15, current_color, -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(pts[idx]), 22, (255, 255, 255), 2, cv2.LINE_AA)

    draw_label(canvas, "Scene anchors: pending real 3D boll triangulation", x0 + 35, y0 + h - 95, 0.62)
    draw_label(canvas, "Geometry layer: point cloud / Gaussian splat target", x0 + 35, y0 + h - 55, 0.62)


def make_slide(row: dict[str, str], idx: int, total: int, width: int, height: int) -> np.ndarray:
    candidate = Path(row["viewer_image"]) if row.get("viewer_image") else None
    image_path = candidate if candidate is not None and candidate.exists() else Path(row["source_image"])
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"Could not read {image_path}")

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (14, 18, 25)
    image_panel = fit_image(bgr, 1180, 760)
    canvas[70:830, 70:1250] = image_panel
    cv2.rectangle(canvas, (70, 70), (1250, 830), (120, 170, 210), 2)

    phase = row["phase"]
    phase_color = (0, 190, 120) if phase == "post" else (255, 170, 50)
    cv2.rectangle(canvas, (70, 850), (1250, 970), (25, 31, 42), -1)
    cv2.rectangle(canvas, (95, 878), (210, 928), phase_color, -1)
    draw_label(canvas, phase.upper(), 115, 912, 0.75)
    draw_label(canvas, f"Frame {idx + 1}/{total}: {Path(row['source_image']).name}", 245, 905, 0.78)
    draw_label(canvas, f"Folder: {row['folder']}", 245, 945, 0.62)

    draw_path_panel(canvas, idx, total, phase)
    draw_label(canvas, "Detection-guided 3D cotton boll phenotyping MVP", 70, 45, 0.92)
    return canvas


def build_video(manifest: Path, out_video: Path, frames_dir: Path, fps: int, hold: int) -> None:
    rows = read_manifest(manifest)
    if not rows:
        raise RuntimeError(f"Manifest has no rows: {manifest}")

    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    for idx, row in enumerate(rows):
        slide = make_slide(row, idx, len(rows), width=1920, height=1080)
        for _ in range(hold):
            frame_path = frames_dir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(frame_path), slide, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            frame_idx += 1

    out_video.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / "frame_%05d.jpg")
    print(f"wrote frames under {frames_dir}")
    print(f"encode with ffmpeg pattern: {pattern}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MVP video frames.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-video", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--hold", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_video(args.manifest, args.out_video, args.frames_dir, args.fps, args.hold)


if __name__ == "__main__":
    main()
