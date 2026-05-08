#!/usr/bin/env python3
"""Build local 2.5D cotton boll/cluster reconstructions from ranked crops.

The output is a monocular proxy: real UAV texture draped over an image-derived
lint/texture height field. It is intended to identify and visualize promising
targets for true multi-view reconstruction, not to claim calibrated boll volume.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    lo = float(np.percentile(values, 2))
    hi = float(np.percentile(values, 98))
    if hi <= lo + 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def read_candidates(path: Path, phase: str, limit: int) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["phase"] == phase]
    rows.sort(key=lambda row: float(row["measurement_ready_score"]), reverse=True)
    return rows[:limit]


def crop_candidate(row: dict[str, str], crop_size: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    image = Image.open(row["image"]).convert("RGB")
    x = int(row["x"])
    y = int(row["y"])
    width = int(row["width"])
    height = int(row["height"])
    cx = x + width // 2
    cy = y + height // 2
    half = max(crop_size // 2, int(2.4 * max(width, height)))
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(image.width, cx + half)
    y1 = min(image.height, cy + half)
    crop = image.crop((x0, y0, x1, y1))
    crop.thumbnail((crop_size, crop_size), Image.Resampling.LANCZOS)
    return crop, (x - x0, y - y0, width, height)


def estimate_local_height(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    value = np.max(rgb_f, axis=2)
    saturation = np.max(rgb_f, axis=2) - np.min(rgb_f, axis=2)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gy, gx = np.gradient(gray)
    texture = normalize(np.sqrt(gx * gx + gy * gy))
    exg = 2.0 * g - r - b
    lint = np.clip((value - 0.48) / 0.38, 0.0, 1.0) * np.clip((0.55 - saturation) / 0.55, 0.0, 1.0)
    lint *= np.clip((70.0 - exg * 255.0) / 120.0, 0.0, 1.0)
    lint_img = Image.fromarray((normalize(lint) * 255).astype(np.uint8))
    lint_smooth = np.asarray(lint_img.filter(ImageFilter.GaussianBlur(radius=2.2)), dtype=np.float32) / 255.0
    lint_broad = np.asarray(lint_img.filter(ImageFilter.GaussianBlur(radius=7.0)), dtype=np.float32) / 255.0
    texture_smooth = np.asarray(
        Image.fromarray((texture * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.4)),
        dtype=np.float32,
    ) / 255.0
    # Keep cotton-like lobes while suppressing needle-like spikes that make the
    # proxy look synthetic. This remains a monocular inspection surface.
    height = normalize(0.56 * lint_smooth + 0.24 * lint_broad + 0.12 * texture_smooth + 0.08 * value)
    height = np.asarray(
        Image.fromarray((height * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.0)),
        dtype=np.float32,
    ) / 255.0
    return height, normalize(lint)


def write_ply(path: Path, rgb: np.ndarray, height: np.ndarray, z_scale: float = 0.20) -> None:
    h, w = height.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = xx.astype(np.float32) / max(w - 1, 1) - 0.5
    y = 0.5 - yy.astype(np.float32) / max(h - 1, 1)
    z = height.astype(np.float32) * z_scale
    points = np.column_stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)])
    colors = rgb.reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def shaded_real_texture(rgb: np.ndarray, height: np.ndarray) -> np.ndarray:
    texture = rgb.astype(np.float32) / 255.0
    height_img = Image.fromarray((height * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.8))
    h = np.asarray(height_img, dtype=np.float32) / 255.0
    gy, gx = np.gradient(h)
    relief = normalize(-0.34 * gx - 0.55 * gy + 0.12 * h)
    cotton_weight = np.clip((h - 0.22) / 0.55, 0.0, 1.0)[..., None]
    shade = (0.95 + 0.16 * (relief - 0.5))[..., None]
    shaded = texture * (1.0 - cotton_weight) + np.clip(texture * shade + 0.02, 0.0, 1.0) * cotton_weight
    return np.clip(shaded * 1.02, 0.0, 1.0)


def plot_surface(ax: plt.Axes, rgb: np.ndarray, height: np.ndarray, title: str, azim: float) -> None:
    h, w = height.shape
    yy, xx = np.mgrid[0:h:h * 1j, 0:w:w * 1j]
    x = xx / max(w - 1, 1)
    y = 1.0 - yy / max(h - 1, 1)
    z = np.asarray(
        Image.fromarray((height * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2.2)),
        dtype=np.float32,
    ) / 255.0
    z *= 0.135
    texture = shaded_real_texture(rgb, height)
    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        facecolors=texture,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    ax.view_init(elev=39, azim=azim)
    ax.set_box_aspect((1, 1, 0.17))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 0.16)
    ax.set_axis_off()
    ax.set_title(title, fontsize=8, fontweight="bold", pad=0)


def build_single_view(rgb: np.ndarray, height: np.ndarray, output_path: Path, title: str, azim: float = -58.0) -> None:
    fig = plt.figure(figsize=(8.0, 6.4), dpi=180)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    plot_surface(ax, rgb, height, title, azim)
    fig.text(
        0.04,
        0.035,
        "Real UAV crop texture draped over a monocular 2.5D relief; not calibrated metric geometry.",
        fontsize=7,
        color="#65715f",
    )
    fig.tight_layout(pad=0.1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def build_rotation_video(
    rgb: np.ndarray,
    height: np.ndarray,
    output_path: Path,
    title: str,
    frames: int = 90,
    fps: int = 18,
) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    frame_dir = output_path.parent / "_rotation_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("frame_*.png"):
        old.unlink()
    for idx in range(frames):
        azim = -70.0 + 360.0 * idx / frames
        frame_path = frame_dir / f"frame_{idx:04d}.png"
        build_single_view(rgb, height, frame_path, title, azim=azim)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def build_gallery(results: list[dict[str, object]], output_path: Path) -> None:
    cols = 4
    rows = math.ceil(len(results) / cols)
    fig = plt.figure(figsize=(cols * 3.35, rows * 3.05), dpi=180)
    fig.patch.set_facecolor("white")
    for idx, result in enumerate(results, start=1):
        ax = fig.add_subplot(rows, cols, idx, projection="3d")
        plot_surface(
            ax,
            result["rgb"],  # type: ignore[arg-type]
            result["height"],  # type: ignore[arg-type]
            f"Rank {result['rank']} | score {result['score']}",
            -56.0,
        )
    fig.suptitle(
        "Local Cotton Boll/Cluster 2.5D Reconstructions From Real UAV Crops\n"
        "Real texture on soft monocular relief for target selection; not calibrated metric 3D.",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def write_summary(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = args.out_dir / "ply"
    png_dir = args.out_dir / "crops"
    rows = read_candidates(args.candidates_csv, args.phase, args.limit)
    results: list[dict[str, object]] = []
    summary_rows: list[dict[str, str | int | float]] = []
    best_result: dict[str, object] | None = None
    for idx, row in enumerate(rows, start=1):
        crop, _ = crop_candidate(row, args.crop_size)
        rgb = np.asarray(crop, dtype=np.uint8)
        height, lint = estimate_local_height(rgb)
        stem = f"{idx:03d}_{row['phase']}_{Path(row['image']).stem}_cand{row['candidate_id']}"
        crop_path = png_dir / f"{stem}.jpg"
        ply_path = ply_dir / f"{stem}.ply"
        png_dir.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path, quality=92)
        write_ply(ply_path, rgb, height)
        results.append(
            {
                "rank": row["rank"],
                "score": row["measurement_ready_score"],
                "rgb": rgb,
                "height": height,
            }
        )
        if best_result is None:
            best_result = {"rank": row["rank"], "score": row["measurement_ready_score"], "rgb": rgb, "height": height}
        summary_rows.append(
            {
                "local_rank": idx,
                "global_rank": int(row["rank"]),
                "phase": row["phase"],
                "source_image": row["image"],
                "candidate_id": int(row["candidate_id"]),
                "measurement_ready_score": float(row["measurement_ready_score"]),
                "crop_path": str(crop_path),
                "ply_path": str(ply_path),
                "height_mean": round(float(np.mean(height)), 5),
                "height_std": round(float(np.std(height)), 5),
                "lint_mean": round(float(np.mean(lint)), 5),
            }
        )
    gallery_path = args.out_dir / "local_boll_2p5d_gallery.png"
    build_gallery(results, gallery_path)
    best_view_path = args.out_dir / "best_local_boll_2p5d_view.png"
    video_path = args.out_dir / "best_local_boll_2p5d_rotation.mp4"
    video_created = False
    if best_result is not None:
        title = f"Best local cotton cluster | rank {best_result['rank']} | score {best_result['score']}"
        build_single_view(best_result["rgb"], best_result["height"], best_view_path, title)  # type: ignore[arg-type]
        if args.video:
            video_created = build_rotation_video(
                best_result["rgb"],  # type: ignore[arg-type]
                best_result["height"],  # type: ignore[arg-type]
                video_path,
                title,
                frames=args.video_frames,
                fps=args.video_fps,
            )
    write_summary(args.out_dir / "local_boll_2p5d_summary.csv", summary_rows)
    manifest = {
        "artifact_type": "local cotton boll 2.5D reconstruction proxy",
        "scientific_boundary": "Single-image monocular proxy; not calibrated boll 3D or volume.",
        "phase": args.phase,
        "limit": args.limit,
        "candidates_csv": str(args.candidates_csv),
        "gallery": str(gallery_path),
        "best_view": str(best_view_path),
        "rotation_video": str(video_path) if video_created else "",
        "summary_csv": str(args.out_dir / "local_boll_2p5d_summary.csv"),
        "ply_dir": str(ply_dir),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates-csv",
        type=Path,
        default=Path("outputs/metrics/measurement_ready_bolls/measurement_ready_candidates.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/metrics/local_boll_2p5d_reconstruction"))
    parser.add_argument("--phase", choices=["pre", "post"], default="post")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--crop-size", type=int, default=300)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-frames", type=int, default=72)
    parser.add_argument("--video-fps", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
