#!/usr/bin/env python3
"""
Prepare a small ordered image set for VPS / 3D reconstruction experiments.

This is the first practical bridge from cotton imagery to a path-based 3D
viewer. It selects sharp frames per phase/folder, writes a reconstruction image
manifest, and creates a simple camera-path scaffold that can later be replaced
by COLMAP/VGGT/MASt3R poses.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2

from run_dataset_counter import list_images, phase_from_path


def sharpness_score(image_path: Path) -> float:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None:
        return -1.0
    return float(cv2.Laplacian(bgr, cv2.CV_64F).var())


def group_key(path: Path) -> tuple[str, str]:
    return (phase_from_path(path), path.parent.name)


def select_images(dataset_root: Path, per_group: int) -> list[Path]:
    grouped: dict[tuple[str, str], list[tuple[float, Path]]] = {}
    for image_path in list_images(dataset_root):
        try:
            key = group_key(image_path)
        except ValueError:
            continue
        grouped.setdefault(key, []).append((sharpness_score(image_path), image_path))

    selected: list[Path] = []
    for key in sorted(grouped):
        scored = sorted(grouped[key], key=lambda item: item[0], reverse=True)
        selected.extend(path for _, path in scored[:per_group])
    return sorted(selected)


def prepare(dataset_root: Path, out_dir: Path, per_group: int, copy_images: bool) -> None:
    selected = select_images(dataset_root, per_group=per_group)
    if not selected:
        raise RuntimeError(f"No reconstruction candidates found under {dataset_root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "images"
    if copy_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    image_manifest_path = out_dir / "reconstruction_images.csv"
    camera_path = out_dir / "camera_path_scaffold.csv"

    with image_manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_index", "source_image", "phase", "folder", "viewer_image"],
        )
        writer.writeheader()
        for idx, image_path in enumerate(selected):
            viewer_image = ""
            if copy_images:
                dst = image_dir / f"{idx:05d}_{phase_from_path(image_path)}_{image_path.name}"
                shutil.copy2(image_path, dst)
                viewer_image = str(dst)
            writer.writerow(
                {
                    "frame_index": idx,
                    "source_image": str(image_path),
                    "phase": phase_from_path(image_path),
                    "folder": image_path.parent.name,
                    "viewer_image": viewer_image,
                }
            )

    with camera_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "x", "y", "z", "qw", "qx", "qy", "qz"])
        writer.writeheader()
        for idx, image_path in enumerate(selected):
            writer.writerow(
                {
                    "frame": image_path.name,
                    "x": f"{idx:.3f}",
                    "y": "0.000",
                    "z": "0.000",
                    "qw": "1.000",
                    "qx": "0.000",
                    "qy": "0.000",
                    "qz": "0.000",
                }
            )

    print(f"selected images: {len(selected)}")
    print(f"image manifest: {image_manifest_path}")
    print(f"camera path scaffold: {camera_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reconstruction input images.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/reconstruction_inputs/icml_dataset"))
    parser.add_argument("--per-group", type=int, default=8)
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        per_group=args.per_group,
        copy_images=args.copy_images,
    )


if __name__ == "__main__":
    main()
