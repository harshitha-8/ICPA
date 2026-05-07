#!/usr/bin/env python3
"""Mine measurement-ready cotton boll candidates from pre/post UAV images.

This experiment is intentionally lightweight and dependency-minimal. It uses a
deterministic bright-low-saturation lint detector on downsampled UAV frames,
ranks candidates by measurement readiness, and exports crops that can seed local
3D reconstruction attempts.

The output is not ground truth. It is a ranked candidate mining pass for the
next stage: local multi-view reconstruction and mask validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


DATASET_ROOT = Path("/Volumes/T9/ICML")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
PHASE_DIRS = {
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
class Candidate:
    phase: str
    image_path: Path
    candidate_id: int
    x: int
    y: int
    width: int
    height: int
    area_px: int
    lint_fraction: float
    green_fraction: float
    brightness: float
    visibility: float
    major_axis_px: float
    minor_axis_px: float
    readiness: float

    @property
    def center_x(self) -> float:
        return self.x + 0.5 * self.width

    @property
    def center_y(self) -> float:
        return self.y + 0.5 * self.height


def list_images(phase: str, limit: int | None = None) -> list[Path]:
    images: list[Path] = []
    for folder in PHASE_DIRS[phase]:
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS and not path.name.startswith("._"):
                images.append(path)
                if limit is not None and len(images) >= limit:
                    return images
    return images


def load_resized_rgb(path: Path, long_edge: int) -> tuple[Image.Image, np.ndarray, float]:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = min(1.0, long_edge / max(width, height))
    if scale < 1.0:
        image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.uint8)
    return image, arr, scale


def rgb_to_hsv_like(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb_f = rgb.astype(np.float32) / 255.0
    mx = np.max(rgb_f, axis=2)
    mn = np.min(rgb_f, axis=2)
    saturation = np.zeros_like(mx, dtype=np.float32)
    np.divide(mx - mn, mx, out=saturation, where=mx > 1e-6)
    value = mx
    return saturation, value


def lint_mask(rgb: np.ndarray, phase: str) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    saturation, value = rgb_to_hsv_like(rgb)
    exg = 2.0 * g - r - b

    bright_lint = (value > 0.58) & (saturation < 0.36)
    very_bright_lint = (value > 0.72) & (saturation < 0.52)
    not_green = exg < (46.0 if phase == "pre" else 62.0)
    mask = (bright_lint | very_bright_lint) & not_green

    # Suppress isolated one-pixel speckles using a small neighborhood count.
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    count = np.zeros(mask.shape, dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            count += padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return mask & (count >= 2)


def connected_components(mask: np.ndarray, min_area: int, max_components: int) -> list[tuple[int, int, int, int, int]]:
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    comps: list[tuple[int, int, int, int, int]] = []
    ys, xs = np.where(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        area = 0
        x0 = x1 = start_x
        y0 = y1 = start_y
        while queue:
            y, x = queue.popleft()
            area += 1
            x0 = min(x0, x)
            x1 = max(x1, x)
            y0 = min(y0, y)
            y1 = max(y1, y)
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if ny == y and nx == x:
                        continue
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        if area >= min_area:
            comps.append((x0, y0, x1 + 1, y1 + 1, area))
    comps.sort(key=lambda item: item[4], reverse=True)
    return comps[:max_components]


def score_candidate(rgb: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int, int], phase: str) -> tuple[float, dict[str, float]]:
    x0, y0, x1, y1, area = box
    crop = rgb[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0, {}

    rgb_f = crop.astype(np.float32)
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    saturation, value = rgb_to_hsv_like(crop)
    exg = 2.0 * g - r - b
    green = (exg > 18.0) & (g > r) & (g > b)
    bright = value > 0.58
    lint = crop_mask.astype(bool)

    width = max(x1 - x0, 1)
    height = max(y1 - y0, 1)
    box_area = width * height
    visibility = min(area / max(box_area, 1), 1.0)
    aspect = max(width, height) / max(min(width, height), 1)
    aspect_score = max(0.0, 1.0 - max(0.0, aspect - 1.0) / 3.5)
    size_score = min(math.sqrt(area) / 20.0, 1.0)
    lint_fraction = float(np.mean(lint))
    green_fraction = float(np.mean(green))
    brightness = float(np.mean(bright))
    phase_bonus = 0.08 if phase == "post" else 0.0

    readiness = (
        0.32 * lint_fraction
        + 0.20 * visibility
        + 0.16 * brightness
        + 0.14 * aspect_score
        + 0.12 * size_score
        + 0.06 * (1.0 - green_fraction)
        + phase_bonus
    )
    if aspect > 4.2:
        readiness *= 0.50
    if green_fraction > 0.55 and lint_fraction < 0.20:
        readiness *= 0.45
    metrics = {
        "lint_fraction": lint_fraction,
        "green_fraction": green_fraction,
        "brightness": brightness,
        "visibility": visibility,
        "major_axis_px": float(max(width, height)),
        "minor_axis_px": float(min(width, height)),
    }
    return float(max(0.0, min(readiness, 1.0))), metrics


def detect_measurement_candidates(image_path: Path, phase: str, long_edge: int, max_candidates_per_image: int) -> tuple[Image.Image, list[Candidate]]:
    image, rgb, scale = load_resized_rgb(image_path, long_edge)
    mask = lint_mask(rgb, phase)
    min_area = max(5, int((long_edge / 900.0) ** 2 * 5))
    comps = connected_components(mask, min_area=min_area, max_components=max_candidates_per_image * 4)
    candidates: list[Candidate] = []
    inv_scale = 1.0 / max(scale, 1e-8)
    for comp_idx, comp in enumerate(comps, start=1):
        x0, y0, x1, y1, area = comp
        readiness, metrics = score_candidate(rgb, mask, comp, phase)
        if readiness < 0.16:
            continue
        candidates.append(
            Candidate(
                phase=phase,
                image_path=image_path,
                candidate_id=comp_idx,
                x=int(round(x0 * inv_scale)),
                y=int(round(y0 * inv_scale)),
                width=max(1, int(round((x1 - x0) * inv_scale))),
                height=max(1, int(round((y1 - y0) * inv_scale))),
                area_px=int(round(area * inv_scale * inv_scale)),
                lint_fraction=metrics["lint_fraction"],
                green_fraction=metrics["green_fraction"],
                brightness=metrics["brightness"],
                visibility=metrics["visibility"],
                major_axis_px=metrics["major_axis_px"] * inv_scale,
                minor_axis_px=metrics["minor_axis_px"] * inv_scale,
                readiness=readiness,
            )
        )
    candidates.sort(key=lambda cand: cand.readiness, reverse=True)
    return image, candidates[:max_candidates_per_image]


def candidate_row(cand: Candidate, rank: int) -> dict[str, str | int | float]:
    return {
        "rank": rank,
        "phase": cand.phase,
        "image": str(cand.image_path),
        "folder": cand.image_path.parent.name,
        "filename": cand.image_path.name,
        "candidate_id": cand.candidate_id,
        "x": cand.x,
        "y": cand.y,
        "width": cand.width,
        "height": cand.height,
        "center_x": round(cand.center_x, 3),
        "center_y": round(cand.center_y, 3),
        "area_px": cand.area_px,
        "mask_major_axis_px": round(cand.major_axis_px, 3),
        "mask_minor_axis_px": round(cand.minor_axis_px, 3),
        "lint_fraction": round(cand.lint_fraction, 4),
        "green_fraction": round(cand.green_fraction, 4),
        "brightness": round(cand.brightness, 4),
        "visibility": round(cand.visibility, 4),
        "measurement_ready_score": round(cand.readiness, 4),
    }


def save_crop(source_path: Path, cand: Candidate, crop_path: Path) -> None:
    image = Image.open(source_path).convert("RGB")
    pad = max(40, int(1.5 * max(cand.width, cand.height)))
    cx = int(cand.center_x)
    cy = int(cand.center_y)
    x0 = max(0, cx - pad)
    y0 = max(0, cy - pad)
    x1 = min(image.width, cx + pad)
    y1 = min(image.height, cy + pad)
    crop = image.crop((x0, y0, x1, y1))
    draw = ImageDraw.Draw(crop)
    draw.rectangle(
        (cand.x - x0, cand.y - y0, cand.x + cand.width - x0, cand.y + cand.height - y0),
        outline=(255, 220, 40),
        width=max(2, crop.width // 150),
    )
    crop.thumbnail((420, 420), Image.Resampling.LANCZOS)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(crop_path, quality=92)


def save_contact_sheet(crop_paths: list[Path], output_path: Path, cols: int = 8, tile: int = 180) -> None:
    if not crop_paths:
        return
    rows = math.ceil(len(crop_paths) / cols)
    sheet = Image.new("RGB", (cols * tile, rows * tile), (255, 255, 255))
    for idx, path in enumerate(crop_paths):
        crop = Image.open(path).convert("RGB")
        crop = ImageOps.pad(crop, (tile, tile), method=Image.Resampling.LANCZOS, color=(255, 255, 255))
        sheet.paste(crop, ((idx % cols) * tile, (idx // cols) * tile))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    out_dir: Path = args.out_dir
    crop_dir = out_dir / "top_crops"
    subset_dir = out_dir / "reconstruction_seed_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_candidates: list[Candidate] = []
    image_rows: list[dict[str, str | int | float]] = []
    for phase in ("pre", "post"):
        images = list_images(phase, limit=args.max_images_per_phase)
        for idx, image_path in enumerate(images, start=1):
            _, candidates = detect_measurement_candidates(
                image_path,
                phase=phase,
                long_edge=args.long_edge,
                max_candidates_per_image=args.max_candidates_per_image,
            )
            all_candidates.extend(candidates)
            image_rows.append(
                {
                    "phase": phase,
                    "image": str(image_path),
                    "folder": image_path.parent.name,
                    "filename": image_path.name,
                    "num_candidates": len(candidates),
                    "mean_readiness": round(float(np.mean([c.readiness for c in candidates])) if candidates else 0.0, 4),
                    "max_readiness": round(float(max([c.readiness for c in candidates])) if candidates else 0.0, 4),
                }
            )
            if idx == 1 or idx % 25 == 0 or idx == len(images):
                print(f"[{phase} {idx}/{len(images)}] {image_path.name}: {len(candidates)} candidates")

    all_candidates.sort(key=lambda cand: cand.readiness, reverse=True)
    candidate_rows = [candidate_row(cand, rank) for rank, cand in enumerate(all_candidates, start=1)]
    write_csv(out_dir / "measurement_ready_candidates.csv", candidate_rows)
    write_csv(out_dir / "image_candidate_summary.csv", image_rows)

    crop_paths: list[Path] = []
    for rank, cand in enumerate(all_candidates[: args.top_crops], start=1):
        crop_path = crop_dir / f"{rank:04d}_{cand.phase}_{cand.image_path.stem}_cand{cand.candidate_id:03d}.jpg"
        save_crop(cand.image_path, cand, crop_path)
        crop_paths.append(crop_path)
    save_contact_sheet(crop_paths[: min(len(crop_paths), 64)], out_dir / "top_crops_contact_sheet.jpg")

    seed_images: list[Path] = []
    for cand in all_candidates:
        if cand.phase != "post":
            continue
        if cand.image_path not in seed_images:
            seed_images.append(cand.image_path)
        if len(seed_images) >= args.reconstruction_seed_images:
            break
    subset_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    for idx, image_path in enumerate(seed_images, start=1):
        target = subset_dir / f"{idx:04d}_{image_path.name}"
        if not target.exists():
            shutil.copy2(image_path, target)
        seed_rows.append({"index": idx, "source": str(image_path), "copied_path": str(target)})
    write_csv(out_dir / "reconstruction_seed_images.csv", seed_rows)

    phase_summary = {}
    for phase in ("pre", "post"):
        phase_candidates = [cand for cand in all_candidates if cand.phase == phase]
        phase_summary[phase] = {
            "candidates": len(phase_candidates),
            "images": len([row for row in image_rows if row["phase"] == phase]),
            "mean_readiness": float(np.mean([c.readiness for c in phase_candidates])) if phase_candidates else 0.0,
            "high_confidence_candidates_score_ge_0_45": sum(c.readiness >= 0.45 for c in phase_candidates),
            "high_confidence_candidates_score_ge_0_55": sum(c.readiness >= 0.55 for c in phase_candidates),
        }
    manifest = {
        "artifact_type": "measurement-ready cotton boll candidate mining",
        "scientific_boundary": "Candidate mining for reconstruction, not ground-truth boll labels.",
        "dataset_root": str(DATASET_ROOT),
        "long_edge": args.long_edge,
        "max_images_per_phase": args.max_images_per_phase,
        "max_candidates_per_image": args.max_candidates_per_image,
        "outputs": {
            "candidates_csv": str(out_dir / "measurement_ready_candidates.csv"),
            "image_summary_csv": str(out_dir / "image_candidate_summary.csv"),
            "top_crops_dir": str(crop_dir),
            "contact_sheet": str(out_dir / "top_crops_contact_sheet.jpg"),
            "reconstruction_seed_images": str(subset_dir),
        },
        "phase_summary": phase_summary,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["phase_summary"], indent=2))
    print(f"outputs: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/metrics/measurement_ready_bolls"))
    parser.add_argument("--max-images-per-phase", type=int, default=120)
    parser.add_argument("--long-edge", type=int, default=900)
    parser.add_argument("--max-candidates-per-image", type=int, default=180)
    parser.add_argument("--top-crops", type=int, default=120)
    parser.add_argument("--reconstruction-seed-images", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
