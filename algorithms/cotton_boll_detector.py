#!/usr/bin/env python3
"""
Cotton boll detection and counting prior.

This module preserves the lightweight detector from the prior accepted work and
turns it into a reusable weak-supervision component for 3D phenotyping. It does
not claim to produce final 3D morphology; it produces candidate 2D boll regions
that later modules can refine, associate across views, and triangulate.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

VALID_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class PhaseParams:
    sat_max: float
    value_min: float
    orig_luma_min: float
    max_aspect: float
    min_area: float
    pre_multiplier: float


@dataclass(frozen=True)
class BollCandidate:
    x: int
    y: int
    width: int
    height: int
    area: float
    mean_saturation: float
    mean_value: float
    phase: str

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + 0.5 * self.width, self.y + 0.5 * self.height)


PHASE_PRE = PhaseParams(
    sat_max=120.0,
    value_min=15.0,
    orig_luma_min=0.0,
    max_aspect=3.0,
    min_area=0.0,
    pre_multiplier=1.6,
)

PHASE_POST = PhaseParams(
    sat_max=120.0,
    value_min=15.0,
    orig_luma_min=0.0,
    max_aspect=3.0,
    min_area=0.0,
    pre_multiplier=1.0,
)


def list_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_EXT and not path.name.startswith("._"):
            yield path


def infer_phase_from_greenness(img_rgb: np.ndarray) -> str:
    img = img_rgb.astype(np.float32) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    exg = 2.0 * g - r - b
    green_frac = float(np.mean(exg > 0))
    return "pre" if green_frac >= 0.67 else "post"


def get_phase_params(phase: str) -> PhaseParams:
    if phase == "pre":
        return PHASE_PRE
    if phase == "post":
        return PHASE_POST
    raise ValueError("phase must be 'pre' or 'post'")


def detect_candidates(img_rgb: np.ndarray, phase: str) -> List[BollCandidate]:
    """Return 2D cotton boll candidate boxes in original image coordinates."""
    params = get_phase_params(phase)
    h, w = img_rgb.shape[:2]

    detect_maxdim = 640
    scale = detect_maxdim / max(h, w)
    if scale < 1.0:
        dw, dh = int(w * scale), int(h * scale)
        small = cv2.resize(img_rgb, (dw, dh), interpolation=cv2.INTER_AREA)
    else:
        dw, dh, scale = w, h, 1.0
        small = img_rgb.copy()

    orig_gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    eq = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    gray = cv2.cvtColor(eq, cv2.COLOR_RGB2GRAY)

    d_small = max(4, int(max(dw, dh) * 0.006))
    d_large = max(9, int(max(dw, dh) * 0.030))
    se_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_small, d_small))
    se_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_large, d_large))
    th_small = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se_small)
    th_large = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se_large)
    th = cv2.max(th_small, th_large)

    _, mask = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hsv_small = cv2.cvtColor(eq, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv_small[:, :, 1]
    val = hsv_small[:, :, 2]

    inv_scale = 1.0 / scale
    candidates: List[BollCandidate] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < params.min_area:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = max(cw, ch) / (min(cw, ch) + 1e-6)
        if aspect > params.max_aspect:
            continue

        roi = np.zeros((dh, dw), dtype=np.uint8)
        cv2.drawContours(roi, [cnt], -1, 255, -1)
        pix = roi == 255
        if np.count_nonzero(pix) == 0:
            continue

        mean_s = float(np.mean(sat[pix]))
        mean_v = float(np.mean(val[pix]))
        mean_orig = float(np.mean(orig_gray[pix]))

        if mean_s > params.sat_max or mean_v < params.value_min or mean_orig < params.orig_luma_min:
            continue

        candidates.append(
            BollCandidate(
                x=int(x * inv_scale),
                y=int(y * inv_scale),
                width=max(1, int(cw * inv_scale)),
                height=max(1, int(ch * inv_scale)),
                area=area * inv_scale * inv_scale,
                mean_saturation=mean_s,
                mean_value=mean_v,
                phase=phase,
            )
        )

    return candidates


def detect_cotton_bolls(img_rgb: np.ndarray, phase: str) -> Tuple[np.ndarray, int, List[BollCandidate]]:
    """Return annotated image, adjusted count, and raw candidates."""
    candidates = detect_candidates(img_rgb, phase)
    count = len(candidates)
    if phase == "pre":
        count = int(round(count * PHASE_PRE.pre_multiplier))

    annotated = img_rgb.copy()
    h, w = img_rgb.shape[:2]
    thick = max(2, int(min(h, w) * 0.001))
    font_scale = max(0.4, min(h, w) * 0.00018)

    for idx, cand in enumerate(candidates, start=1):
        cv2.rectangle(
            annotated,
            (cand.x, cand.y),
            (cand.x + cand.width, cand.y + cand.height),
            (0, 180, 80),
            thick,
        )
        cv2.putText(
            annotated,
            str(idx),
            (cand.x + 2, max(cand.y - 4, thick + 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, thick - 1),
            cv2.LINE_AA,
        )

    return annotated, count, candidates


def run_single(image_path: Path, phase: str, save_annotated: Optional[Path]) -> int:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resolved_phase = infer_phase_from_greenness(rgb) if phase == "auto" else phase
    annotated, count, _ = detect_cotton_bolls(rgb, resolved_phase)

    if save_annotated is not None:
        save_annotated.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_annotated), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    print(f"{image_path} -> count={count}, phase={resolved_phase}")
    return count


def run_folder(input_dir: Path, out_dir: Path, phase: str) -> None:
    images = list(list_images(input_dir))
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "counts.csv"

    rows = []
    for idx, img_path in enumerate(images, start=1):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resolved_phase = infer_phase_from_greenness(rgb) if phase == "auto" else phase
        annotated, count, candidates = detect_cotton_bolls(rgb, resolved_phase)

        ann_path = out_dir / f"{img_path.stem}_annotated.jpg"
        cv2.imwrite(str(ann_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        rows.append(
            {
                "image": str(img_path),
                "count": count,
                "num_candidates": len(candidates),
                "phase_used": resolved_phase,
                "annotated_path": str(ann_path),
            }
        )
        print(f"[{idx}/{len(images)}] {img_path.name}: {count} ({resolved_phase})")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "count", "num_candidates", "phase_used", "annotated_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"done. csv: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cotton boll counting prior.")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--phase", default="auto", choices=["auto", "pre", "post"])
    parser.add_argument("--save-annotated", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("./count_results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image is None and args.input_dir is None:
        raise SystemExit("Provide either --image or --input-dir.")
    if args.image is not None and args.input_dir is not None:
        raise SystemExit("Use one mode only: --image OR --input-dir.")
    if args.image is not None:
        run_single(args.image, args.phase, args.save_annotated)
    else:
        run_folder(args.input_dir, args.out_dir, args.phase)


if __name__ == "__main__":
    main()
