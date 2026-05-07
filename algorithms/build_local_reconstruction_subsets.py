#!/usr/bin/env python3
"""Build local overlapping image subsets for cotton boll 3D reconstruction.

This script consumes the measurement-ready candidate CSV and creates small
phase-specific image windows around high-ranking frames. The folders are meant
for COLMAP/OpenDroneMap/NAS3R/MASt3R/VGGT smoke tests. Raw copied images are
local artifacts and are ignored by Git; the CSV manifests are committed.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path


def read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sibling_images(anchor: Path) -> list[Path]:
    valid = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    return [
        p
        for p in sorted(anchor.parent.iterdir())
        if p.is_file() and p.suffix.lower() in valid and not p.name.startswith("._")
    ]


def centered_window(images: list[Path], anchor: Path, size: int) -> list[Path]:
    if anchor not in images:
        return images[:size]
    idx = images.index(anchor)
    start = max(0, idx - size // 2)
    end = min(len(images), start + size)
    start = max(0, end - size)
    return images[start:end]


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def choose_anchors(candidates: list[dict[str, str]], phase: str, count: int) -> list[dict[str, str]]:
    anchors: list[dict[str, str]] = []
    used_folders: set[str] = set()
    used_images: set[str] = set()
    phase_rows = [row for row in candidates if row["phase"] == phase]
    phase_rows.sort(key=lambda row: float(row["measurement_ready_score"]), reverse=True)

    for row in phase_rows:
        folder = row["folder"]
        image = row["image"]
        if image in used_images:
            continue
        # Prefer coverage over folders first, then fill remaining anchors.
        if folder in used_folders and len(used_folders) < count:
            continue
        anchors.append(row)
        used_folders.add(folder)
        used_images.add(image)
        if len(anchors) >= count:
            return anchors

    for row in phase_rows:
        image = row["image"]
        if image in used_images:
            continue
        anchors.append(row)
        used_images.add(image)
        if len(anchors) >= count:
            break
    return anchors


def build_subset(
    anchor_row: dict[str, str],
    phase: str,
    subset_index: int,
    window_size: int,
    out_dir: Path,
    copy_images: bool,
) -> dict[str, str | int | float]:
    anchor = Path(anchor_row["image"])
    images = sibling_images(anchor)
    window = centered_window(images, anchor, window_size)
    subset_name = f"{phase}_subset{subset_index:02d}_{window_size}frames_{anchor.stem}"
    subset_dir = out_dir / subset_name
    image_dir = subset_dir / "images"
    if copy_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float]] = []
    for idx, image_path in enumerate(window, start=1):
        copied_path = image_dir / f"{idx:04d}_{image_path.name}"
        if copy_images and not copied_path.exists():
            shutil.copyfile(image_path, copied_path)
        rows.append(
            {
                "index": idx,
                "phase": phase,
                "source": str(image_path),
                "copied_path": str(copied_path) if copy_images else "",
                "is_anchor": int(image_path == anchor),
                "anchor_candidate_rank": anchor_row["rank"],
                "anchor_candidate_score": anchor_row["measurement_ready_score"],
            }
        )
    write_csv(subset_dir / "image_manifest.csv", rows)
    (subset_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {subset_name}",
                "",
                "Purpose: local image window for cotton boll 3D reconstruction tests.",
                "",
                f"- Phase: {phase}",
                f"- Anchor image: `{anchor}`",
                f"- Window size: {len(window)}",
                f"- Anchor measurement-ready score: {anchor_row['measurement_ready_score']}",
                "",
                "Suggested tests: COLMAP/OpenDroneMap first; learned reconstruction if classical matching fails.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "subset": subset_name,
        "phase": phase,
        "anchor_image": str(anchor),
        "anchor_rank": int(anchor_row["rank"]),
        "anchor_score": float(anchor_row["measurement_ready_score"]),
        "folder": anchor.parent.name,
        "window_size": len(window),
        "manifest": str(subset_dir / "image_manifest.csv"),
        "image_dir": str(image_dir) if copy_images else "",
    }


def run(args: argparse.Namespace) -> None:
    candidates = read_candidates(args.candidates_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str | int | float]] = []
    phase_anchor_counts = defaultdict(int)

    for phase in ("post", "pre"):
        anchors = choose_anchors(candidates, phase, args.anchors_per_phase)
        for anchor_row in anchors:
            phase_anchor_counts[phase] += 1
            for window_size in args.window_sizes:
                summary_rows.append(
                    build_subset(
                        anchor_row=anchor_row,
                        phase=phase,
                        subset_index=phase_anchor_counts[phase],
                        window_size=window_size,
                        out_dir=args.out_dir,
                        copy_images=not args.manifest_only,
                    )
                )

    write_csv(args.out_dir / "local_reconstruction_subsets.csv", summary_rows)
    manifest = {
        "artifact_type": "local reconstruction subset builder",
        "source_candidates": str(args.candidates_csv),
        "out_dir": str(args.out_dir),
        "anchors_per_phase": args.anchors_per_phase,
        "window_sizes": args.window_sizes,
        "manifest_only": args.manifest_only,
        "summary_csv": str(args.out_dir / "local_reconstruction_subsets.csv"),
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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/metrics/measurement_ready_bolls/local_reconstruction_subsets"),
    )
    parser.add_argument("--anchors-per-phase", type=int, default=3)
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[20, 60])
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
