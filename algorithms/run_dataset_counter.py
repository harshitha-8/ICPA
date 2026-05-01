#!/usr/bin/env python3
"""
Run the cotton boll counter over a mixed pre/post-defoliation dataset.

The dataset can contain folders such as:
  - 205_Post_Def_rgb
  - Post_def_rgb_part1
  - Part_one_pre_def_rgb
  - part 2_pre_def_rgb

Phase is resolved from the path, not from image content, so the experiment
uses the acquisition protocol rather than a visual heuristic.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from cotton_boll_detector import VALID_EXT, detect_cotton_bolls


def list_images(dataset_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(dataset_root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in VALID_EXT
        and not path.name.startswith("._")
    ]


def phase_from_path(path: Path) -> str:
    text = str(path).lower()
    name = path.name.lower()
    if name.startswith("pre") or "/pre" in text or "_pre" in text or " pre" in text:
        return "pre"
    if name.startswith("post") or "/post" in text or "_post" in text or " post" in text:
        return "post"
    raise ValueError(f"Could not infer pre/post phase from path: {path}")


def run_dataset(
    dataset_root: Path,
    out_dir: Path,
    max_images: int | None,
    save_annotated_limit: int,
    write_candidates: bool,
) -> None:
    images = list_images(dataset_root)
    if max_images is not None:
        images = images[:max_images]
    if not images:
        raise RuntimeError(f"No images found under {dataset_root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = out_dir / "annotated_samples"
    if save_annotated_limit > 0:
        ann_dir.mkdir(parents=True, exist_ok=True)

    counts_path = out_dir / "counts_by_image.csv"
    candidates_path = out_dir / "boll_candidates.csv"

    count_fields = [
        "image",
        "folder",
        "phase",
        "count",
        "num_raw_candidates",
        "annotated_path",
    ]
    candidate_fields = [
        "image",
        "phase",
        "candidate_id",
        "x",
        "y",
        "width",
        "height",
        "center_x",
        "center_y",
        "area",
        "mean_saturation",
        "mean_value",
    ]

    phase_totals = {
        "pre": {"images": 0, "count": 0, "raw_candidates": 0},
        "post": {"images": 0, "count": 0, "raw_candidates": 0},
    }

    with counts_path.open("w", newline="", encoding="utf-8") as counts_file:
        count_writer = csv.DictWriter(counts_file, fieldnames=count_fields)
        count_writer.writeheader()

        candidate_file = None
        candidate_writer = None
        if write_candidates:
            candidate_file = candidates_path.open("w", newline="", encoding="utf-8")
            candidate_writer = csv.DictWriter(candidate_file, fieldnames=candidate_fields)
            candidate_writer.writeheader()

        try:
            for idx, image_path in enumerate(images, start=1):
                phase = phase_from_path(image_path)
                bgr = cv2.imread(str(image_path))
                if bgr is None:
                    print(f"[skip] could not read {image_path}")
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                annotated, count, candidates = detect_cotton_bolls(rgb, phase)

                annotated_path = ""
                if idx <= save_annotated_limit:
                    annotated_path = str(ann_dir / f"{image_path.stem}_{phase}_annotated.jpg")
                    cv2.imwrite(annotated_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

                count_writer.writerow(
                    {
                        "image": str(image_path),
                        "folder": image_path.parent.name,
                        "phase": phase,
                        "count": count,
                        "num_raw_candidates": len(candidates),
                        "annotated_path": annotated_path,
                    }
                )

                phase_totals[phase]["images"] += 1
                phase_totals[phase]["count"] += count
                phase_totals[phase]["raw_candidates"] += len(candidates)

                if candidate_writer is not None:
                    for cand_idx, cand in enumerate(candidates, start=1):
                        cx, cy = cand.center
                        candidate_writer.writerow(
                            {
                                "image": str(image_path),
                                "phase": phase,
                                "candidate_id": cand_idx,
                                "x": cand.x,
                                "y": cand.y,
                                "width": cand.width,
                                "height": cand.height,
                                "center_x": f"{cx:.3f}",
                                "center_y": f"{cy:.3f}",
                                "area": f"{cand.area:.3f}",
                                "mean_saturation": f"{cand.mean_saturation:.3f}",
                                "mean_value": f"{cand.mean_value:.3f}",
                            }
                        )

                if idx == 1 or idx % 25 == 0 or idx == len(images):
                    print(f"[{idx}/{len(images)}] {image_path.name}: {count} ({phase})")
        finally:
            if candidate_file is not None:
                candidate_file.close()

    summary_path = out_dir / "phase_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["phase", "images", "total_count", "total_raw_candidates", "mean_count"],
        )
        writer.writeheader()
        for phase, stats in phase_totals.items():
            images_n = stats["images"]
            mean_count = stats["count"] / images_n if images_n else 0.0
            writer.writerow(
                {
                    "phase": phase,
                    "images": images_n,
                    "total_count": stats["count"],
                    "total_raw_candidates": stats["raw_candidates"],
                    "mean_count": f"{mean_count:.3f}",
                }
            )

    print(f"counts: {counts_path}")
    print(f"summary: {summary_path}")
    if write_candidates:
        print(f"candidates: {candidates_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mixed pre/post cotton boll counting.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/counts/icml_dataset"))
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-annotated-limit", type=int, default=12)
    parser.add_argument("--write-candidates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dataset(
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        max_images=args.max_images,
        save_annotated_limit=args.save_annotated_limit,
        write_candidates=args.write_candidates,
    )


if __name__ == "__main__":
    main()
