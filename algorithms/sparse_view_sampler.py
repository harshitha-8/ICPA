#!/usr/bin/env python3
"""Create dense and sparse-view manifests for reconstruction robustness tests.

The design is inspired by MegaDepth-X style evaluation: keep the image content
fixed, then vary the number and spacing of input views to measure how geometry
and organ-level traits degrade.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "frame_index",
        "source_image",
        "phase",
        "folder",
        "viewer_image",
        "sparse_split",
        "split_frame_index",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("phase", ""), row.get("folder", ""))].append(row)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda item: int(item.get("frame_index", "0")))
    return dict(grouped)


def sample_stride(rows: list[dict[str, str]], stride: int, split_name: str) -> list[dict[str, str]]:
    grouped = group_rows(rows)
    sampled: list[dict[str, str]] = []
    for key in sorted(grouped):
        for row in grouped[key][::stride]:
            copied = dict(row)
            copied["sparse_split"] = split_name
            copied["split_frame_index"] = str(len(sampled))
            sampled.append(copied)
    return sampled


def sample_budget(rows: list[dict[str, str]], per_group: int, split_name: str) -> list[dict[str, str]]:
    grouped = group_rows(rows)
    sampled: list[dict[str, str]] = []
    for key in sorted(grouped):
        group = grouped[key]
        if len(group) <= per_group:
            chosen = group
        else:
            indices = evenly_spaced_indices(len(group), per_group)
            chosen = [group[index] for index in indices]
        for row in chosen:
            copied = dict(row)
            copied["sparse_split"] = split_name
            copied["split_frame_index"] = str(len(sampled))
            sampled.append(copied)
    return sampled


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 1:
        return [0]
    return sorted({round(i * (length - 1) / (count - 1)) for i in range(count)})


def summarize(rows_by_split: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    for split, rows in rows_by_split.items():
        grouped = group_rows(rows)
        for (phase, folder), group in sorted(grouped.items()):
            summary.append(
                {
                    "split": split,
                    "phase": phase,
                    "folder": folder,
                    "num_frames": str(len(group)),
                }
            )
    return summary


def build_splits(
    manifest: Path,
    out_dir: Path,
    strides: list[int],
    per_group_budget: int | None,
) -> None:
    rows = read_manifest(manifest)
    if not rows:
        raise RuntimeError(f"No rows in manifest: {manifest}")

    rows_by_split: dict[str, list[dict[str, str]]] = {}
    for stride in strides:
        split = f"stride_{stride}"
        rows_by_split[split] = sample_stride(rows, stride=stride, split_name=split)
        write_manifest(out_dir / f"{split}.csv", rows_by_split[split])

    if per_group_budget is not None:
        split = f"balanced_{per_group_budget}_per_group"
        rows_by_split[split] = sample_budget(rows, per_group=per_group_budget, split_name=split)
        write_manifest(out_dir / f"{split}.csv", rows_by_split[split])

    summary_rows = summarize(rows_by_split)
    write_manifest(out_dir / "sparse_view_summary.csv", summary_rows)
    print(f"wrote sparse-view manifests to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create sparse-view reconstruction manifests.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--per-group-budget", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_splits(
        manifest=args.manifest,
        out_dir=args.out_dir,
        strides=args.strides,
        per_group_budget=args.per_group_budget,
    )


if __name__ == "__main__":
    main()
