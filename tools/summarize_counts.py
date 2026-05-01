#!/usr/bin/env python3
"""Summarize per-image boll counts by folder and phase."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def summarize(counts_csv: Path, out_csv: Path) -> None:
    groups: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"images": 0, "count": 0.0, "raw_candidates": 0.0}
    )
    with counts_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["phase"], row["folder"])
            groups[key]["images"] += 1
            groups[key]["count"] += float(row["count"])
            groups[key]["raw_candidates"] += float(row["num_raw_candidates"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "folder",
                "images",
                "total_count",
                "total_raw_candidates",
                "mean_count",
                "mean_raw_candidates",
            ],
        )
        writer.writeheader()
        for (phase, folder), stats in sorted(groups.items()):
            images = stats["images"]
            writer.writerow(
                {
                    "phase": phase,
                    "folder": folder,
                    "images": int(images),
                    "total_count": int(stats["count"]),
                    "total_raw_candidates": int(stats["raw_candidates"]),
                    "mean_count": f"{stats['count'] / images:.3f}",
                    "mean_raw_candidates": f"{stats['raw_candidates'] / images:.3f}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize count CSV by folder.")
    parser.add_argument("--counts-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize(args.counts_csv, args.out_csv)
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
