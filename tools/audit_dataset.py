#!/usr/bin/env python3
"""
Audit the configured UAV cotton dataset before running expensive experiments.

This script is intentionally lightweight: it counts images, parses DJI timestamps
from filenames, checks duplicate basenames across folders, and compares observed
counts with the manuscript/config claims. It writes JSON and Markdown reports
under outputs/metrics by default.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path_value: str, base: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else base / path


def parse_dji_timestamp(filename: str) -> Optional[str]:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 3 or parts[0] != "DJI":
        return None
    try:
        return datetime.strptime(parts[1], "%Y%m%d%H%M%S").isoformat()
    except ValueError:
        return None


def collect_folder(folder: Path, extensions: list[str]) -> dict:
    files = []
    for ext in extensions:
        files.extend(folder.glob(f"*{ext}"))
    files = sorted(p for p in files if not p.name.startswith("._"))

    timestamps = [parse_dji_timestamp(p.name) for p in files]
    timestamps = [t for t in timestamps if t]

    return {
        "folder": str(folder),
        "exists": folder.exists(),
        "image_count": len(files),
        "first_file": files[0].name if files else None,
        "last_file": files[-1].name if files else None,
        "timestamp_earliest": min(timestamps) if timestamps else None,
        "timestamp_latest": max(timestamps) if timestamps else None,
        "basenames": [p.name for p in files],
    }


def audit(config_path: Path) -> dict:
    root = repo_root()
    config = yaml.safe_load(config_path.read_text())
    dataset_root = resolve_path(config["dataset"]["root"], root)
    extensions = config["preprocessing"].get("image_extensions", [".JPG", ".jpg"])

    report = {
        "config_path": str(config_path),
        "dataset_root": str(dataset_root),
        "capture_date": config["dataset"].get("capture_date"),
        "conditions": {},
        "global_duplicate_basenames": [],
        "warnings": [],
    }

    all_basenames = []
    basename_locations = defaultdict(list)

    for condition_key in ["pre_defoliation", "post_defoliation"]:
        condition_cfg = config["dataset"][condition_key]
        folders = []
        for folder_name in condition_cfg["folders"]:
            folder_report = collect_folder(dataset_root / folder_name, extensions)
            folders.append(folder_report)
            for basename in folder_report["basenames"]:
                all_basenames.append(basename)
                basename_locations[basename].append(folder_report["folder"])

        observed = sum(folder["image_count"] for folder in folders)
        expected = condition_cfg.get("total_images")
        condition_report = {
            "description": condition_cfg.get("description"),
            "expected_total_images": expected,
            "observed_total_images": observed,
            "count_delta": observed - expected if expected is not None else None,
            "folders": [
                {k: v for k, v in folder.items() if k != "basenames"}
                for folder in folders
            ],
        }
        report["conditions"][condition_key] = condition_report

        if expected is not None and observed != expected:
            report["warnings"].append(
                f"{condition_key}: observed {observed} images but config expects {expected}."
            )

    duplicates = {
        basename: locations
        for basename, locations in basename_locations.items()
        if len(locations) > 1
    }
    report["global_duplicate_basenames"] = [
        {"filename": name, "locations": locations}
        for name, locations in sorted(duplicates.items())
    ]
    if duplicates:
        report["warnings"].append(
            f"Found {len(duplicates)} duplicate image basenames across configured folders."
        )

    report["grand_total_observed"] = len(all_basenames)
    report["unique_basenames"] = len(Counter(all_basenames))
    return report


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Dataset Audit",
        "",
        f"- Dataset root: `{report['dataset_root']}`",
        f"- Capture date: `{report['capture_date']}`",
        f"- Grand total observed: `{report['grand_total_observed']}`",
        f"- Unique basenames: `{report['unique_basenames']}`",
        "",
        "## Condition Counts",
        "",
        "| Condition | Expected | Observed | Delta |",
        "|---|---:|---:|---:|",
    ]
    for condition, data in report["conditions"].items():
        lines.append(
            f"| {condition} | {data['expected_total_images']} | "
            f"{data['observed_total_images']} | {data['count_delta']} |"
        )

    lines.extend(["", "## Folder Counts", ""])
    for condition, data in report["conditions"].items():
        lines.extend([f"### {condition}", ""])
        for folder in data["folders"]:
            lines.extend(
                [
                    f"- `{folder['folder']}`",
                    f"  - Exists: `{folder['exists']}`",
                    f"  - Images: `{folder['image_count']}`",
                    f"  - Timestamp range: `{folder['timestamp_earliest']}` to `{folder['timestamp_latest']}`",
                    "",
                ]
            )

    lines.extend(["## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- None")

    if report["global_duplicate_basenames"]:
        lines.extend(["", "## Duplicate Basenames", ""])
        for item in report["global_duplicate_basenames"][:100]:
            lines.append(f"- `{item['filename']}` in {len(item['locations'])} folders")
        if len(report["global_duplicate_basenames"]) > 100:
            lines.append(
                f"- ... {len(report['global_duplicate_basenames']) - 100} more omitted"
            )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset_config.yaml")
    parser.add_argument("--out-dir", default="outputs/metrics")
    args = parser.parse_args()

    root = repo_root()
    config_path = resolve_path(args.config, root)
    out_dir = resolve_path(args.out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = audit(config_path)
    json_path = out_dir / "dataset_audit.json"
    md_path = out_dir / "dataset_audit.md"
    json_path.write_text(json.dumps(report, indent=2))
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
