#!/usr/bin/env python3
"""
Create a lightweight scene-viewer manifest for field-scale 3D inspection.

This script does not perform reconstruction. It packages outputs from the
reconstruction and morphology stages into a simple JSON contract that a WebGL,
Open3D, Potree, or Gaussian-splat viewer can consume.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CameraPose:
    frame: str
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float


@dataclass(frozen=True)
class BollAnchor:
    boll_id: str
    x: float
    y: float
    z: float
    diameter_mm: float | None
    volume_mm3: float | None
    visibility: float | None
    occlusion: float | None
    phase: str | None
    confidence: float | None


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def read_camera_poses(path: Path) -> list[CameraPose]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            CameraPose(
                frame=row["frame"],
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row["z"]),
                qw=float(row.get("qw", 1.0)),
                qx=float(row.get("qx", 0.0)),
                qy=float(row.get("qy", 0.0)),
                qz=float(row.get("qz", 0.0)),
            )
            for row in reader
        ]


def read_boll_anchors(path: Path) -> list[BollAnchor]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            BollAnchor(
                boll_id=row.get("boll_id", row.get("id", "")),
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row["z"]),
                diameter_mm=_to_float(row.get("diameter_mm")),
                volume_mm3=_to_float(row.get("volume_mm3")),
                visibility=_to_float(row.get("visibility")),
                occlusion=_to_float(row.get("occlusion")),
                phase=row.get("phase") or None,
                confidence=_to_float(row.get("confidence")),
            )
            for row in reader
        ]


def build_manifest(
    scene_name: str,
    geometry_path: Path,
    camera_poses: Iterable[CameraPose],
    boll_anchors: Iterable[BollAnchor],
) -> dict:
    return {
        "schema": "icpa.field_scene.v1",
        "scene_name": scene_name,
        "geometry": {
            "path": str(geometry_path),
            "type": geometry_path.suffix.lower().lstrip(".") or "unknown",
        },
        "camera_path": [asdict(pose) for pose in camera_poses],
        "boll_anchors": [asdict(anchor) for anchor in boll_anchors],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a 3D field-viewer manifest.")
    parser.add_argument("--scene-name", required=True, help="Human-readable scene name.")
    parser.add_argument("--geometry", type=Path, required=True, help="Point cloud, mesh, or splat file.")
    parser.add_argument("--camera-poses", type=Path, required=True, help="CSV with frame,x,y,z,qw,qx,qy,qz.")
    parser.add_argument("--boll-anchors", type=Path, required=True, help="CSV with boll 3D measurements.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        scene_name=args.scene_name,
        geometry_path=args.geometry,
        camera_poses=read_camera_poses(args.camera_poses),
        boll_anchors=read_boll_anchors(args.boll_anchors),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
