#!/usr/bin/env python3
"""
Geometry and morphology primitives for detection-guided 3D cotton boll analysis.

The functions here are deliberately small and testable. They define the paper's
computational contract: 2D detections become multi-view tracks, tracks become
3D centers, and 3D centers/masks become morphology estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Camera:
    image_id: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

    @property
    def projection(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])


@dataclass(frozen=True)
class Detection2D:
    image_id: str
    center: Tuple[float, float]
    width: float
    height: float
    score: float = 1.0

    @property
    def pixel_diameter(self) -> float:
        return 0.5 * (self.width + self.height)


@dataclass(frozen=True)
class BollTrack:
    track_id: int
    detections: Tuple[Detection2D, ...]


@dataclass(frozen=True)
class BollMorphology:
    track_id: int
    center_3d: Tuple[float, float, float]
    diameter_mm: float
    volume_mm3: float
    visibility: float
    occlusion: float
    num_views: int


def linear_triangulate(points: List[Tuple[float, float]], cameras: List[Camera]) -> np.ndarray:
    """Triangulate one 3D point from corresponding image observations."""
    if len(points) != len(cameras):
        raise ValueError("points and cameras must have equal length")
    if len(points) < 2:
        raise ValueError("at least two views are required")

    rows = []
    for (u, v), camera in zip(points, cameras):
        P = camera.projection
        rows.append(u * P[2, :] - P[0, :])
        rows.append(v * P[2, :] - P[1, :])
    A = np.asarray(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(A)
    X = vt[-1, :]
    X = X / (X[3] + 1e-12)
    return X[:3]


def reprojection_error(point_3d: np.ndarray, detection: Detection2D, camera: Camera) -> float:
    """Compute pixel reprojection error for one detection."""
    X = np.append(point_3d, 1.0)
    x = camera.projection @ X
    x = x[:2] / (x[2] + 1e-12)
    observed = np.asarray(detection.center, dtype=np.float64)
    return float(np.linalg.norm(x - observed))


def robust_track_center(track: BollTrack, camera_lookup: dict) -> Tuple[np.ndarray, float]:
    """Triangulate a track and return its median reprojection error."""
    detections = [d for d in track.detections if d.image_id in camera_lookup]
    if len(detections) < 2:
        raise ValueError("track needs at least two detections with known cameras")

    cameras = [camera_lookup[d.image_id] for d in detections]
    points = [d.center for d in detections]
    center = linear_triangulate(points, cameras)
    errors = [reprojection_error(center, d, camera_lookup[d.image_id]) for d in detections]
    return center, float(np.median(errors))


def estimate_diameter_from_views(
    detections: Iterable[Detection2D],
    mm_per_pixel: Iterable[float],
) -> float:
    """Estimate boll diameter as the median metric diameter across views."""
    diameters = [
        det.pixel_diameter * scale
        for det, scale in zip(detections, mm_per_pixel)
        if det.pixel_diameter > 0 and scale > 0
    ]
    if not diameters:
        return float("nan")
    return float(np.median(diameters))


def ellipsoid_volume_mm3(diameter_major_mm: float, diameter_minor_mm: Optional[float] = None) -> float:
    """
    Estimate volume using a conservative ellipsoid model.

    If only one diameter is available, the boll is approximated as a sphere.
    """
    if diameter_minor_mm is None:
        diameter_minor_mm = diameter_major_mm
    a = 0.5 * diameter_major_mm
    b = 0.5 * diameter_minor_mm
    c = 0.5 * np.sqrt(max(diameter_major_mm * diameter_minor_mm, 0.0))
    return float((4.0 / 3.0) * np.pi * a * b * c)


def visibility_score(num_detected_views: int, num_expected_views: int) -> float:
    """Fraction of overlapping views where the boll was detected."""
    if num_expected_views <= 0:
        return 0.0
    return float(np.clip(num_detected_views / num_expected_views, 0.0, 1.0))


def summarize_track(
    track: BollTrack,
    camera_lookup: dict,
    mm_per_pixel_by_image: dict,
    expected_views: int,
    max_reprojection_error: float = 20.0,
) -> Optional[BollMorphology]:
    """Convert a multi-view boll track into one morphology record."""
    try:
        center, reproj = robust_track_center(track, camera_lookup)
    except ValueError:
        return None
    if reproj > max_reprojection_error:
        return None

    scales = [mm_per_pixel_by_image.get(d.image_id, float("nan")) for d in track.detections]
    diameter = estimate_diameter_from_views(track.detections, scales)
    if not np.isfinite(diameter):
        return None

    visibility = visibility_score(len(track.detections), expected_views)
    return BollMorphology(
        track_id=track.track_id,
        center_3d=(float(center[0]), float(center[1]), float(center[2])),
        diameter_mm=diameter,
        volume_mm3=ellipsoid_volume_mm3(diameter),
        visibility=visibility,
        occlusion=1.0 - visibility,
        num_views=len(track.detections),
    )
