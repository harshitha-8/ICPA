#!/usr/bin/env python3
"""
Metric definitions for robust 3D cotton boll phenotyping evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float
    median: float
    ci95_low: float
    ci95_high: float


@dataclass(frozen=True)
class PointCloudQuality:
    chamfer_l1: float
    precision_at_tau: float
    recall_at_tau: float
    fscore_at_tau: float


@dataclass(frozen=True)
class GeometrySanity:
    num_points: int
    finite_fraction: float
    bbox_volume: float
    density_per_bbox_volume: float


def summarize(values: Sequence[float]) -> MetricSummary:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return MetricSummary(*(float("nan"),) * 5)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    sem = std / np.sqrt(arr.size) if arr.size > 1 else 0.0
    return MetricSummary(
        mean=mean,
        std=std,
        median=float(np.median(arr)),
        ci95_low=mean - 1.96 * sem,
        ci95_high=mean + 1.96 * sem,
    )


def boll_recovery_rate(num_3d_bolls: int, num_2d_candidates: int) -> float:
    if num_2d_candidates <= 0:
        return 0.0
    return float(num_3d_bolls / num_2d_candidates)


def mean_absolute_error(pred: Sequence[float], target: Sequence[float]) -> float:
    pred_arr = np.asarray(pred, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    if pred_arr.shape != target_arr.shape:
        raise ValueError("pred and target must have the same shape")
    return float(np.mean(np.abs(pred_arr - target_arr)))


def relative_error(pred: Sequence[float], target: Sequence[float]) -> float:
    pred_arr = np.asarray(pred, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    denom = np.maximum(np.abs(target_arr), 1e-12)
    return float(np.mean(np.abs(pred_arr - target_arr) / denom))


def coefficient_of_variation(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    mean = np.mean(arr)
    if abs(mean) < 1e-12:
        return float("nan")
    return float(np.std(arr, ddof=1) / mean) if arr.size > 1 else 0.0


def geometry_sanity(points: Sequence[Sequence[float]]) -> GeometrySanity:
    """Report lightweight geometry validity checks for benchmark tables.

    This is inspired by modern 3D benchmark practice: before reporting pretty
    images, verify that the point set is finite, non-empty, and reasonably dense
    in its occupied bounding volume.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.size == 0:
        return GeometrySanity(0, 0.0, float("nan"), float("nan"))

    finite_mask = np.all(np.isfinite(pts), axis=1)
    finite_fraction = float(np.mean(finite_mask))
    pts = pts[finite_mask]
    if pts.size == 0:
        return GeometrySanity(0, finite_fraction, float("nan"), float("nan"))

    spans = np.maximum(np.ptp(pts, axis=0), 1e-12)
    bbox_volume = float(np.prod(spans))
    return GeometrySanity(
        num_points=int(len(pts)),
        finite_fraction=finite_fraction,
        bbox_volume=bbox_volume,
        density_per_bbox_volume=float(len(pts) / bbox_volume),
    )


def sampled_pointcloud_quality(
    pred_points: Sequence[Sequence[float]],
    reference_points: Sequence[Sequence[float]],
    tau: float,
    max_points: int = 4096,
) -> PointCloudQuality:
    """Compute simple Chamfer/F-score metrics for sampled point clouds.

    Use this when a LiDAR scan, repeated reconstruction, or manually validated
    reference point set exists. For large fields, downsample before computing
    pairwise distances so the metric remains lightweight and reproducible.
    """
    pred = _sample_points(np.asarray(pred_points, dtype=np.float64), max_points)
    ref = _sample_points(np.asarray(reference_points, dtype=np.float64), max_points)
    if pred.size == 0 or ref.size == 0:
        return PointCloudQuality(*(float("nan"),) * 4)

    pred_to_ref = _nearest_distances(pred, ref)
    ref_to_pred = _nearest_distances(ref, pred)
    precision = float(np.mean(pred_to_ref <= tau))
    recall = float(np.mean(ref_to_pred <= tau))
    denom = max(precision + recall, 1e-12)
    return PointCloudQuality(
        chamfer_l1=float(np.mean(pred_to_ref) + np.mean(ref_to_pred)),
        precision_at_tau=precision,
        recall_at_tau=recall,
        fscore_at_tau=float(2.0 * precision * recall / denom),
    )


def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[idx]


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    chunks = []
    for start in range(0, len(source), 512):
        block = source[start : start + 512]
        dist2 = np.sum((block[:, None, :] - target[None, :, :]) ** 2, axis=2)
        chunks.append(np.sqrt(np.min(dist2, axis=1)))
    return np.concatenate(chunks)
