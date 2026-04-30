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
