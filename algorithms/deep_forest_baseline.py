#!/usr/bin/env python3
"""Cascade-forest baseline for cotton candidate or plot-cell prediction.

This is a practical baseline inspired by the deep-forest figure in the close
ISPRS cotton 3D reconstruction paper. It is intentionally tabular: it consumes
features exported by the app or future plot-cell summaries. It does not perform
3D reconstruction.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

DEFAULT_FEATURES = [
    "diameter_px",
    "diameter_cm_proxy",
    "volume_cm3_proxy",
    "visibility_proxy",
    "depth_score",
    "lint_fraction",
    "green_fraction",
    "brightness_score",
    "size_score",
    "extraction_quality",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def matrix_from_rows(
    rows: list[dict[str, str]],
    feature_names: list[str],
    target_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    usable = [row for row in rows if target_name in row and row[target_name] not in {"", "nan", "NaN"}]
    if not usable:
        raise RuntimeError(f"No labeled rows found for target '{target_name}'")

    x = np.asarray(
        [[float(row.get(name, "0") or 0.0) for name in feature_names] for row in usable],
        dtype=np.float64,
    )
    y = np.asarray([float(row[target_name]) for row in usable], dtype=np.float64)
    return x, y


def train_classifier(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    y = y.astype(int)
    stratify = y if len(set(y.tolist())) > 1 and min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=7,
        stratify=stratify,
    )
    layer_a = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=7, class_weight="balanced")
    layer_b = ExtraTreesClassifier(n_estimators=300, max_depth=None, random_state=11, class_weight="balanced")
    layer_a.fit(x_train, y_train)
    layer_b.fit(x_train, y_train)

    train_aug = augment_with_probabilities(x_train, [layer_a, layer_b])
    test_aug = augment_with_probabilities(x_test, [layer_a, layer_b])
    cascade = RandomForestClassifier(n_estimators=300, random_state=17, class_weight="balanced")
    cascade.fit(train_aug, y_train)
    pred = cascade.predict(test_aug)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, average="binary" if len(set(y.tolist())) == 2 else "macro")),
    }
    if len(set(y.tolist())) == 2:
        prob = cascade.predict_proba(test_aug)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, prob))
    return metrics


def train_regressor(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=7)
    layer_a = RandomForestRegressor(n_estimators=300, random_state=7)
    layer_b = RandomForestRegressor(n_estimators=300, random_state=11, max_features=0.75)
    layer_a.fit(x_train, y_train)
    layer_b.fit(x_train, y_train)
    train_aug = np.column_stack([x_train, layer_a.predict(x_train), layer_b.predict(x_train)])
    test_aug = np.column_stack([x_test, layer_a.predict(x_test), layer_b.predict(x_test)])
    cascade = RandomForestRegressor(n_estimators=300, random_state=17)
    cascade.fit(train_aug, y_train)
    pred = cascade.predict(test_aug)
    return {
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }


def augment_with_probabilities(x: np.ndarray, models: list[object]) -> np.ndarray:
    prob_blocks = [model.predict_proba(x) for model in models]
    return np.column_stack([x, *prob_blocks])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a cascade-forest baseline from exported cotton features.")
    parser.add_argument("--csv", type=Path, required=True, help="Feature CSV from the app or a labeled annotation table.")
    parser.add_argument("--target", required=True, help="Target column, e.g. valid_boll or yield.")
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.csv)
    x, y = matrix_from_rows(rows, args.features, args.target)
    metrics = train_classifier(x, y) if args.task == "classification" else train_regressor(x, y)
    print(metrics)


if __name__ == "__main__":
    main()
