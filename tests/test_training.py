from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)

from src.training.train import (
    build_pipeline,
    evaluate,
    select_threshold,
    split_dataset,
)


def make_dataset(
    rows: int = 1000,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)

    y = np.zeros(
        rows,
        dtype=int,
    )

    y[:100] = 1

    rng.shuffle(y)

    X = pd.DataFrame(
        {
            "Type": rng.choice(
                ["L", "M", "H"],
                size=rows,
            ),
            "Air temperature": rng.normal(
                300,
                2,
                size=rows,
            ),
            "Process temperature": rng.normal(
                310,
                2,
                size=rows,
            ),
            "Rotational speed": rng.normal(
                1500,
                150,
                size=rows,
            ),
            "Torque": rng.normal(
                40,
                10,
                size=rows,
            ),
            "Tool wear": rng.uniform(
                0,
                250,
                size=rows,
            ),
        }
    )

    return X, pd.Series(y)


def test_split_dataset_has_expected_sizes():
    X, y = make_dataset()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(X, y)

    assert len(X_train) == 700
    assert len(X_val) == 150
    assert len(X_test) == 150

    assert len(y_train) == 700
    assert len(y_val) == 150
    assert len(y_test) == 150


def test_split_dataset_preserves_class_ratio():
    X, y = make_dataset()

    (
        _,
        _,
        _,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(X, y)

    original_rate = y.mean()

    assert abs(
        y_train.mean() - original_rate
    ) < 0.02

    assert abs(
        y_val.mean() - original_rate
    ) < 0.02

    assert abs(
        y_test.mean() - original_rate
    ) < 0.02


def test_threshold_selection_uses_best_f1():
    y_true = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.1, 0.4, 0.6, 0.9]
    )

    threshold, results = select_threshold(
        y_true,
        probabilities,
        thresholds=np.array(
            [0.3, 0.5, 0.7]
        ),
    )

    assert threshold == 0.5
    assert len(results) == 3


def test_evaluate_returns_expected_metrics():
    y_true = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.1, 0.2, 0.8, 0.9]
    )

    metrics = evaluate(
        y_true,
        probabilities,
        threshold=0.5,
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0

    assert metrics[
        "confusion_matrix"
    ] == {
        "tn": 2,
        "fp": 0,
        "fn": 0,
        "tp": 2,
    }


def test_production_pipeline_uses_hist_gradient_boosting():
    pipeline = build_pipeline()

    classifier = pipeline.named_steps[
        "classifier"
    ]

    assert isinstance(
        classifier,
        HistGradientBoostingClassifier,
    )