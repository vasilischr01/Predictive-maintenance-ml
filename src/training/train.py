from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.validate import validate_dataframe

DATA_PATH = Path("data/raw/ai4i2020.csv")

MODEL_PATH = Path("artifacts/model.joblib")
METRICS_PATH = Path("artifacts/metrics.json")
THRESHOLD_PATH = Path("artifacts/selected_threshold.json")

TARGET = "Machine failure"

CATEGORICAL_FEATURES = ["Type"]

NUMERIC_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

RANDOM_STATE = 42

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "predictive-maintenance",
)


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Create deterministic stratified train/validation/test splits.

    Final proportions:
        Train:      70%
        Validation: 15%
        Test:       15%
    """

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    validation_fraction_of_train_val = (
        VALIDATION_SIZE / (1.0 - TEST_SIZE)
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_fraction_of_train_val,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


def evaluate(
    y_true,
    probabilities,
    threshold: float,
) -> dict:
    predictions = (
        np.asarray(probabilities) >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
    ).ravel()

    return {
        "threshold": round(float(threshold), 4),
        "precision": round(
            float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "roc_auc": round(
            float(
                roc_auc_score(
                    y_true,
                    probabilities,
                )
            ),
            4,
        ),
        "pr_auc": round(
            float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            ),
            4,
        ),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def select_threshold(
    y_true,
    probabilities,
    thresholds: np.ndarray | None = None,
) -> tuple[float, list[dict]]:
    """
    Select the threshold that maximizes validation F1.

    Threshold selection is performed only on validation data.
    """

    if thresholds is None:
        thresholds = np.arange(
            0.10,
            0.91,
            0.01,
        )

    results = []

    for threshold in thresholds:
        metrics = evaluate(
            y_true,
            probabilities,
            float(threshold),
        )

        results.append(metrics)

    best_result = max(
        results,
        key=lambda item: (
            item["f1"],
            item["recall"],
            item["precision"],
        ),
    )

    return (
        float(best_result["threshold"]),
        results,
    )


def train(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    threshold_path: Path = THRESHOLD_PATH,
) -> dict:
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} does not exist. "
            "Run: python -m src.data.download"
        )

    df = pd.read_csv(data_path)

    validate_dataframe(df)

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(
        X,
        y,
    )

    pipeline = build_pipeline()

    pipeline.fit(
        X_train,
        y_train,
    )

    validation_probabilities = (
        pipeline.predict_proba(X_val)[:, 1]
    )

    selected_threshold, threshold_results = (
        select_threshold(
            y_val,
            validation_probabilities,
        )
    )

    validation_metrics = evaluate(
        y_val,
        validation_probabilities,
        selected_threshold,
    )

    # The test set is evaluated only after the operating
    # threshold has been selected on validation data.
    test_probabilities = (
        pipeline.predict_proba(X_test)[:, 1]
    )

    test_metrics = evaluate(
        y_test,
        test_probabilities,
        selected_threshold,
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    threshold_payload = {
        "selected_threshold": selected_threshold,
        "selection_metric": "f1",
        "selection_split": "validation",
        "candidate_thresholds": len(
            threshold_results
        ),
    }

    threshold_path.write_text(
        json.dumps(
            threshold_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = {
        "model": "HistGradientBoostingClassifier",
        "random_state": RANDOM_STATE,
        "dataset_rows": len(df),
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
        "positive_rate": {
            "train": round(
                float(y_train.mean()),
                4,
            ),
            "validation": round(
                float(y_val.mean()),
                4,
            ),
            "test": round(
                float(y_test.mean()),
                4,
            ),
        },
        "threshold_selection": threshold_payload,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    metrics_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name="hist-gradient-boosting-validation-threshold",
    ):
        mlflow.log_params(
            {
                "model": "HistGradientBoostingClassifier",
                "max_iter": 300,
                "learning_rate": 0.05,
                "max_leaf_nodes": 31,
                "random_state": RANDOM_STATE,
                "selected_threshold": selected_threshold,
                "threshold_selection_metric": "f1",
                "train_fraction": 0.70,
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
            }
        )

        mlflow.log_metrics(
            {
                "validation_precision": (
                    validation_metrics["precision"]
                ),
                "validation_recall": (
                    validation_metrics["recall"]
                ),
                "validation_f1": (
                    validation_metrics["f1"]
                ),
                "validation_roc_auc": (
                    validation_metrics["roc_auc"]
                ),
                "validation_pr_auc": (
                    validation_metrics["pr_auc"]
                ),
                "test_precision": (
                    test_metrics["precision"]
                ),
                "test_recall": (
                    test_metrics["recall"]
                ),
                "test_f1": (
                    test_metrics["f1"]
                ),
                "test_roc_auc": (
                    test_metrics["roc_auc"]
                ),
                "test_pr_auc": (
                    test_metrics["pr_auc"]
                ),
            }
        )

        input_example = X_train.iloc[:5].copy()

        # Cast numeric columns to float in the MLflow input example.
        # This avoids integer-schema issues when future inference
        # requests contain missing numeric values.
        for column in NUMERIC_FEATURES:
            input_example[column] = input_example[column].astype(float)

        mlflow.sklearn.log_model(
            pipeline,
            name="predictive-maintenance-model",
            input_example=input_example,
            serialization_format="cloudpickle",
        )

        mlflow.log_artifact(
            str(metrics_path)
        )

        mlflow.log_artifact(
            str(threshold_path)
        )

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )

    print(
        f"Saved model to {model_path}"
    )

    return payload


if __name__ == "__main__":
    train()