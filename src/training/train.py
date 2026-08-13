from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
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

TARGET = "Machine failure"
CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

RANDOM_STATE = 42

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "predictive-maintenance"


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate(y_true, probabilities, threshold: float = 0.5) -> dict:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
    ).ravel()

    return {
        "threshold": threshold,
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
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def train(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict:
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} does not exist. "
            "Run: python -m src.data.download"
        )

    df = pd.read_csv(data_path)
    validate_dataframe(df)

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name="random-forest-baseline-0.40",
    ):
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate(
            y_test,
            probabilities,
            threshold=0.40
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

        payload = {
            "model": "RandomForestClassifier",
            "random_state": RANDOM_STATE,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "positive_rate_train": round(
                float(y_train.mean()),
                4,
            ),
            "positive_rate_test": round(
                float(y_test.mean()),
                4,
            ),
            **metrics,
        }

        metrics_path.write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        mlflow.log_params(
            {
                "model": "RandomForestClassifier",
                "n_estimators": 400,
                "class_weight": "balanced",
                "min_samples_leaf": 2,
                "random_state": RANDOM_STATE,
                "threshold": metrics["threshold"],
            }
        )

        mlflow.log_metrics(
            {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "train_positive_rate": float(
                    y_train.mean()
                ),
                "test_positive_rate": float(
                    y_test.mean()
                ),
            }
        )

        input_example = X_train.iloc[:5].copy()

        mlflow.sklearn.log_model(
            pipeline,
            name="predictive-maintenance-model",
            input_example=input_example,
            serialization_format="cloudpickle",
        )

        mlflow.log_artifact(
            str(metrics_path)
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