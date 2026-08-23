from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.validate import validate_dataframe
from src.training.train import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    select_threshold,
    split_dataset,
)

OUTPUT_PATH = Path("artifacts/model_benchmark.json")


def build_preprocessor() -> ColumnTransformer:
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
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=400,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_at_threshold(
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


def run_benchmark(
    data_path: Path = DATA_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict:
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} does not exist."
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
    ) = split_dataset(X, y)

    models = build_models()

    results = {}

    for model_name, pipeline in models.items():
        print(f"\nTraining: {model_name}")

        pipeline.fit(X_train, y_train)

        validation_probabilities = (
            pipeline.predict_proba(X_val)[:, 1]
        )

        selected_threshold, _ = select_threshold(
            y_val,
            validation_probabilities,
        )

        validation_metrics = evaluate_at_threshold(
            y_val,
            validation_probabilities,
            selected_threshold,
        )

        results[model_name] = {
            "selected_threshold": selected_threshold,
            "validation_metrics": validation_metrics,
        }

        print(
            json.dumps(
                results[model_name],
                indent=2,
            )
        )

    winner = max(
        results,
        key=lambda model_name: (
            results[model_name]["validation_metrics"][
                "pr_auc"
            ],
            results[model_name]["validation_metrics"][
                "f1"
            ],
            results[model_name]["validation_metrics"][
                "recall"
            ],
        ),
    )

    print(f"\nSelected model: {winner}")

    winning_pipeline = models[winner]

    # Refit only on training data.
    # Validation remains part of model selection and is not merged
    # into training before the final test evaluation.
    winning_pipeline.fit(
        X_train,
        y_train,
    )

    test_probabilities = (
        winning_pipeline.predict_proba(X_test)[:, 1]
    )

    selected_threshold = results[winner][
        "selected_threshold"
    ]

    test_metrics = evaluate_at_threshold(
        y_test,
        test_probabilities,
        selected_threshold,
    )

    payload = {
        "selection_policy": {
            "primary_metric": "validation_pr_auc",
            "tie_breakers": [
                "validation_f1",
                "validation_recall",
            ],
            "threshold_selection_metric": "validation_f1",
            "test_set_used_for_selection": False,
        },
        "dataset": {
            "rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
        },
        "models": results,
        "selected_model": winner,
        "selected_threshold": selected_threshold,
        "final_test_metrics": test_metrics,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nFinal benchmark:")
    print(
        json.dumps(
            payload,
            indent=2,
        )
    )

    print(
        f"\nSaved benchmark to {output_path}"
    )

    return payload


if __name__ == "__main__":
    run_benchmark()