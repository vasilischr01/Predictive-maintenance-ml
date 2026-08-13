from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.data.validate import validate_dataframe
from src.training.train import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
)


OUTPUT_PATH = Path("artifacts/threshold_analysis.json")


def analyse_thresholds() -> list[dict]:
    df = pd.read_csv(DATA_PATH)
    validate_dataframe(df)

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET].astype(int)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(X_test)[:, 1]

    results = []

    for threshold in np.arange(0.10, 0.91, 0.05):
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
        ).ravel()

        results.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": round(
                    float(
                        precision_score(
                            y_test,
                            predictions,
                            zero_division=0,
                        )
                    ),
                    4,
                ),
                "recall": round(
                    float(
                        recall_score(
                            y_test,
                            predictions,
                            zero_division=0,
                        )
                    ),
                    4,
                ),
                "f1": round(
                    float(
                        f1_score(
                            y_test,
                            predictions,
                            zero_division=0,
                        )
                    ),
                    4,
                ),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    return results


if __name__ == "__main__":
    results = analyse_thresholds()

    print(
        f"{'Threshold':<10}"
        f"{'Precision':<12}"
        f"{'Recall':<10}"
        f"{'F1':<10}"
        f"{'TP':<6}"
        f"{'FP':<6}"
        f"{'FN':<6}"
    )

    print("-" * 60)

    for row in results:
        print(
            f"{row['threshold']:<10.2f}"
            f"{row['precision']:<12.4f}"
            f"{row['recall']:<10.4f}"
            f"{row['f1']:<10.4f}"
            f"{row['tp']:<6}"
            f"{row['fp']:<6}"
            f"{row['fn']:<6}"
        )
        