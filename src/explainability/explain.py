from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


MODEL_PATH = Path("artifacts/model.joblib")


def explain_prediction(row: pd.DataFrame) -> list[dict]:
    """
    Return SHAP feature contributions for the positive
    'machine failure' class.
    """

    pipeline = joblib.load(MODEL_PATH)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    transformed_row = preprocessor.transform(row)

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer(transformed_row)

    # For sklearn RandomForestClassifier SHAP returns
    # one output per class. We want class 1: machine failure.
    if shap_values.values.ndim == 3:
        values = shap_values.values[0, :, 1]
    else:
        values = shap_values.values[0]

    explanation = []

    for feature, value in zip(feature_names, values):
        clean_feature = (
            str(feature)
            .replace("num__", "")
            .replace("cat__", "")
        )

        explanation.append(
            {
                "feature": clean_feature,
                "shap_value": round(float(value), 6),
            }
        )

    explanation.sort(
    key=lambda item: abs(item["shap_value"]),
    reverse=True,
    )

    return explanation[:5]
