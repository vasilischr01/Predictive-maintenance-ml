from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import shap

MODEL_PATH = Path("artifacts/model.joblib")


@lru_cache(maxsize=1)
def load_explainability_components():
    """
    Load and cache the trained pipeline and SHAP explainer.

    The cache avoids reloading the serialized model and rebuilding
    the explainer for every inference request.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}"
        )

    pipeline = joblib.load(MODEL_PATH)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    explainer = shap.TreeExplainer(classifier)

    return pipeline, preprocessor, classifier, explainer


def explain_prediction(
    row: pd.DataFrame,
    top_k: int = 5,
) -> list[dict]:
    """
    Return the top SHAP feature contributions for one prediction.

    The input row is transformed using the same preprocessing
    pipeline used during model training.
    """

    (
        _pipeline,
        preprocessor,
        _classifier,
        explainer,
    ) = load_explainability_components()

    transformed_row = preprocessor.transform(row)

    feature_names = preprocessor.get_feature_names_out()

    shap_values = explainer(transformed_row)

    values = shap_values.values

    # HistGradientBoostingClassifier produces one SHAP value
    # per transformed feature for binary classification.
    if values.ndim == 2:
        feature_values = values[0]
    elif values.ndim == 1:
        feature_values = values
    else:
        raise ValueError(
            "Unexpected SHAP output shape: "
            f"{values.shape}"
        )

    explanation = []

    for feature, value in zip(
        feature_names,
        feature_values,
    ):
        clean_feature = (
            str(feature)
            .replace("num__", "")
            .replace("cat__", "")
        )

        explanation.append(
            {
                "feature": clean_feature,
                "shap_value": round(
                    float(value),
                    6,
                ),
            }
        )

    explanation.sort(
        key=lambda item: abs(
            item["shap_value"]
        ),
        reverse=True,
    )

    return explanation[:top_k]


def clear_explainer_cache() -> None:
    """
    Clear the cached model/explainer.

    Useful after replacing the production model artifact.
    """

    load_explainability_components.cache_clear()