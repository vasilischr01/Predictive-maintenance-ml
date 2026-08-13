from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


DATA_PATH = Path("data/raw/ai4i2020.csv")
OUTPUT_PATH = Path("artifacts/drift_report.json")

NUMERIC_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

ALPHA = 0.05


def detect_numeric_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> list[dict]:
    results = []

    for feature in NUMERIC_FEATURES:
        reference_values = reference[feature].dropna()
        current_values = current[feature].dropna()

        statistic, p_value = ks_2samp(
            reference_values,
            current_values,
        )

        drift_detected = bool(p_value < ALPHA)

        results.append(
            {
                "feature": feature,
                "ks_statistic": round(float(statistic), 4),
                "p_value": round(float(p_value), 6),
                "drift_detected": drift_detected,
                "reference_mean": round(
                    float(reference_values.mean()),
                    4,
                ),
                "current_mean": round(
                    float(current_values.mean()),
                    4,
                ),
            }
        )

    return results


def create_simulated_production_data(
    reference: pd.DataFrame,
) -> pd.DataFrame:
    current = reference.sample(
        n=2000,
        random_state=123,
    ).copy()

    rng = np.random.default_rng(123)

    current["Torque"] = (
        current["Torque"]
        + rng.normal(
            loc=8.0,
            scale=2.0,
            size=len(current),
        )
    )

    current["Tool wear"] = (
        current["Tool wear"]
        + rng.normal(
            loc=20.0,
            scale=5.0,
            size=len(current),
        )
    ).clip(lower=0)

    return current


def run_drift_analysis() -> dict:
    df = pd.read_csv(DATA_PATH)

    reference = df.iloc[:8000].copy()

    current = create_simulated_production_data(
        df.iloc[8000:].copy()
    )

    feature_results = detect_numeric_drift(
        reference,
        current,
    )

    drifted_features = [
        item["feature"]
        for item in feature_results
        if item["drift_detected"]
    ]

    report = {
        "alpha": ALPHA,
        "drift_detected": len(drifted_features) > 0,
        "drifted_features": drifted_features,
        "features": feature_results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":
    report = run_drift_analysis()

    print(
        json.dumps(
            report,
            indent=2,
        )
    )