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
KS_STATISTIC_THRESHOLD = 0.08


def _validate_monitoring_frame(
    frame: pd.DataFrame,
    frame_name: str,
) -> None:
    missing_features = [
        feature
        for feature in NUMERIC_FEATURES
        if feature not in frame.columns
    ]

    if missing_features:
        raise ValueError(
            f"{frame_name} is missing required "
            f"features: {missing_features}"
        )

    if frame.empty:
        raise ValueError(
            f"{frame_name} must not be empty."
        )


def detect_numeric_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    alpha: float = ALPHA,
    ks_statistic_threshold: float = KS_STATISTIC_THRESHOLD,
) -> list[dict]:
    """
    Detect univariate numeric feature drift using the
    two-sample Kolmogorov-Smirnov test.

    Drift is flagged only when both:
      1. the p-value is below alpha, and
      2. the KS statistic exceeds a minimum effect-size threshold.

    This prevents very large samples from being flagged solely
    because of statistically significant but practically small
    distribution changes.
    """

    _validate_monitoring_frame(
        reference,
        "reference",
    )
    _validate_monitoring_frame(
        current,
        "current",
    )

    results = []

    for feature in NUMERIC_FEATURES:
        reference_values = (
            reference[feature]
            .dropna()
            .astype(float)
        )

        current_values = (
            current[feature]
            .dropna()
            .astype(float)
        )

        if reference_values.empty:
            raise ValueError(
                f"Reference feature '{feature}' "
                "contains no usable values."
            )

        if current_values.empty:
            raise ValueError(
                f"Current feature '{feature}' "
                "contains no usable values."
            )

        statistic, p_value = ks_2samp(
            reference_values,
            current_values,
            alternative="two-sided",
            method="auto",
        )

        statistically_significant = bool(
            p_value < alpha
        )

        practically_significant = bool(
            statistic >= ks_statistic_threshold
        )

        drift_detected = bool(
            statistically_significant
            and practically_significant
        )

        reference_mean = float(
            reference_values.mean()
        )

        current_mean = float(
            current_values.mean()
        )

        reference_std = float(
            reference_values.std(ddof=0)
        )

        current_std = float(
            current_values.std(ddof=0)
        )

        mean_shift = (
            current_mean - reference_mean
        )

        results.append(
            {
                "feature": feature,
                "reference_count": len(reference_values),
                "current_count": len(current_values),
                "ks_statistic": round(
                    float(statistic),
                    4,
                ),
                "p_value": round(
                    float(p_value),
                    6,
                ),
                "statistically_significant": (
                    statistically_significant
                ),
                "practically_significant": (
                    practically_significant
                ),
                "drift_detected": (
                    drift_detected
                ),
                "reference_mean": round(
                    reference_mean,
                    4,
                ),
                "current_mean": round(
                    current_mean,
                    4,
                ),
                "mean_shift": round(
                    mean_shift,
                    4,
                ),
                "reference_std": round(
                    reference_std,
                    4,
                ),
                "current_std": round(
                    current_std,
                    4,
                ),
            }
        )

    return results


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict:
    feature_results = detect_numeric_drift(
        reference,
        current,
    )

    drifted_features = [
        item["feature"]
        for item in feature_results
        if item["drift_detected"]
    ]

    drift_fraction = (
        len(drifted_features)
        / len(NUMERIC_FEATURES)
    )

    return {
        "method": "two_sample_ks",
        "alpha": ALPHA,
        "ks_statistic_threshold": (
            KS_STATISTIC_THRESHOLD
        ),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "drift_detected": bool(
            drifted_features
        ),
        "drifted_feature_count": len(drifted_features),
        "monitored_feature_count": len(NUMERIC_FEATURES),
        "drift_fraction": round(
            float(drift_fraction),
            4,
        ),
        "drifted_features": (
            drifted_features
        ),
        "features": feature_results,
    }


def create_simulated_production_data(
    source: pd.DataFrame,
    sample_size: int = 1500,
    random_state: int = 123,
) -> pd.DataFrame:
    """
    Create a deterministic synthetic production batch.

    This is used only for demonstration/testing.
    Production monitoring should call build_drift_report()
    with real observed production data instead.
    """

    if len(source) < sample_size:
        raise ValueError(
            "Source dataset does not contain "
            f"enough rows for sample_size={sample_size}."
        )

    current = source.sample(
        n=sample_size,
        random_state=random_state,
    ).copy()

    rng = np.random.default_rng(
        random_state
    )

    current["Torque"] = (
        current["Torque"].astype(float)
        + rng.normal(
            loc=8.0,
            scale=2.0,
            size=len(current),
        )
    )

    current["Tool wear"] = (
        current["Tool wear"].astype(float)
        + rng.normal(
            loc=20.0,
            scale=5.0,
            size=len(current),
        )
    ).clip(lower=0)

    return current


def load_reference_and_demo_current(
    data_path: Path = DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build deterministic reference and demo-current datasets.

    Both batches are sampled from the same source distribution
    before controlled synthetic drift is introduced. This makes
    the demo suitable for verifying that the detector identifies
    the intentionally shifted features rather than artifacts from
    dataset ordering.
    """

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}"
        )

    df = pd.read_csv(data_path)

    _validate_monitoring_frame(
        df,
        "dataset",
    )

    shuffled = df.sample(
        frac=1.0,
        random_state=123,
    ).reset_index(drop=True)

    reference = (
        shuffled.iloc[:7000]
        .copy()
    )

    current_source = (
        shuffled.iloc[7000:]
        .copy()
    )

    current = create_simulated_production_data(
        current_source,
        sample_size=1500,
        random_state=123,
    )

    return reference, current


def run_drift_analysis(
    reference: pd.DataFrame | None = None,
    current: pd.DataFrame | None = None,
    output_path: Path = OUTPUT_PATH,
) -> dict:
    """
    Run drift analysis.

    When reference/current dataframes are supplied, they are used
    directly. When omitted, a deterministic demo scenario is used.
    """

    if (
        reference is None
        and current is None
    ):
        reference, current = (
            load_reference_and_demo_current()
        )

    elif (
        reference is None
        or current is None
    ):
        raise ValueError(
            "reference and current must either "
            "both be provided or both be omitted."
        )

    report = build_drift_report(
        reference,
        current,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
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