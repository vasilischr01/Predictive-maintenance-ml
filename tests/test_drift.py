from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    NUMERIC_FEATURES,
    build_drift_report,
    create_simulated_production_data,
    detect_numeric_drift,
    run_drift_analysis,
)


def make_monitoring_frame(
    rows: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        random_state
    )

    return pd.DataFrame(
        {
            "Air temperature": rng.normal(
                300,
                2,
                rows,
            ),
            "Process temperature": rng.normal(
                310,
                2,
                rows,
            ),
            "Rotational speed": rng.normal(
                1500,
                150,
                rows,
            ),
            "Torque": rng.normal(
                40,
                10,
                rows,
            ),
            "Tool wear": rng.uniform(
                0,
                250,
                rows,
            ),
        }
    )


def test_identical_distributions_have_no_drift():
    reference = make_monitoring_frame()

    current = reference.copy()

    report = build_drift_report(
        reference,
        current,
    )

    assert report[
        "drift_detected"
    ] is False

    assert report[
        "drifted_features"
    ] == []


def test_controlled_torque_shift_is_detected():
    reference = make_monitoring_frame()

    current = reference.copy()

    current["Torque"] = (
        current["Torque"] + 15
    )

    results = detect_numeric_drift(
        reference,
        current,
    )

    torque_result = next(
        item
        for item in results
        if item["feature"] == "Torque"
    )

    assert torque_result[
        "drift_detected"
    ] is True

    assert torque_result[
        "statistically_significant"
    ] is True

    assert torque_result[
        "practically_significant"
    ] is True


def test_missing_feature_is_rejected():
    reference = make_monitoring_frame()

    current = (
        make_monitoring_frame()
        .drop(columns=["Torque"])
    )

    with pytest.raises(
        ValueError,
        match="missing required features",
    ):
        detect_numeric_drift(
            reference,
            current,
        )


def test_empty_frame_is_rejected():
    reference = make_monitoring_frame()

    current = pd.DataFrame(
        columns=NUMERIC_FEATURES
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        detect_numeric_drift(
            reference,
            current,
        )


def test_run_drift_analysis_writes_report(
    tmp_path,
):
    reference = make_monitoring_frame()

    current = reference.copy()

    output_path = (
        tmp_path / "drift_report.json"
    )

    report = run_drift_analysis(
        reference=reference,
        current=current,
        output_path=output_path,
    )

    assert output_path.exists()

    assert report[
        "reference_rows"
    ] == 1000

    assert report[
        "current_rows"
    ] == 1000