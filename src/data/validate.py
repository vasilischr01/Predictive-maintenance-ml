from __future__ import annotations

import pandas as pd


EXPECTED_COLUMNS = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Machine failure",
]

ALLOWED_TYPES = {"L", "M", "H"}


def validate_dataframe(df: pd.DataFrame) -> None:
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[EXPECTED_COLUMNS].isnull().any().any():
        raise ValueError("Missing values found in required columns.")

    unknown_types = set(df["Type"].unique()) - ALLOWED_TYPES
    if unknown_types:
        raise ValueError(f"Unexpected machine Type values: {sorted(unknown_types)}")

    target_values = set(df["Machine failure"].unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"Target must be binary, found: {sorted(target_values)}")

    if df["Machine failure"].nunique() < 2:
        raise ValueError("Target contains only one class.")
