import pandas as pd
import pytest

from src.data.validate import validate_dataframe


def valid_df():
    return pd.DataFrame(
        {
            "Type": ["L", "H"],
            "Air temperature": [298.1, 301.0],
            "Process temperature": [308.6, 310.0],
            "Rotational speed": [1551, 1400],
            "Torque": [42.8, 55.0],
            "Tool wear": [0, 200],
            "Machine failure": [0, 1],
        }
    )


def test_valid_dataframe_passes():
    validate_dataframe(valid_df())


def test_missing_column_fails():
    df = valid_df().drop(columns=["Torque"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(df)


def test_unknown_type_fails():
    df = valid_df()
    df.loc[0, "Type"] = "X"

    with pytest.raises(ValueError, match="Unexpected machine Type"):
        validate_dataframe(df)
