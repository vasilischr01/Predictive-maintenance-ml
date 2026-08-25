from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

DATASET_ID = 601
OUTPUT_PATH = Path("data/raw/ai4i2020.csv")


def download_dataset(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = fetch_ucirepo(id=DATASET_ID)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    if "Machine failure" not in y.columns:
        raise RuntimeError(
            "Expected target 'Machine failure' was not found in the UCI dataset."
        )

    df = pd.concat([X, y[["Machine failure"]]], axis=1)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df):,} rows to {output_path}")
    return output_path


if __name__ == "__main__":
    download_dataset()
