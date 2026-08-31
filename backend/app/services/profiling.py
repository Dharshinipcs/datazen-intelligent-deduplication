from pathlib import Path

import pandas as pd


def load_dataset(file_path: Path, file_extension: str) -> pd.DataFrame:
    """Load a supported dataset into a pandas DataFrame."""
    extension = file_extension.lower()

    if extension == ".csv":
        return pd.read_csv(file_path)

    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    raise ValueError(f"Unsupported file extension: {file_extension}")


def profile_dataset(
    file_path: Path,
    file_extension: str,
) -> dict:
    """
    Generate domain-agnostic profiling information for a dataset.

    The profile contains dataset-level and column-level information
    useful for later schema detection, standardization, blocking,
    and entity matching.
    """
    df = load_dataset(file_path, file_extension)

    columns = []

    for column in df.columns:
        series = df[column]

        non_null = series.dropna()
        unique_count = series.nunique(dropna=True)
        null_count = int(series.isna().sum())
        row_count = len(df)

        columns.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "row_count": row_count,
                "null_count": null_count,
                "null_percentage": (
                    round((null_count / row_count) * 100, 2)
                    if row_count
                    else 0.0
                ),
                "unique_count": int(unique_count),
                "unique_percentage": (
                    round((unique_count / len(non_null)) * 100, 2)
                    if len(non_null)
                    else 0.0
                ),
                "sample_values": [
                    str(value)
                    for value in non_null.head(5).tolist()
                ],
            }
        )

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
    }
