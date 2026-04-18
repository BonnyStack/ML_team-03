"""Utility helpers shared across the step_3 pipeline."""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    """Load the source CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


def fix_decimal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string columns that use commas as decimal separators into floats."""
    fixed_df = df.copy()
    string_cols = fixed_df.select_dtypes(include=["object", "string"]).columns

    for col in string_cols:
        series = fixed_df[col].astype("string")
        non_null = series.dropna()
        if non_null.empty or not non_null.str.contains(",", regex=False).any():
            continue

        is_decimal_like = non_null.str.fullmatch(r"-?\d+(,\d+)?")
        if bool(is_decimal_like.all()):
            fixed_df[col] = pd.to_numeric(series.str.replace(",", ".", regex=False), errors="coerce")

    return fixed_df


def parse_dates(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    """Parse the provided columns into pandas datetimes."""
    parsed_df = df.copy()
    for col in date_cols:
        if col in parsed_df.columns:
            parsed_df[col] = pd.to_datetime(parsed_df[col], errors="coerce")
    return parsed_df


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted missing-value summary for all columns."""
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean() * 100).values,
        }
    )
    return summary.sort_values(["missing_count", "missing_pct"], ascending=False).reset_index(drop=True)


def question_mark_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Find columns that contain the exact '?' token as a placeholder value."""
    records: list[dict[str, object]] = []
    string_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in string_cols:
        series = df[col].astype("string")
        count = int(series.eq("?").sum())
        if count > 0:
            records.append({"column": col, "token": "?", "count": count})

    if not records:
        return pd.DataFrame(columns=["column", "token", "count"])

    return pd.DataFrame(records).sort_values(["count", "column"], ascending=[False, True]).reset_index(drop=True)


def low_cardinality_unique_values(
    df: pd.DataFrame,
    max_unique: int = 10,
    max_display: int = 10,
) -> pd.DataFrame:
    """Return unique values for low-cardinality string columns to spot odd categories quickly."""
    records: list[dict[str, object]] = []
    string_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in string_cols:
        unique_values = pd.Series(df[col].dropna().astype(str).unique())
        unique_count = int(unique_values.shape[0])
        if unique_count <= max_unique:
            sample_values = unique_values.sort_values().head(max_display).tolist()
            records.append(
                {
                    "column": col,
                    "unique_count": unique_count,
                    "sample_unique_values": sample_values,
                }
            )

    if not records:
        return pd.DataFrame(columns=["column", "unique_count", "sample_unique_values"])

    return pd.DataFrame(records).sort_values("column").reset_index(drop=True)


def iqr_outlier_summary(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Summarize numeric outliers using the IQR rule for the selected columns."""
    records: list[dict[str, object]] = []

    for col in columns:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0:
            records.append(
                {
                    "column": col,
                    "outlier_count": outlier_count,
                    "outlier_pct": round(outlier_count / len(series) * 100, 3),
                    "lower_bound": round(float(lower_bound), 3),
                    "upper_bound": round(float(upper_bound), 3),
                }
            )

    if not records:
        return pd.DataFrame(columns=["column", "outlier_count", "outlier_pct", "lower_bound", "upper_bound"])

    return pd.DataFrame(records).sort_values("outlier_count", ascending=False).reset_index(drop=True)


def haversine_distance(
    lat1: pd.Series | np.ndarray | float,
    lon1: pd.Series | np.ndarray | float,
    lat2: pd.Series | np.ndarray | float,
    lon2: pd.Series | np.ndarray | float,
) -> np.ndarray:
    """Compute the Haversine distance in kilometers between two points."""
    radius_km = 6371.0

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return radius_km * c


def plot_saver(fig: plt.Figure, filename: str, output_dir: Path) -> None:
    """Save a matplotlib figure as a PNG inside the requested output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
