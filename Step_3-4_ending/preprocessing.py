"""Data cleaning and feature engineering for the step_3 pipeline."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from config import (
    ACTUAL_DELIVERY_TARGET,
    CLASSIFICATION_TARGET,
    DATA_PATH,
    DATE_COLS,
    DROP_FEATURE_COLS,
    ID_COLS,
    KEY_NUMERIC_FEATURES,
    LEAKAGE_COLS,
    PACKAGE_DIMENSION_COLS,
    RAW_GEO_COLS,
    REGRESSION_TARGET,
)
from utils import (
    fix_decimal_columns,
    haversine_distance,
    iqr_outlier_summary,
    load_data,
    low_cardinality_unique_values,
    missing_summary,
    parse_dates,
    question_mark_summary,
)


def _available_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Return the subset of requested columns that exist in the DataFrame."""
    return [col for col in columns if col in df.columns]


def _print_drop_summary(reason: str, columns: list[str]) -> None:
    """Print a clear summary of dropped columns for a given reason."""
    if columns:
        print(f"{reason} ({len(columns)} columns): {columns}")
    else:
        print(f"{reason}: none")


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-friendly features from timestamps, dimensions, and geolocation fields."""
    featured_df = df.copy()

    featured_df[ACTUAL_DELIVERY_TARGET] = (
        featured_df["order_delivered_customer_date"] - featured_df["order_purchase_timestamp"]
    ).dt.days
    featured_df["promised_days"] = (
        featured_df["order_estimated_delivery_date"] - featured_df["order_purchase_timestamp"]
    ).dt.days
    featured_df["seller_prep_days"] = (
        featured_df["shipping_limit_date"] - featured_df["order_purchase_timestamp"]
    ).dt.days
    featured_df["purchase_hour"] = featured_df["order_purchase_timestamp"].dt.hour
    featured_df["purchase_dayofweek"] = featured_df["order_purchase_timestamp"].dt.dayofweek
    featured_df["is_weekend"] = np.where(featured_df["purchase_dayofweek"] >= 5, 1, 0)
    featured_df["purchase_month"] = featured_df["order_purchase_timestamp"].dt.month
    featured_df["distance_km"] = haversine_distance(
        featured_df["customer_geolocation_lat_mean"],
        featured_df["customer_geolocation_lng_mean"],
        featured_df["seller_geolocation_lat_mean"],
        featured_df["seller_geolocation_lng_mean"],
    )
    featured_df["same_state"] = np.where(
        featured_df["customer_state"].eq(featured_df["seller_state"]),
        1,
        0,
    )
    featured_df["product_volume_cm3"] = (
        featured_df["product_length_cm"] * featured_df["product_height_cm"] * featured_df["product_width_cm"]
    )

    return featured_df


def run_preprocessing() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean, engineer, and split the data for classification and actual-delivery regression."""
    print("=" * 80)
    print("STEP 3 PREPROCESSING")
    print("=" * 80)
    print(f"Loading raw data from: {DATA_PATH}")

    df = load_data(DATA_PATH)
    print(f"Raw shape: {df.shape}")

    print("\nChecking string columns for suspicious special tokens before cleaning.")
    token_summary = question_mark_summary(df)
    if token_summary.empty:
        print("No '?' placeholder tokens detected.")
    else:
        print(token_summary.to_string(index=False))

    print("\nReplacing '?' tokens with NaN where present.")
    df = df.replace("?", np.nan)

    print("Fixing decimal columns that use commas as decimal separators.")
    df = fix_decimal_columns(df)

    date_cols_to_parse = DATE_COLS + ["order_delivered_customer_date"]
    print(f"Parsing configured date columns: {date_cols_to_parse}")
    df = parse_dates(df, date_cols_to_parse)

    print("\nEngineering new features before leakage removal.")
    df = _engineer_features(df)

    id_cols_to_drop = _available_columns(df, ID_COLS)
    _print_drop_summary("Dropping identifier columns because they are not predictive features", id_cols_to_drop)
    df = df.drop(columns=id_cols_to_drop)

    datetime_cols = _available_columns(df, DATE_COLS)
    remaining_datetime_cols = list(df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)
    datetime_cols_to_drop = list(dict.fromkeys(datetime_cols + remaining_datetime_cols))
    _print_drop_summary(
        "Dropping raw datetime columns because engineered features replace them for modeling",
        datetime_cols_to_drop,
    )
    df = df.drop(columns=datetime_cols_to_drop)

    feature_cols_to_drop = _available_columns(df, DROP_FEATURE_COLS)
    _print_drop_summary(
        "Dropping low-value or redundant columns before EDA and modeling",
        feature_cols_to_drop,
    )
    df = df.drop(columns=feature_cols_to_drop)

    raw_geo_cols_to_drop = _available_columns(df, RAW_GEO_COLS)
    _print_drop_summary(
        "Dropping raw geolocation coordinates because distance_km already summarizes seller-buyer distance",
        raw_geo_cols_to_drop,
    )
    df = df.drop(columns=raw_geo_cols_to_drop)

    dimension_cols_to_drop = _available_columns(df, PACKAGE_DIMENSION_COLS)
    _print_drop_summary(
        "Dropping individual package dimensions because product_volume_cm3 combines them into one size feature",
        dimension_cols_to_drop,
    )
    df = df.drop(columns=dimension_cols_to_drop)

    print("\nMissing value summary:")
    print(missing_summary(df).to_string(index=False))

    print("\nLow-cardinality text columns and their unique values:")
    unique_summary = low_cardinality_unique_values(df)
    if unique_summary.empty:
        print("No low-cardinality string columns to display.")
    else:
        print(unique_summary.to_string(index=False))

    print("\nIQR outlier summary for key numeric features:")
    outlier_summary = iqr_outlier_summary(df, KEY_NUMERIC_FEATURES)
    if outlier_summary.empty:
        print("No IQR outliers detected in the selected numeric features.")
    else:
        print(outlier_summary.to_string(index=False))

    clf_drop_cols = [
        col for col in LEAKAGE_COLS if col in df.columns and col != CLASSIFICATION_TARGET
    ]
    if ACTUAL_DELIVERY_TARGET in df.columns:
        clf_drop_cols.append(ACTUAL_DELIVERY_TARGET)
    clf_drop_cols = list(dict.fromkeys(clf_drop_cols))
    _print_drop_summary(
        "Classification dataset: dropping leakage columns recorded after delivery or derived from the regression targets",
        clf_drop_cols,
    )
    df_clf = df.drop(columns=clf_drop_cols).copy()
    before_clf = len(df_clf)
    df_clf = df_clf.dropna(subset=[CLASSIFICATION_TARGET]).copy()
    print(
        f"Classification dataset: dropped {before_clf - len(df_clf)} rows with missing "
        f"'{CLASSIFICATION_TARGET}'. Final shape: {df_clf.shape}"
    )

    actual_drop_cols = [
        col for col in LEAKAGE_COLS if col in df.columns and col != ACTUAL_DELIVERY_TARGET
    ]
    _print_drop_summary(
        "Actual-delivery regression dataset: dropping leakage columns and derived post-delivery targets",
        actual_drop_cols,
    )
    df_actual = df.drop(columns=actual_drop_cols).copy()
    before_actual = len(df_actual)
    df_actual = df_actual.dropna(subset=[ACTUAL_DELIVERY_TARGET]).copy()
    print(
        f"Actual-delivery regression dataset: dropped {before_actual - len(df_actual)} rows with missing "
        f"'{ACTUAL_DELIVERY_TARGET}'. Final shape: {df_actual.shape}"
    )

    if REGRESSION_TARGET in df.columns:
        print(
            f"\nNote: '{REGRESSION_TARGET}' remains in the raw dataset only as a derived field and is excluded "
            "from both final modeling tasks."
        )

    return df_clf, df_actual


if __name__ == "__main__":
    classification_df, actual_delivery_df = run_preprocessing()
    print("\nReturned datasets:")
    print(f"Classification shape: {classification_df.shape}")
    print(f"Actual delivery regression shape: {actual_delivery_df.shape}")
