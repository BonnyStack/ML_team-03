"""Exploratory data analysis for the step_3 pipeline."""

from __future__ import annotations

from io import StringIO
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import (
    ACTUAL_DELIVERY_TARGET,
    CLASSIFICATION_TARGET,
    DROP_FEATURE_COLS,
    ID_COLS,
    KEY_NUMERIC_FEATURES,
    LEAKAGE_COLS,
    OUTPUT_DIR,
    PACKAGE_DIMENSION_COLS,
    RAW_GEO_COLS,
)
from preprocessing import run_preprocessing
from utils import missing_summary, plot_saver


sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")


def _clear_previous_outputs() -> None:
    """Remove existing PNG files so the folder contains only the current targeted charts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for file_path in OUTPUT_DIR.glob("*.png"):
        file_path.unlink()


def _annotate_bar_percentages(ax: plt.Axes) -> None:
    """Annotate count bars with percentages based on the total count."""
    total = sum(patch.get_height() for patch in ax.patches)
    if total == 0:
        return

    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{(height / total) * 100:.1f}%",
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 5),
            textcoords="offset points",
        )


def _plot_missing_values(df: pd.DataFrame) -> None:
    """Plot the top 20 missing-value columns."""
    summary = missing_summary(df)
    summary = summary[summary["missing_count"] > 0].head(20)
    if summary.empty:
        print("No missing values found for the combined dataset plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=summary, x="missing_count", y="column", ax=ax, color="steelblue")
    ax.set_title("Top 20 Columns With Missing Values")
    ax.set_xlabel("Missing Count")
    ax.set_ylabel("Column")
    fig.tight_layout()
    plot_saver(fig, "missing_values_top20", OUTPUT_DIR)


def _plot_class_distribution(df_clf: pd.DataFrame) -> None:
    """Plot the target class distribution for is_delayed."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df_clf, x=CLASSIFICATION_TARGET, ax=ax, color="mediumseagreen")
    ax.set_title("Class Distribution of is_delayed")
    ax.set_xlabel("is_delayed")
    ax.set_ylabel("Count")
    _annotate_bar_percentages(ax)
    fig.tight_layout()
    plot_saver(fig, "classification_target_distribution", OUTPUT_DIR)


def _plot_numeric_boxplots(df_clf: pd.DataFrame) -> None:
    """Create the targeted classification boxplots for the selected numeric features."""
    for feature in [col for col in KEY_NUMERIC_FEATURES if col in df_clf.columns]:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df_clf, x=CLASSIFICATION_TARGET, y=feature, ax=ax)
        ax.set_title(f"{feature} vs is_delayed")
        ax.set_xlabel("is_delayed")
        ax.set_ylabel(feature)
        fig.tight_layout()
        plot_saver(fig, f"classification_boxplot_{feature}", OUTPUT_DIR)


def _plot_delay_rate_by_category(df_clf: pd.DataFrame, column: str, filename: str, title: str) -> None:
    """Plot average delay rate by a categorical column."""
    if column not in df_clf.columns:
        return

    top_categories = df_clf[column].value_counts(dropna=True).head(15).index
    summary = (
        df_clf[df_clf[column].isin(top_categories)]
        .groupby(column, dropna=False)[CLASSIFICATION_TARGET]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=summary, x=CLASSIFICATION_TARGET, y=column, ax=ax, color="slateblue")
    ax.set_title(title)
    ax.set_xlabel("Delay Rate")
    ax.set_ylabel(column)
    fig.tight_layout()
    plot_saver(fig, filename, OUTPUT_DIR)


def _plot_correlation_heatmap(df: pd.DataFrame, filename: str, title: str) -> None:
    """Plot an annotated correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    plot_saver(fig, filename, OUTPUT_DIR)


def _plot_regression_histogram(df_actual: pd.DataFrame) -> None:
    """Plot the actual-delivery target distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_actual[ACTUAL_DELIVERY_TARGET], bins=50, kde=True, ax=ax, color="teal")
    ax.set_title("Distribution of actual_delivery_days")
    ax.set_xlabel("actual_delivery_days")
    ax.set_ylabel("Count")
    fig.tight_layout()
    plot_saver(fig, "regression_target_histogram", OUTPUT_DIR)


def _plot_regression_scatterplots(df_actual: pd.DataFrame) -> None:
    """Plot selected numeric features against actual delivery time."""
    for feature in ["distance_km", "freight_value", "promised_days"]:
        if feature not in df_actual.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_actual, x=feature, y=ACTUAL_DELIVERY_TARGET, ax=ax, alpha=0.35)
        ax.set_title(f"{feature} vs actual_delivery_days")
        ax.set_xlabel(feature)
        ax.set_ylabel(ACTUAL_DELIVERY_TARGET)
        fig.tight_layout()
        plot_saver(fig, f"regression_scatter_{feature}", OUTPUT_DIR)


def _plot_pairplot(df_clf: pd.DataFrame) -> None:
    """Create a lightweight pair plot for the main numeric features."""
    pairplot_features = [col for col in KEY_NUMERIC_FEATURES if col in df_clf.columns]
    if len(pairplot_features) < 2:
        return

    sample_size = min(3000, len(df_clf))
    pairplot_df = df_clf[pairplot_features + [CLASSIFICATION_TARGET]].sample(n=sample_size, random_state=42)
    grid = sns.pairplot(pairplot_df, hue=CLASSIFICATION_TARGET, corner=True, diag_kind="hist")
    grid.fig.set_size_inches(14, 14)
    grid.fig.suptitle("Pair Plot of Key Numeric Features", y=1.02)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.savefig(OUTPUT_DIR / "pairplot_key_numeric_features.png", dpi=300, bbox_inches="tight")
    plt.close(grid.fig)


def _top_correlated_features(df: pd.DataFrame, target: str, top_n: int = 10) -> pd.Series:
    """Return the most correlated numeric features with the selected target."""
    numeric_df = df.select_dtypes(include="number")
    if target not in numeric_df.columns:
        return pd.Series(dtype=float)

    correlations = numeric_df.corr()[target].drop(labels=[target]).dropna()
    ordered_index = correlations.abs().sort_values(ascending=False).index
    return correlations.reindex(ordered_index).head(top_n)


def run_eda() -> None:
    """Run exploratory data analysis and save the focused plot set."""
    _clear_previous_outputs()
    df_clf, df_actual = run_preprocessing()
    combined_df = pd.concat(
        [df_clf.assign(dataset="classification"), df_actual.assign(dataset="actual_delivery_regression")],
        ignore_index=True,
        sort=False,
    )

    print("\n" + "=" * 80)
    print("BASIC INSPECTION")
    print("=" * 80)
    print(f"Classification shape: {df_clf.shape}")
    print(f"Actual-delivery regression shape: {df_actual.shape}")
    info_buffer = StringIO()
    df_clf.info(buf=info_buffer)
    print("\nInfo():")
    print(info_buffer.getvalue())
    print("\nDtypes:")
    print(df_clf.dtypes.to_string())
    print("\nHead(5) - classification dataset:")
    print(df_clf.head().to_string())
    print("\nDescribe - classification dataset:")
    print(df_clf.describe(include="all").transpose().to_string())

    print("\nGenerating missing-value plot.")
    _plot_missing_values(combined_df)

    print("Generating classification plots.")
    _plot_class_distribution(df_clf)
    _plot_numeric_boxplots(df_clf)
    _plot_delay_rate_by_category(
        df_clf,
        "customer_state",
        "classification_delay_rate_customer_state",
        "Delay Rate by customer_state (Top 15)",
    )
    _plot_delay_rate_by_category(
        df_clf,
        "product_category_name_english",
        "classification_delay_rate_product_category",
        "Delay Rate by product_category_name_english (Top 15)",
    )
    _plot_correlation_heatmap(df_clf, "classification_correlation_heatmap", "Classification Correlation Heatmap")

    print("Generating actual-delivery regression plots.")
    _plot_regression_histogram(df_actual)
    _plot_regression_scatterplots(df_actual)

    print("Generating pair plot.")
    _plot_pairplot(df_clf)

    clf_top = _top_correlated_features(df_clf, CLASSIFICATION_TARGET)
    actual_top = _top_correlated_features(df_actual, ACTUAL_DELIVERY_TARGET)

    print("\n" + "=" * 80)
    print("EDA SUMMARY")
    print("=" * 80)
    print("Top features recommended for classification:")
    print(clf_top.to_string() if not clf_top.empty else "No numeric correlation summary available.")
    print("\nTop features recommended for actual-delivery regression:")
    print(actual_top.to_string() if not actual_top.empty else "No numeric correlation summary available.")
    print("\nColumns dropped (IDs):")
    print(ID_COLS)
    print("\nColumns dropped as low-value or redundant:")
    print(DROP_FEATURE_COLS)
    print("\nRaw geolocation columns dropped after computing distance_km:")
    print(RAW_GEO_COLS)
    print("\nPackage dimension columns dropped after creating product_volume_cm3:")
    print(PACKAGE_DIMENSION_COLS)
    print("\nColumns dropped as leakage candidates:")
    print(LEAKAGE_COLS)
    print(f"\nPlots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_eda()
