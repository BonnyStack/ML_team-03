from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_DIR = Path(r"C:\Users\alber\Downloads\archive")
INPUT_CSV = DATA_DIR / "olist_merged_dataset.csv"
OUTPUT_DIR = DATA_DIR / "olist_visualizations"
SAMPLE_SIZE = 10_000

USECOLS = [
    "order_id",
    "price",
    "freight_value",
    "order_status",
    "order_purchase_timestamp",
    "is_delayed",
    "delivery_delay_days",
    "customer_state",
    "payment_value_total",
    "payment_type_main",
    "review_score_mean",
    "product_category_name_english",
]

DATE_COLUMNS = ["order_purchase_timestamp"]


def log(message: str) -> None:
    print(message, flush=True)


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_CSV}")

    log(f"Loading dataset from: {INPUT_CSV}")
    df = pd.read_csv(
        INPUT_CSV,
        usecols=USECOLS,
        parse_dates=DATE_COLUMNS,
        low_memory=False,
    )
    log(f"Loaded {len(df):,} rows and {len(df.columns):,} columns")
    return df


def save_table(df: pd.DataFrame, filename: str) -> None:
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    log(f"Saved table: {path}")


def finish_plot(filename: str) -> None:
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    log(f"Saved plot: {path}")


def plot_order_status(df: pd.DataFrame) -> None:
    order_status = (
        df.groupby("order_status")["order_id"]
        .nunique()
        .sort_values(ascending=False)
        .rename("order_count")
        .reset_index()
    )
    save_table(order_status, "order_status_counts.csv")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=order_status, x="order_status", y="order_count", color="steelblue")
    plt.title("Order Count by Status")
    plt.xlabel("Order Status")
    plt.ylabel("Unique Orders")
    plt.xticks(rotation=25, ha="right")
    finish_plot("order_status_counts.png")


def plot_payment_type(df: pd.DataFrame) -> None:
    payment_type = (
        df.groupby("payment_type_main")["order_id"]
        .nunique()
        .sort_values(ascending=False)
        .rename("order_count")
        .reset_index()
    )
    save_table(payment_type, "payment_type_counts.csv")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=payment_type, x="payment_type_main", y="order_count", palette="viridis")
    plt.title("Payment Type Distribution")
    plt.xlabel("Payment Type")
    plt.ylabel("Unique Orders")
    finish_plot("payment_type_distribution.png")


def plot_top_categories(df: pd.DataFrame) -> None:
    category_counts = (
        df.dropna(subset=["product_category_name_english"])
        .groupby("product_category_name_english")["order_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(15)
        .rename("order_count")
        .reset_index()
    )
    save_table(category_counts, "top_categories.csv")

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=category_counts,
        x="order_count",
        y="product_category_name_english",
        palette="mako",
    )
    plt.title("Top 15 Product Categories by Unique Orders")
    plt.xlabel("Unique Orders")
    plt.ylabel("Category")
    finish_plot("top_categories.png")


def plot_monthly_orders(df: pd.DataFrame) -> None:
    monthly_orders = (
        df.dropna(subset=["order_purchase_timestamp"])
        .assign(order_month=lambda x: x["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp())
        .groupby("order_month")["order_id"]
        .nunique()
        .rename("order_count")
        .reset_index()
    )
    save_table(monthly_orders, "monthly_orders.csv")

    plt.figure(figsize=(12, 5))
    sns.lineplot(data=monthly_orders, x="order_month", y="order_count", marker="o", color="coral")
    plt.title("Monthly Unique Orders")
    plt.xlabel("Month")
    plt.ylabel("Unique Orders")
    plt.xticks(rotation=45, ha="right")
    finish_plot("monthly_orders_lineplot.png")


def plot_review_histogram(df: pd.DataFrame) -> None:
    review_scores = df["review_score_mean"].dropna()

    plt.figure(figsize=(8, 5))
    sns.histplot(review_scores, bins=10, kde=True, color="slateblue")
    plt.title("Review Score Distribution")
    plt.xlabel("Review Score Mean")
    plt.ylabel("Frequency")
    finish_plot("review_score_histplot.png")


def plot_delay_boxplot(df: pd.DataFrame) -> None:
    delivered = df[df["order_status"] == "delivered"].copy()
    delivered = delivered.dropna(subset=["delivery_delay_days"])

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=delivered, x="is_delayed", y="delivery_delay_days", palette="Set2")
    plt.title("Delivery Delay Days by Delay Flag")
    plt.xlabel("Is Delayed")
    plt.ylabel("Delivery Delay Days")
    finish_plot("delivery_delay_boxplot.png")


def plot_customer_state_sales(df: pd.DataFrame) -> None:
    state_sales = (
        df.dropna(subset=["customer_state"])
        .groupby("customer_state")["payment_value_total"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .rename("total_payment_value")
        .reset_index()
    )
    save_table(state_sales, "customer_state_sales.csv")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=state_sales, x="customer_state", y="total_payment_value", palette="crest")
    plt.title("Top 15 Customer States by Total Payment Value")
    plt.xlabel("Customer State")
    plt.ylabel("Total Payment Value")
    finish_plot("customer_state_sales.png")


def plot_price_vs_freight(df: pd.DataFrame) -> None:
    scatter_df = (
        df.dropna(subset=["price", "freight_value", "is_delayed"])
        .sample(n=min(SAMPLE_SIZE, len(df.dropna(subset=["price", "freight_value", "is_delayed"]))), random_state=42)
        .copy()
    )

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=scatter_df,
        x="price",
        y="freight_value",
        hue="is_delayed",
        alpha=0.6,
        s=35,
        palette="Set1",
    )
    plt.title("Price vs Freight Value")
    plt.xlabel("Price")
    plt.ylabel("Freight Value")
    finish_plot("price_vs_freight_scatter.png")


def plot_state_category_heatmap(df: pd.DataFrame) -> None:
    base = df.dropna(subset=["customer_state", "product_category_name_english"]).copy()

    top_states = (
        base.groupby("customer_state")["order_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
        .index
    )
    top_categories = (
        base.groupby("product_category_name_english")["order_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
        .index
    )

    heatmap_df = base[
        base["customer_state"].isin(top_states)
        & base["product_category_name_english"].isin(top_categories)
    ]

    table = pd.pivot_table(
        heatmap_df,
        index="customer_state",
        columns="product_category_name_english",
        values="order_id",
        aggfunc="nunique",
        fill_value=0,
    )
    table = table.loc[top_states, top_categories]

    save_table(
        table.reset_index(),
        "state_category_heatmap_table.csv",
    )

    plt.figure(figsize=(14, 6))
    sns.heatmap(table, annot=True, fmt="g", cmap="YlGnBu", linewidths=0.5)
    plt.title("Unique Orders: Top States vs Top Categories")
    plt.xlabel("Product Category")
    plt.ylabel("Customer State")
    finish_plot("state_category_heatmap.png")


def main() -> None:
    sns.set_theme(style="whitegrid")
    ensure_output_dir()
    df = load_data()

    plot_order_status(df)
    plot_payment_type(df)
    plot_top_categories(df)
    plot_monthly_orders(df)
    plot_review_histogram(df)
    plot_delay_boxplot(df)
    plot_customer_state_sales(df)
    plot_price_vs_freight(df)
    plot_state_category_heatmap(df)

    log(f"Visualization output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
