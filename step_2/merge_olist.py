## SCRIPT TO MERGE THE SMALL DATASETS INTO ONE MASTER DATASET

from pathlib import Path
import pandas as pd
import numpy as np

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "archive" / "dataset_components"

OUTPUT_CSV = DATA_DIR / "olist_merged_dataset.csv"
OUTPUT_XLSX = DATA_DIR / "olist_merged_dataset.xlsx"

# =========================================================
# HELPERS
# =========================================================
def read_csv(name: str) -> pd.DataFrame:
    file_path = DATA_DIR / name
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
    return pd.read_csv(file_path)


def mode_or_first(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]


# =========================================================
# LOAD FILES
# =========================================================
orders = read_csv("olist_orders_dataset.csv")
order_items = read_csv("olist_order_items_dataset.csv")
payments = read_csv("olist_order_payments_dataset.csv")
reviews = read_csv("olist_order_reviews_dataset.csv")
customers = read_csv("olist_customers_dataset.csv")
products = read_csv("olist_products_dataset.csv")
sellers = read_csv("olist_sellers_dataset.csv")
geolocation = read_csv("olist_geolocation_dataset.csv")
category_translation = read_csv("product_category_name_translation.csv")


# =========================================================
# PARSE DATES
# =========================================================
date_cols_orders = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

for col in date_cols_orders:
    if col in orders.columns:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

if "review_creation_date" in reviews.columns:
    reviews["review_creation_date"] = pd.to_datetime(
        reviews["review_creation_date"], errors="coerce"
    )

if "review_answer_timestamp" in reviews.columns:
    reviews["review_answer_timestamp"] = pd.to_datetime(
        reviews["review_answer_timestamp"], errors="coerce"
    )


# =========================================================
# CREATE DELAY FLAG
# =========================================================
# 1 = delivered after estimated date
# 0 = delivered on/before estimated date OR missing one of the dates
orders["is_delayed"] = (
    (
        orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
        & (
            orders["order_delivered_customer_date"]
            > orders["order_estimated_delivery_date"]
        )
    )
    .astype(int)
)

# Optional: days late / early
orders["delivery_delay_days"] = (
    orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
).dt.days


# =========================================================
# AGGREGATE TABLES THAT CAN CAUSE DUPLICATES
# =========================================================

# ---- Payments: many rows per order possible
payments_agg = (
    payments.groupby("order_id", as_index=False)
    .agg(
        payment_sequential_max=("payment_sequential", "max"),
        payment_installments_max=("payment_installments", "max"),
        payment_value_total=("payment_value", "sum"),
        payment_value_mean=("payment_value", "mean"),
        payment_type_main=("payment_type", mode_or_first),
        payment_type_nunique=("payment_type", "nunique"),
        payment_records_count=("payment_type", "size"),
    )
)

# ---- Reviews: usually one per order, but aggregate anyway to be safe
reviews_agg = (
    reviews.groupby("order_id", as_index=False)
    .agg(
        review_score_mean=("review_score", "mean"),
        review_score_min=("review_score", "min"),
        review_score_max=("review_score", "max"),
        review_records_count=("review_id", "count"),
        review_creation_date_max=("review_creation_date", "max"),
        review_answer_timestamp_max=("review_answer_timestamp", "max"),
    )
)

# ---- Geolocation: many rows per zip prefix, compress to one row per zip
geo_agg = (
    geolocation.groupby("geolocation_zip_code_prefix", as_index=False)
    .agg(
        geolocation_lat_mean=("geolocation_lat", "mean"),
        geolocation_lng_mean=("geolocation_lng", "mean"),
        geolocation_city_main=("geolocation_city", mode_or_first),
        geolocation_state_main=("geolocation_state", mode_or_first),
    )
)


# =========================================================
# ENRICH PRODUCTS WITH ENGLISH CATEGORY
# =========================================================
products = products.merge(
    category_translation,
    how="left",
    on="product_category_name"
)


# =========================================================
# BUILD MASTER DATASET
# Grain: one row per order item
# =========================================================
df = order_items.merge(
    orders,
    how="left",
    on="order_id",
    validate="many_to_one"
)

df = df.merge(
    customers,
    how="left",
    on="customer_id",
    validate="many_to_one"
)

df = df.merge(
    payments_agg,
    how="left",
    on="order_id",
    validate="many_to_one"
)

df = df.merge(
    reviews_agg,
    how="left",
    on="order_id",
    validate="many_to_one"
)

df = df.merge(
    products,
    how="left",
    on="product_id",
    validate="many_to_one"
)

df = df.merge(
    sellers,
    how="left",
    on="seller_id",
    validate="many_to_one"
)

# =========================================================
# MERGE CUSTOMER GEOLOCATION
# =========================================================
df = df.merge(
    geo_agg.add_prefix("customer_"),
    how="left",
    left_on="customer_zip_code_prefix",
    right_on="customer_geolocation_zip_code_prefix",
    validate="many_to_one"
)

# =========================================================
# MERGE SELLER GEOLOCATION
# =========================================================
df = df.merge(
    geo_agg.add_prefix("seller_"),
    how="left",
    left_on="seller_zip_code_prefix",
    right_on="seller_geolocation_zip_code_prefix",
    validate="many_to_one"
)


# =========================================================
# OPTIONAL CLEANUP OF DUPLICATE KEY COLUMNS
# =========================================================
cols_to_drop = [
    "customer_geolocation_zip_code_prefix",
    "seller_geolocation_zip_code_prefix",
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])


# =========================================================
# SAVE
# =========================================================
df.to_csv(OUTPUT_CSV, index=False)
df.to_excel(OUTPUT_XLSX, index=False)

print("Merged dataset created successfully.")
print(f"CSV file:  {OUTPUT_CSV}")
print(f"XLSX file: {OUTPUT_XLSX}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")