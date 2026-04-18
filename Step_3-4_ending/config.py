"""Shared configuration for the step_3 pipeline."""

from pathlib import Path

DATA_PATH = Path("archive/olist_merged_dataset.csv")
OUTPUT_DIR = Path("step_3/eda_outputs")

CLASSIFICATION_TARGET = "is_delayed"
REGRESSION_TARGET = "delivery_delay_days"
ACTUAL_DELIVERY_TARGET = "actual_delivery_days"

ID_COLS = [
    "order_id",
    "order_item_id",
    "product_id",
    "seller_id",
    "customer_id",
    "customer_unique_id",
]

LEAKAGE_COLS = [
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "order_approved_at",
    "review_score_mean",
    "review_score_min",
    "review_score_max",
    "review_records_count",
    "review_creation_date_max",
    "review_answer_timestamp_max",
    "delivery_delay_days",
    "is_delayed",
]

DATE_COLS = [
    "shipping_limit_date",
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
]

DROP_FEATURE_COLS = [
    "customer_zip_code_prefix",
    "payment_sequential_max",
    "payment_installments_max",
    "payment_value_total",
    "payment_value_mean",
    "payment_records_count",
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "seller_zip_code_prefix",
]

RAW_GEO_COLS = [
    "customer_geolocation_lat_mean",
    "customer_geolocation_lng_mean",
    "seller_geolocation_lat_mean",
    "seller_geolocation_lng_mean",
]

PACKAGE_DIMENSION_COLS = [
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]

KEY_NUMERIC_FEATURES = [
    "distance_km",
    "freight_value",
    "product_volume_cm3",
    "product_weight_g",
    "promised_days",
]

LOG_TRANSFORM_COLS = [
    "price",
    "freight_value",
    "product_weight_g",
    "promised_days",
    "seller_prep_days",
    "distance_km",
    "product_volume_cm3",
]
