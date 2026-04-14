# Step 3 Pipeline

This folder contains a complete Python pipeline for two prediction tasks on `archive/olist_merged_dataset.csv`:

- Classification: predict `is_delayed` (`0` = not delayed, `1` = delayed)
- Regression: predict `actual_delivery_days`, the total number of days from purchase to customer delivery

## Project Structure

```text
step_3/
|-- config.py
|-- utils.py
|-- preprocessing.py
|-- eda.py
|-- modeling.py
|-- eda_outputs/
`-- README.md
```

## How To Run

Run these commands from the repository root:

```bash
python step_3/preprocessing.py
python step_3/eda.py
python step_3/modeling.py
```

## Targets

`is_delayed` is a binary label showing whether an order arrived after the estimated date.

`actual_delivery_days` is calculated inside the Python preprocessing step as:

- `order_delivered_customer_date - order_purchase_timestamp`

It measures the full end-to-end delivery duration in days.

## Leakage

Target leakage happens when a model is trained with information that would not be available at prediction time. That makes evaluation look better than real-world performance.

The pipeline removes columns such as:

- Delivery timestamps recorded after shipping or final delivery
- Review metrics created after the customer received the order
- Derived delivery outcome columns such as `delivery_delay_days`
- `order_approved_at` as a conservative choice

For classification, both `delivery_delay_days` and `actual_delivery_days` are removed because they directly reveal delivery performance.

For actual-delivery regression, columns such as `order_delivered_customer_date`, `delivery_delay_days`, and `is_delayed` are not used as features because they directly reveal the outcome we are trying to predict.

## Engineered Features

The preprocessing step creates these model-ready features:

- `promised_days`: days between purchase and estimated delivery date
- `seller_prep_days`: days between purchase and shipping deadline
- `purchase_hour`: hour of day when the order was placed
- `purchase_dayofweek`: weekday of purchase (`0=Monday`, `6=Sunday`)
- `is_weekend`: whether the purchase happened on Saturday or Sunday
- `purchase_month`: calendar month of purchase
- `distance_km`: Haversine distance between seller and customer
- `same_state`: whether seller and customer are in the same state
- `product_volume_cm3`: package volume calculated from length x height x width

## Removed Variables Before Modeling

The final modeling datasets remove these additional fields because they are identifiers, redundant summaries, or raw inputs replaced by engineered features:

- ID-like columns including `customer_unique_id`
- ZIP code prefixes
- duplicated payment summary/count columns
- product text-length and photo-count fields
- raw latitude and longitude columns after `distance_km` is created
- individual package dimensions after `product_volume_cm3` is created

## Why `distance_km` Matters

Physical distance is a practical logistics signal. Longer routes often mean more handoffs, more transport time, and more chance of delay. Even when it is not the strongest predictor alone, it helps explain shipping cost and delivery timing together with `freight_value`, `promised_days`, and seller/customer location fields.

## Modeling Notes

- Classification uses Logistic Regression with `class_weight="balanced"` because `is_delayed` is imbalanced.
- Evaluation for classification focuses on precision, recall, F1, and ROC-AUC, not just accuracy.
- Regression uses Ridge as the stable baseline for `actual_delivery_days`.
- Natural next-step upgrades are Random Forest or XGBoost for classification, and Gradient Boosting or XGBoost for regression.

## Does The CSV Change?

No. The original `archive/olist_merged_dataset.csv` is not modified by this pipeline.

All cleaning, feature engineering, filtering, and modeling happen in Python at runtime on pandas DataFrames held in memory. The only files written by the pipeline are output files such as the EDA plots inside `step_3/eda_outputs`.
