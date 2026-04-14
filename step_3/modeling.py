"""Baseline modeling for classification and actual-delivery regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import ACTUAL_DELIVERY_TARGET, CLASSIFICATION_TARGET
from preprocessing import run_preprocessing


def prepare_features(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Split features and target, then build the shared preprocessing transformer."""
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    return X, y, preprocessor


def _print_feature_summary(X: pd.DataFrame) -> None:
    """Print the feature groups used for the current model."""
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
    print(f"Numeric features used ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features used ({len(categorical_features)}): {categorical_features}")


def _print_top_coefficients(pipeline: Pipeline, top_n: int = 10) -> None:
    """Print the top coefficients by absolute value for a fitted linear model."""
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_
    coef_summary = (
        pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
        .assign(abs_coefficient=lambda frame: frame["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .head(top_n)
    )

    print(f"\nTop {top_n} most important coefficients by absolute value:")
    print(coef_summary[["feature", "coefficient"]].to_string(index=False))


def run_classification(df_clf: pd.DataFrame) -> None:
    """Train and evaluate the logistic-regression baseline for is_delayed."""
    print("\n" + "=" * 80)
    print("TASK A: CLASSIFICATION - is_delayed")
    print("=" * 80)

    X, y, preprocessor = prepare_features(df_clf, CLASSIFICATION_TARGET)
    _print_feature_summary(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("Model used: LogisticRegression with class_weight='balanced'")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


def run_actual_delivery_regression(df_actual: pd.DataFrame) -> None:
    """Train and evaluate the ridge-regression baseline for actual delivery time."""
    print("\n" + "=" * 80)
    print("TASK B: REGRESSION - actual_delivery_days")
    print("=" * 80)

    X, y, preprocessor = prepare_features(df_actual, ACTUAL_DELIVERY_TARGET)
    _print_feature_summary(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = Ridge()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model used: Ridge regression")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R^2 : {r2:.4f}")
    _print_top_coefficients(pipeline)


def main() -> None:
    """Run the two final modeling tasks."""
    df_clf, df_actual = run_preprocessing()
    run_classification(df_clf)
    run_actual_delivery_regression(df_actual)
    print("\nSuggested next step models:")
    print("- Classification: Random Forest or XGBoost")
    print("- Regression: Gradient Boosting or XGBoost")


if __name__ == "__main__":
    main()
