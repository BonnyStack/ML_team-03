"""Baseline and upgraded modeling for classification and actual-delivery regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
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
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils import resample

from config import ACTUAL_DELIVERY_TARGET, CLASSIFICATION_TARGET, LOG_TRANSFORM_COLS
from preprocessing import run_preprocessing


def prepare_linear_features(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Split features and target, then build the preprocessing transformer for linear models."""
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
    log_numeric_features = [col for col in numeric_features if col in LOG_TRANSFORM_COLS]
    standard_numeric_features = [col for col in numeric_features if col not in log_numeric_features]

    log_numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # log1p reduces skew for long-tailed positive variables such as freight, weight, and distance.
            ("log", FunctionTransformer(np.log1p, validate=False, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )
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
            ("log_numeric", log_numeric_transformer, log_numeric_features),
            ("numeric", numeric_transformer, standard_numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )
    return X, y, preprocessor


def prepare_tree_features(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Split features and target, then build a dense preprocessing transformer for tree-based models."""
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )
    return X, y, preprocessor


def oversample_minority_class(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Simple random oversampling for the minority class using only the training split."""
    train_df = X.copy()
    train_df["_target"] = y.values

    class_counts = train_df["_target"].value_counts()
    if class_counts.empty or class_counts.shape[0] < 2:
        return X, y

    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    majority_df = train_df[train_df["_target"] == majority_class]
    minority_df = train_df[train_df["_target"] == minority_class]

    oversampled_minority_df = resample(
        minority_df,
        replace=True,
        n_samples=len(majority_df),
        random_state=42,
    )
    balanced_df = pd.concat([majority_df, oversampled_minority_df], axis=0)
    balanced_df = balanced_df.sample(frac=1.0, random_state=42)

    y_balanced = balanced_df.pop("_target")
    return balanced_df, y_balanced


def _print_feature_summary(X: pd.DataFrame) -> None:
    """Print the feature groups used for the current model."""
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
    log_features = [col for col in numeric_features if col in LOG_TRANSFORM_COLS]
    standard_features = [col for col in numeric_features if col not in log_features]
    print(f"Log-transformed numeric features ({len(log_features)}): {log_features}")
    print(f"Standard numeric features ({len(standard_features)}): {standard_features}")
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


def _find_best_threshold(y_true: pd.Series, y_proba: np.ndarray) -> tuple[float, float]:
    """Tune the classification threshold on a validation set by maximizing F1."""
    best_threshold = 0.50
    best_f1 = -1.0
    for threshold in np.arange(0.10, 0.91, 0.05):
        preds = (y_proba >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def _print_classification_metrics(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Print and return the main classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    print(f"\n{model_name}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    return metrics


def _print_regression_metrics(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Print and return the main regression metrics."""
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    print(f"\n{model_name}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE : {metrics['mae']:.4f}")
    print(f"R^2 : {metrics['r2']:.4f}")
    return metrics


def _print_metric_comparison(title: str, baseline: dict[str, float], alternative: dict[str, float]) -> None:
    """Print a concise baseline-vs-alternative comparison."""
    print(f"\n{title}")
    for metric_name in baseline:
        delta = alternative[metric_name] - baseline[metric_name]
        print(
            f"- {metric_name}: baseline={baseline[metric_name]:.4f}, "
            f"alternative={alternative[metric_name]:.4f}, delta={delta:+.4f}"
        )


def run_classification(df_clf: pd.DataFrame) -> None:
    """Train and compare classification models for is_delayed."""
    print("\n" + "=" * 80)
    print("TASK A: CLASSIFICATION - is_delayed")
    print("=" * 80)
    print("Upgrade process:")
    print("- Baseline: Logistic Regression")
    print("- Upgrade 1: log transforms for skewed numeric predictors")
    print("- Upgrade 2: random oversampling on the training split only")
    print("- Upgrade 3: threshold tuning on a validation split")
    print("- Comparison model: Random Forest classifier")

    X_linear, y, linear_preprocessor = prepare_linear_features(df_clf, CLASSIFICATION_TARGET)
    _print_feature_summary(X_linear)
    X_tree, _, tree_preprocessor = prepare_tree_features(df_clf, CLASSIFICATION_TARGET)

    X_train_full_linear, X_test_linear, y_train_full, y_test = train_test_split(
        X_linear,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    X_train_linear, X_val_linear, y_train, y_val = train_test_split(
        X_train_full_linear,
        y_train_full,
        test_size=0.25,
        stratify=y_train_full,
        random_state=42,
    )

    X_train_full_tree = X_tree.loc[X_train_full_linear.index]
    X_test_tree = X_tree.loc[X_test_linear.index]
    X_train_tree = X_tree.loc[X_train_linear.index]
    X_val_tree = X_tree.loc[X_val_linear.index]

    print(f"Training class distribution before oversampling: {y_train.value_counts().to_dict()}")
    X_train_linear_balanced, y_train_balanced = oversample_minority_class(X_train_linear, y_train)
    X_train_tree_balanced = X_tree.loc[X_train_linear_balanced.index].copy()
    print(f"Training class distribution after oversampling: {y_train_balanced.value_counts().to_dict()}")

    baseline_model = LogisticRegression(max_iter=1000)
    baseline_pipeline = Pipeline(
        steps=[
            ("preprocessor", linear_preprocessor),
            ("model", baseline_model),
        ]
    )
    baseline_pipeline.fit(X_train_linear_balanced, y_train_balanced)
    baseline_val_proba = baseline_pipeline.predict_proba(X_val_linear)[:, 1]
    baseline_threshold, baseline_val_f1 = _find_best_threshold(y_val, baseline_val_proba)
    print(f"Baseline validation-tuned threshold: {baseline_threshold:.2f} (validation F1={baseline_val_f1:.4f})")

    baseline_test_proba = baseline_pipeline.predict_proba(X_test_linear)[:, 1]
    baseline_test_pred = (baseline_test_proba >= baseline_threshold).astype(int)
    baseline_metrics = _print_classification_metrics(
        "Baseline model: LogisticRegression with log transforms + oversampling + threshold tuning",
        y_test,
        baseline_test_pred,
        baseline_test_proba,
        baseline_threshold,
    )
    print("\nClassification report:")
    print(classification_report(y_test, baseline_test_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, baseline_test_pred))

    alternative_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight=None,
        n_jobs=1,
        random_state=42,
    )
    alternative_pipeline = Pipeline(
        steps=[
            ("preprocessor", tree_preprocessor),
            ("model", alternative_model),
        ]
    )
    alternative_pipeline.fit(X_train_tree_balanced, y_train_balanced)
    alternative_val_proba = alternative_pipeline.predict_proba(X_val_tree)[:, 1]
    alternative_threshold, alternative_val_f1 = _find_best_threshold(y_val, alternative_val_proba)
    print(
        f"Random Forest validation-tuned threshold: {alternative_threshold:.2f} "
        f"(validation F1={alternative_val_f1:.4f})"
    )

    alternative_test_proba = alternative_pipeline.predict_proba(X_test_tree)[:, 1]
    alternative_test_pred = (alternative_test_proba >= alternative_threshold).astype(int)
    alternative_metrics = _print_classification_metrics(
        "Comparison model: RandomForestClassifier with oversampling + threshold tuning",
        y_test,
        alternative_test_pred,
        alternative_test_proba,
        alternative_threshold,
    )
    _print_metric_comparison("Classification comparison", baseline_metrics, alternative_metrics)


def run_actual_delivery_regression(df_actual: pd.DataFrame) -> None:
    """Train and compare regression models for actual delivery time."""
    print("\n" + "=" * 80)
    print("TASK B: REGRESSION - actual_delivery_days")
    print("=" * 80)
    print("Upgrade process:")
    print("- Baseline: Ridge regression")
    print("- Upgrade 1: log transforms for skewed numeric predictors")
    print("- Comparison model: GradientBoostingRegressor")

    X_linear, y, linear_preprocessor = prepare_linear_features(df_actual, ACTUAL_DELIVERY_TARGET)
    _print_feature_summary(X_linear)
    X_tree, _, tree_preprocessor = prepare_tree_features(df_actual, ACTUAL_DELIVERY_TARGET)

    X_train_linear, X_test_linear, y_train, y_test = train_test_split(
        X_linear,
        y,
        test_size=0.2,
        random_state=42,
    )
    X_train_tree = X_tree.loc[X_train_linear.index]
    X_test_tree = X_tree.loc[X_test_linear.index]

    baseline_model = Ridge()
    baseline_pipeline = Pipeline(
        steps=[
            ("preprocessor", linear_preprocessor),
            ("model", baseline_model),
        ]
    )
    baseline_pipeline.fit(X_train_linear, y_train)
    baseline_pred = baseline_pipeline.predict(X_test_linear)
    baseline_metrics = _print_regression_metrics(
        "Baseline model: Ridge regression with log-transformed skewed predictors",
        y_test,
        baseline_pred,
    )
    _print_top_coefficients(baseline_pipeline)

    alternative_model = GradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=3,
        n_estimators=100,
        min_samples_leaf=20,
        random_state=42,
    )
    alternative_pipeline = Pipeline(
        steps=[
            ("preprocessor", tree_preprocessor),
            ("model", alternative_model),
        ]
    )
    alternative_pipeline.fit(X_train_tree, y_train)
    alternative_pred = alternative_pipeline.predict(X_test_tree)
    alternative_metrics = _print_regression_metrics(
        "Comparison model: GradientBoostingRegressor",
        y_test,
        alternative_pred,
    )
    _print_metric_comparison("Regression comparison", baseline_metrics, alternative_metrics)


def main() -> None:
    """Run the two final modeling tasks."""
    df_clf, df_actual = run_preprocessing()
    run_classification(df_clf)
    run_actual_delivery_regression(df_actual)
    print("\nConclusion:")
    print("- We are still predicting both is_delayed and actual_delivery_days.")
    print("- The upgraded models test whether imbalance handling and nonlinear patterns improve results.")


if __name__ == "__main__":
    main()
