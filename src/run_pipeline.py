import time
from pathlib import Path
import numpy as np

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from config import (
    CLASS_LABELS,
    EXTERNAL_DATASET,
    ID_TO_LABEL,
    MAX_ROWS,
    MODELS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    RESULTS_DIR,
    TEST_SPLIT,
    TRAIN_DATASET,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    WRITE_FINAL_PREDICTIONS,
)
from data_utils import (
    dataset_profile,
    ensure_dirs,
    evaluate_predictions,
    load_dataset,
    per_class_confusion,
    predict_with_timing,
    prepare_xy,
    remove_rows_seen_in_training,
    remove_training_duplicates,
    select_compatible_feature_columns,
    split_profile,
    temporal_split,
    write_json,
)





def build_models(feature_columns):
    numeric_preprocessor_scaled = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_preprocessor_tree = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
        ]
    )
    scaled_preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_preprocessor_scaled, feature_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    tree_preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_preprocessor_tree, feature_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return {
        "lightgbm": Pipeline(
            steps=[
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    lgb.LGBMClassifier(
                        learning_rate=0.08,
                        n_estimators=180,
                        max_depth=-1,
                        num_leaves=31,
                        reg_alpha=0.05,
                        reg_lambda=0.05,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    xgb.XGBClassifier(
                        learning_rate=0.08,
                        n_estimators=180,
                        max_depth=10,
                        reg_alpha=0.05,
                        reg_lambda=0.05,
                        scale_pos_weight=1,
                        n_jobs=-1,
                        random_state=42,
                        eval_metric='logloss',
                    ),
                ),
            ]
        ),
        "catboost": Pipeline(
            steps=[
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    CatBoostClassifier(
                        iterations=180,
                        learning_rate=0.08,
                        depth=6,
                        l2_leaf_reg=3,
                        random_seed=42,
                        verbose=0,
                        thread_count=-1
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model_name, model, split_name, x, y):
    y_pred, y_proba, inference_elapsed = predict_with_timing(model, x)
    metrics, report_df, conf_df = evaluate_predictions(
        model_name,
        split_name,
        y.to_numpy(),
        y_pred,
        y_proba,
        inference_elapsed,
    )
    per_class_df = per_class_confusion(y.to_numpy(), y_pred)
    per_class_df.insert(0, "split", split_name)
    per_class_df.insert(0, "model", model_name)
    return metrics, report_df, conf_df, per_class_df, y_pred, y_proba


def write_predictions(model_name, split_name, y_true, y_pred, y_proba):
    predictions = pd.DataFrame(
        {
            "true_class_id": y_true.to_numpy(),
            "true_label": [ID_TO_LABEL[int(class_id)] for class_id in y_true.to_numpy()],
            "predicted_class_id": y_pred,
            "predicted_label": [ID_TO_LABEL[int(class_id)] for class_id in y_pred],
        }
    )
    if y_proba is not None:
        for class_id, label in ID_TO_LABEL.items():
            predictions[f"probability_{class_id}_{label}"] = y_proba[:, class_id]
    predictions.to_csv(RESULTS_DIR / f"predictions_{model_name}_{split_name}.csv", index=False)


def write_markdown_reports(metadata, all_metrics, best_model_name):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_lines = [
        "# Dataset Analysis Report",
        "",
        "## Class Taxonomy",
        "",
        "| Class ID | Label |",
        "|---:|---|",
    ]
    for class_id, label in ID_TO_LABEL.items():
        dataset_lines.append(f"| {class_id} | {label} |")

    dataset_lines.extend(
        [
            "",
            "## Training Dataset",
            "",
            f"- Path: `{metadata['train_dataset']['path']}`",
            f"- SHA256: `{metadata['train_dataset']['sha256']}`",
            f"- Rows after filtering: {metadata['train_dataset']['rows_after_filtering']}",
            f"- Time range: {metadata['train_dataset']['timestamp_min']} to {metadata['train_dataset']['timestamp_max']}",
            f"- Exact duplicate rows after filtering: {metadata['train_dataset']['exact_duplicate_rows_after_filtering']}",
            "",
            "## External Dataset",
            "",
            f"- Path: `{metadata['external_dataset']['path']}`",
            f"- SHA256: `{metadata['external_dataset']['sha256']}`",
            f"- Rows after filtering: {metadata['external_dataset']['rows_after_filtering']}",
            f"- Time range: {metadata['external_dataset']['timestamp_min']} to {metadata['external_dataset']['timestamp_max']}",
            f"- Exact duplicate rows after filtering: {metadata['external_dataset']['exact_duplicate_rows_after_filtering']}",
            "",
            "## Feature Alignment",
            "",
            f"- Compatible model features: {len(metadata['feature_columns'])}",
            f"- Dropped training-only features: {', '.join(metadata['dropped_training_only_features']) or 'None'}",
            "",
            "The external dataset is filtered to labels present in the predefined class taxonomy. "
            "Rows outside the taxonomy are excluded before inference and are not used for training, tuning, or preprocessing. "
            "External feature names are normalized through the configured CICIDS alias map before schema alignment.",
        ]
    )
    (REPORTS_DIR / "dataset_analysis.md").write_text("\n".join(dataset_lines), encoding="utf-8")

    split_lines = [
        "# Temporal Splitting Documentation",
        "",
        "The training dataset is sorted by `Timestamp` using stable ordering. No random shuffle or stratified split is used.",
        "",
        f"- Training split: {TRAIN_SPLIT:.0%}",
        f"- Validation split: {VALIDATION_SPLIT:.0%}",
        f"- Internal test split: {TEST_SPLIT:.0%}",
        "",
        "| Split | Rows | Timestamp Min | Timestamp Max |",
        "|---|---:|---|---|",
    ]
    for split in metadata["splits"]:
        split_lines.append(
            f"| {split['name']} | {split['rows']} | {split['timestamp_min']} | {split['timestamp_max']} |"
        )
    split_lines.extend(
        [
            "",
            "## Split Label Distribution",
            "",
            "Because strict temporal order is preserved, some later splits may not contain every class. "
            "This is reported explicitly instead of using stratified sampling, which would violate the project guideline.",
            "",
        ]
    )
    for split in metadata["splits"]:
        split_lines.extend([f"### {split['name']}", "", "| Label | Count |", "|---|---:|"])
        for label, count in split["labels"].items():
            split_lines.append(f"| {label} | {count} |")
        split_lines.append("")
    (REPORTS_DIR / "temporal_split_report.md").write_text("\n".join(split_lines), encoding="utf-8")

    leakage_lines = [
        "# Leakage Prevention Documentation",
        "",
        "- Preprocessing is fitted only on the temporal training split.",
        "- Validation, internal test, and external datasets are transformed with the fitted preprocessing pipeline.",
        "- Feature selection through variance filtering is learned only from the training split.",
        "- Normalization statistics are learned only from the training split.",
        "- Training duplicates are removed before model fitting.",
        "- Rows duplicated from training are removed from validation, internal test, and external evaluation.",
        "- External dataset labels are filtered before inference according to the predefined taxonomy.",
        "- External data is not used for model selection, threshold tuning, balancing, or hyperparameter tuning.",
    ]
    (REPORTS_DIR / "leakage_prevention.md").write_text("\n".join(leakage_lines), encoding="utf-8")

    comparison = pd.DataFrame(all_metrics).sort_values(["split", "f1_macro"], ascending=[True, False])
    comparison.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    comparison_preview = comparison[
        [
            "model",
            "split",
            "samples",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "inference_latency_ms_per_sample",
            "training_time_seconds",
        ]
    ].copy()
    for column in [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "inference_latency_ms_per_sample",
        "training_time_seconds",
    ]:
        comparison_preview[column] = comparison_preview[column].map(lambda value: f"{value:.6f}")
    markdown_table = dataframe_to_markdown(comparison_preview)
    report_lines = [
        "# Comparative Evaluation Report",
        "",
        f"Selected final model: `{best_model_name}` based on validation macro F1.",
        "",
        markdown_table,
    ]
    (REPORTS_DIR / "comparative_evaluation.md").write_text("\n".join(report_lines), encoding="utf-8")

    balancing_lines = [
        "# Data Balancing Report",
        "",
        "The implementation uses cost-sensitive learning where supported instead of synthetic oversampling.",
        "",
        "- Logistic Regression uses `class_weight='balanced'`.",
        "- Random Forest uses `class_weight='balanced_subsample'`.",
        "- HistGradientBoosting is trained without synthetic resampling.",
        "",
        "SMOTE/ADASYN are not applied by default because the dataset is temporal and flow-based; generating synthetic "
        "minority flows before careful analysis could distort the traffic chronology and feature relationships.",
    ]
    (REPORTS_DIR / "balancing_report.md").write_text("\n".join(balancing_lines), encoding="utf-8")

    external_metrics = comparison[comparison["split"] == "external"].copy()
    generalization_lines = [
        "# Generalization Study Report",
        "",
        f"External dataset: `{metadata['external_dataset']['path']}`",
        f"External SHA256: `{metadata['external_dataset']['sha256']}`",
        "",
        "The external dataset is evaluated after normalizing labels and filtering to the predefined class taxonomy. "
        "Feature names are normalized into the training schema before inference.",
        "",
        dataframe_to_markdown(
            external_metrics[
                [
                    "model",
                    "samples",
                    "accuracy",
                    "precision_macro",
                    "recall_macro",
                    "f1_macro",
                    "inference_latency_ms_per_sample",
                ]
            ].assign(
                accuracy=lambda df: df["accuracy"].map(lambda value: f"{value:.6f}"),
                precision_macro=lambda df: df["precision_macro"].map(lambda value: f"{value:.6f}"),
                recall_macro=lambda df: df["recall_macro"].map(lambda value: f"{value:.6f}"),
                f1_macro=lambda df: df["f1_macro"].map(lambda value: f"{value:.6f}"),
                inference_latency_ms_per_sample=lambda df: df["inference_latency_ms_per_sample"].map(lambda value: f"{value:.6f}"),
            )
        ),
    ]
    (REPORTS_DIR / "generalization_study.md").write_text("\n".join(generalization_lines), encoding="utf-8")


def dataframe_to_markdown(df):
    columns = list(df.columns)
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def main():
    import step1_preprocess
    import step2_train
    import step3_reports

    step1_preprocess.main()
    step2_train.main()
    step3_reports.main()


if __name__ == "__main__":
    main()
