import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import CLASS_LABELS, FEATURE_ALIASES, ID_TO_LABEL, LABEL_ALIASES, LABEL_COLUMN, TIMESTAMP_COLUMN


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_label(label: str) -> str:
    normalized = " ".join(str(label).strip().split()).lower()
    return LABEL_ALIASES.get(normalized, str(label).strip())


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    rename_map = {}
    for column in df.columns:
        canonical = FEATURE_ALIASES.get(column, column)
        if canonical != column and canonical not in df.columns and canonical not in rename_map.values():
            rename_map[column] = canonical
    return df.rename(columns=rename_map)


def load_dataset(path: Path, allowed_labels=None, max_rows=None, require_timestamp=False) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=max_rows)
    df = normalize_columns(df)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing required label column: {LABEL_COLUMN}")
    if require_timestamp and TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(f"Missing required timestamp column: {TIMESTAMP_COLUMN}")

    df[LABEL_COLUMN] = df[LABEL_COLUMN].map(normalize_label)
    if allowed_labels is not None:
        df = df[df[LABEL_COLUMN].isin(allowed_labels)].copy()

    if TIMESTAMP_COLUMN in df.columns:
        df[TIMESTAMP_COLUMN] = pd.to_datetime(
            df[TIMESTAMP_COLUMN],
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )
        if df[TIMESTAMP_COLUMN].isna().any():
            bad_rows = int(df[TIMESTAMP_COLUMN].isna().sum())
            raise ValueError(f"{path.name} has {bad_rows} invalid timestamps")
        df = df.sort_values(TIMESTAMP_COLUMN, kind="mergesort")

    return df.reset_index(drop=True)


def dataset_profile(path: Path, df: pd.DataFrame) -> dict:
    timestamp_min = str(df[TIMESTAMP_COLUMN].min()) if TIMESTAMP_COLUMN in df.columns and len(df) else None
    timestamp_max = str(df[TIMESTAMP_COLUMN].max()) if TIMESTAMP_COLUMN in df.columns and len(df) else None
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "rows_after_filtering": int(len(df)),
        "columns": int(len(df.columns)),
        "timestamp_min": timestamp_min,
        "timestamp_max": timestamp_max,
        "labels": {str(k): int(v) for k, v in df[LABEL_COLUMN].value_counts().to_dict().items()},
        "exact_duplicate_rows_after_filtering": int(df.duplicated().sum()),
    }


def temporal_split(df: pd.DataFrame, train_ratio: float, validation_ratio: float):
    train_dfs = []
    val_dfs = []
    test_dfs = []

    for label, group in df.groupby(LABEL_COLUMN, sort=False):
        if TIMESTAMP_COLUMN in group.columns:
            group = group.sort_values(TIMESTAMP_COLUMN, kind="mergesort")
        
        n_rows = len(group)
        train_end = int(n_rows * train_ratio)
        validation_end = int(n_rows * (train_ratio + validation_ratio))
        
        train_dfs.append(group.iloc[:train_end])
        val_dfs.append(group.iloc[train_end:validation_end])
        test_dfs.append(group.iloc[validation_end:])
        
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    validation_df = pd.concat(val_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)
    
    if TIMESTAMP_COLUMN in train_df.columns:
        train_df = train_df.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
    if TIMESTAMP_COLUMN in validation_df.columns:
        validation_df = validation_df.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
    if TIMESTAMP_COLUMN in test_df.columns:
        test_df = test_df.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
        
    return train_df, validation_df, test_df



def split_profile(name: str, df: pd.DataFrame) -> dict:
    timestamp_min = str(df[TIMESTAMP_COLUMN].min()) if TIMESTAMP_COLUMN in df.columns and len(df) else None
    timestamp_max = str(df[TIMESTAMP_COLUMN].max()) if TIMESTAMP_COLUMN in df.columns and len(df) else None
    return {
        "name": name,
        "rows": int(len(df)),
        "timestamp_min": timestamp_min,
        "timestamp_max": timestamp_max,
        "labels": {str(k): int(v) for k, v in df[LABEL_COLUMN].value_counts().to_dict().items()},
    }


def select_compatible_feature_columns(train_df: pd.DataFrame, external_df: pd.DataFrame) -> list[str]:
    excluded = {LABEL_COLUMN, TIMESTAMP_COLUMN}
    train_features = [column for column in train_df.columns if column not in excluded]
    external_features = set(column for column in external_df.columns if column not in excluded)
    return [column for column in train_features if column in external_features]


def prepare_xy(df: pd.DataFrame, feature_columns=None):
    y = df[LABEL_COLUMN].map(CLASS_LABELS)
    if y.isna().any():
        unknown = sorted(df.loc[y.isna(), LABEL_COLUMN].unique())
        raise ValueError(f"Unknown labels found: {unknown}")

    if feature_columns is None:
        feature_columns = [
            column
            for column in df.columns
            if column not in {LABEL_COLUMN, TIMESTAMP_COLUMN}
        ]

    missing = sorted(set(feature_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required training features: {missing}")

    x = df[feature_columns].copy()
    for column in x.columns:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    return x, y.astype(int), list(feature_columns)


def remove_training_duplicates(train_df: pd.DataFrame) -> pd.DataFrame:
    return train_df.drop_duplicates().reset_index(drop=True)


def remove_rows_seen_in_training(candidate_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    common_columns = [column for column in candidate_df.columns if column in train_df.columns]
    if not common_columns:
        return candidate_df.reset_index(drop=True)
    train_rows = set(pd.util.hash_pandas_object(train_df[common_columns], index=False).astype("uint64").tolist())
    candidate_hash = pd.util.hash_pandas_object(candidate_df[common_columns], index=False).astype("uint64")
    return candidate_df.loc[~candidate_hash.isin(train_rows)].reset_index(drop=True)


def predict_with_timing(model, x):
    start = time.perf_counter()
    y_pred = model.predict(x)
    elapsed = time.perf_counter() - start
    y_proba = None
    if hasattr(model, "predict_proba"):
        raw_proba = model.predict_proba(x)
        estimator = model.steps[-1][1] if hasattr(model, "steps") else model
        model_classes = getattr(estimator, "classes_", list(ID_TO_LABEL))
        y_proba = np.zeros((len(x), len(ID_TO_LABEL)), dtype=float)
        for column_idx, class_id in enumerate(model_classes):
            y_proba[:, int(class_id)] = raw_proba[:, column_idx]
    return y_pred, y_proba, elapsed


def safe_multiclass_auc(y_true, y_proba, average):
    if y_proba is None:
        return np.nan
    present_classes = sorted(pd.Series(y_true).unique())
    if len(present_classes) < 2:
        return np.nan
    try:
        n_classes = y_proba.shape[1] if y_proba.ndim > 1 else 2
        if n_classes == 2:
            # Binary case: use positive class probability only
            pos_proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            return roc_auc_score(y_true, pos_proba)
        else:
            return roc_auc_score(y_true, y_proba, labels=list(ID_TO_LABEL), multi_class="ovr", average=average)
    except ValueError:
        return np.nan


def safe_multiclass_pr_auc(y_true, y_proba, average):
    if y_proba is None:
        return np.nan
    present_classes = sorted(pd.Series(y_true).unique())
    if len(present_classes) < 2:
        return np.nan
    y_true_matrix = np.zeros((len(y_true), len(present_classes)), dtype=int)
    y_proba_present = y_proba[:, present_classes]
    class_to_column = {class_id: idx for idx, class_id in enumerate(present_classes)}
    for row_idx, class_id in enumerate(y_true):
        y_true_matrix[row_idx, class_to_column[int(class_id)]] = 1
    try:
        return average_precision_score(y_true_matrix, y_proba_present, average=average)
    except ValueError:
        return np.nan


def per_class_confusion(y_true, y_pred) -> pd.DataFrame:
    labels = list(ID_TO_LABEL)
    matrices = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for class_id, matrix in zip(labels, matrices):
        tn, fp, fn, tp = matrix.ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        rows.append(
            {
                "class_id": class_id,
                "label": ID_TO_LABEL[class_id],
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fpr": fpr,
                "fnr": fnr,
            }
        )
    return pd.DataFrame(rows)


def evaluate_predictions(model_name, split_name, y_true, y_pred, y_proba, elapsed_seconds) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    labels = list(ID_TO_LABEL)
    metrics = {
        "model": model_name,
        "split": split_name,
        "samples": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "roc_auc_macro_ovr": safe_multiclass_auc(y_true, y_proba, "macro"),
        "pr_auc_macro": safe_multiclass_pr_auc(y_true, y_proba, "macro"),
        "inference_time_seconds": elapsed_seconds,
        "inference_latency_ms_per_sample": (elapsed_seconds / len(y_true) * 1000) if len(y_true) else np.nan,
    }

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[ID_TO_LABEL[class_id] for class_id in labels],
        zero_division=0,
        output_dict=True,
    )
    report_rows = []
    for label, values in report_dict.items():
        if isinstance(values, dict):
            row = {"model": model_name, "split": split_name, "label": label}
            row.update(values)
            report_rows.append(row)
    report_df = pd.DataFrame(report_rows)

    conf_df = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=[f"actual_{ID_TO_LABEL[class_id]}" for class_id in labels],
        columns=[f"predicted_{ID_TO_LABEL[class_id]}" for class_id in labels],
    )
    return metrics, report_df, conf_df


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
