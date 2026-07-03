import sys
from pathlib import Path
import joblib

sys.path.append(str(Path(__file__).parent))

from config import (
    CLASS_LABELS,
    EXTERNAL_DATASET,
    MAX_ROWS,
    MODELS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    RESULTS_DIR,
    TRAIN_DATASET,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)
from data_utils import (
    dataset_profile,
    ensure_dirs,
    load_dataset,
    prepare_xy,
    remove_rows_seen_in_training,
    remove_training_duplicates,
    select_compatible_feature_columns,
    split_profile,
    temporal_split,
    write_json,
)

def main():
    print("="*60)
    print("PIPELINE STEP 1: PREPROCESSING & DATA SPLITTING")
    print("="*60)
    
    # 1. Ensure output directories exist
    ensure_dirs(REPORTS_DIR, RESULTS_DIR, MODELS_DIR, RESULTS_DIR / "confusion_matrices")
    print(f"[OK] Created directories: {REPORTS_DIR}, {RESULTS_DIR}, {MODELS_DIR}")

    # 2. Load datasets
    allowed_labels = set(CLASS_LABELS)
    print(f"Loading training dataset from: {TRAIN_DATASET}")
    train_df = load_dataset(TRAIN_DATASET, allowed_labels=allowed_labels, max_rows=MAX_ROWS, require_timestamp=True)
    print(f"[OK] Training dataset loaded: {len(train_df)} rows")
    
    print(f"Loading external dataset from: {EXTERNAL_DATASET}")
    external_df = load_dataset(EXTERNAL_DATASET, allowed_labels=allowed_labels, max_rows=MAX_ROWS)
    print(f"[OK] External dataset loaded: {len(external_df)} rows")

    # 3. Temporal Splitting (strict ordering)
    print(f"Split ratios: Train={TRAIN_SPLIT:.0%}, Validation={VALIDATION_SPLIT:.0%}, Test={1-TRAIN_SPLIT-VALIDATION_SPLIT:.0%}")
    train_split_df, validation_df, test_df = temporal_split(train_df, TRAIN_SPLIT, VALIDATION_SPLIT)
    print(f"[OK] Train split: {len(train_split_df)} rows")
    print(f"[OK] Validation split: {len(validation_df)} rows")
    print(f"[OK] Test split: {len(test_df)} rows")

    # 4. Leakage Prevention & Deduplication
    print("Removing exact duplicates from training set...")
    train_split_df = remove_training_duplicates(train_split_df)
    print(f"[OK] Training set after deduplication: {len(train_split_df)} rows")
    
    print("Removing training rows from validation set...")
    validation_df = remove_rows_seen_in_training(validation_df, train_split_df)
    print(f"[OK] Validation set after leakage removal: {len(validation_df)} rows")
    
    print("Removing training rows from test set...")
    test_df = remove_rows_seen_in_training(test_df, train_split_df)
    print(f"[OK] Test set after leakage removal: {len(test_df)} rows")
    
    print("Removing training rows from external set...")
    external_df = remove_rows_seen_in_training(external_df, train_split_df)
    print(f"[OK] External set after leakage removal: {len(external_df)} rows")

    # 5. Feature Selection
    feature_columns = select_compatible_feature_columns(train_split_df, external_df)
    if not feature_columns:
        raise ValueError("No compatible feature columns found.")
    print(f"[OK] Found {len(feature_columns)} compatible features")
    dropped_training_only_features = [
        col for col in train_split_df.columns
        if col not in set(feature_columns) | {"Label", "Timestamp"}
    ]
    if dropped_training_only_features:
        print(f"[OK] Dropped training-only features: {', '.join(dropped_training_only_features)}")

    # 6. Prepare Matrices
    x_train, y_train, feature_columns = prepare_xy(train_split_df, feature_columns)
    x_val, y_val, _ = prepare_xy(validation_df, feature_columns)
    x_test, y_test, _ = prepare_xy(test_df, feature_columns)
    x_ext, y_ext, _ = prepare_xy(external_df, feature_columns)

    # 7. Generate Run Metadata
    metadata = {
        "project_root": str(PROJECT_ROOT),
        "class_labels": CLASS_LABELS,
        "feature_columns": feature_columns,
        "dropped_training_only_features": dropped_training_only_features,
        "train_dataset": dataset_profile(TRAIN_DATASET, train_df),
        "external_dataset": dataset_profile(EXTERNAL_DATASET, external_df),
        "splits": [
            split_profile("train", train_split_df),
            split_profile("validation", validation_df),
            split_profile("internal_test", test_df),
            split_profile("external", external_df),
        ],
    }
    write_json(REPORTS_DIR / "run_metadata.json", metadata)
    print(f"[OK] Metadata written to: {REPORTS_DIR / 'run_metadata.json'}")

    # Save cache
    cache_dir = PROJECT_ROOT / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / "preprocessed_data.joblib"
    joblib.dump({
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
        "x_ext": x_ext, "y_ext": y_ext,
        "feature_columns": feature_columns,
        "dropped_training_only_features": dropped_training_only_features
    }, cache_path)
    print(f"[OK] Preprocessed splits cached to: {cache_path}")

if __name__ == "__main__":
    main()
