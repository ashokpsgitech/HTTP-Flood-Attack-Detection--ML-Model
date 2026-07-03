import sys
import time
import joblib
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import (
    MODELS_DIR,
    RESULTS_DIR,
    PROJECT_ROOT,
    CLASS_LABELS,
    WRITE_FINAL_PREDICTIONS,
)
from run_pipeline import build_models, evaluate_model, write_predictions
from data_utils import write_json, predict_with_timing

def main():
    print("="*60)
    print("PIPELINE STEP 2: MODEL TRAINING & EVALUATION")
    print("="*60)

    cache_path = PROJECT_ROOT / "cache" / "preprocessed_data.joblib"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cached preprocessing data not found at {cache_path}. Run step1_preprocess.py first.")

    cache = joblib.load(cache_path)
    x_train = cache["x_train"]
    y_train = cache["y_train"]
    x_validation = cache["x_val"]
    y_validation = cache["y_val"]
    x_test = cache["x_test"]
    y_test = cache["y_test"]
    x_external = cache["x_ext"]
    y_external = cache["y_ext"]
    feature_columns = cache["feature_columns"]
    dropped_training_only_features = cache["dropped_training_only_features"]

    # 1. Build models
    models = build_models(feature_columns)

    # 2. Train and evaluate
    validation_scores = {}
    all_metrics = []
    reports_dfs = []
    per_class_dfs = []

    for model_name, model in models.items():
        print("\n" + "-"*60)
        print(f"Training {model_name}...")
        print("-"*60)
        
        start_time = time.perf_counter()
        model.fit(x_train, y_train)
        training_time = time.perf_counter() - start_time
        print(f"[OK] Training completed in {training_time:.2f} seconds")

        # Save model
        model_path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        print(f"[OK] Model saved to: {model_path}")

        # Evaluate on splits
        for split_name, x_split, y_split in [
            ("validation", x_validation, y_validation),
            ("internal_test", x_test, y_test),
            ("external", x_external, y_external),
        ]:
            print(f"  Evaluating on {split_name} split ({len(x_split)} samples)...")
            metrics, report_df, conf_df, per_class_df, _, _ = evaluate_model(
                model_name, model, split_name, x_split, y_split
            )
            metrics["training_time_seconds"] = training_time
            all_metrics.append(metrics)
            reports_dfs.append(report_df)
            per_class_dfs.append(per_class_df)
            print(f"  [OK] {split_name} evaluation completed")

            # Output validation score for selection
            if split_name == "validation":
                validation_scores[model_name] = metrics["f1_macro"]

            print("\n" + "="*50)
            print(f"MODEL: {model_name} | SPLIT: {split_name}")
            print("="*50)
            for k, v in metrics.items():
                if k not in ["model", "split", "experiment"]:
                    val_format = f"{v:.6f}" if isinstance(v, float) else str(v)
                    # Convert key to human readable
                    key_str = k.replace("_", " ").title().replace("Roc", "ROC").replace("Pr", "PR").replace("Fps", "FPS").replace("Fpr", "FPR").replace("Fnr", "FNR")
                    unit = " ms/sample" if "Latency" in key_str else " seconds" if "Time" in key_str else ""
                    print(f"* {key_str}:{val_format.rjust(30 - len(key_str))}{unit}")
            print("="*50)

    # 3. Save evaluation results
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    print(f"\n[OK] Metrics summary saved to: {RESULTS_DIR / 'metrics_summary.csv'}")

    reports_df = pd.concat(reports_dfs, ignore_index=True)
    reports_df.to_csv(RESULTS_DIR / "classification_reports.csv", index=False)
    print(f"[OK] Classification reports saved to: {RESULTS_DIR / 'classification_reports.csv'}")
    
    per_class_df = pd.concat(per_class_dfs, ignore_index=True)
    per_class_df.to_csv(RESULTS_DIR / "per_class_tp_tn_fp_fn.csv", index=False)
    print(f"[OK] Per-class metrics saved to: {RESULTS_DIR / 'per_class_tp_tn_fp_fn.csv'}")

    # 4. Save best model
    print("\n" + "="*60)
    print("Selecting and saving best model based on validation macro F1")
    print("="*60)
    best_model_name = max(validation_scores, key=validation_scores.get)
    print("Validation scores:")
    for model, score in validation_scores.items():
        marker = "<-- SELECTED" if model == best_model_name else ""
        print(f"  {model}: {score:.6f} {marker}")
    
    joblib.dump(models[best_model_name], MODELS_DIR / "final_model.joblib")
    print(f"[OK] Best model ({best_model_name}) saved to: {MODELS_DIR / 'final_model.joblib'}")
    
    write_json(
        MODELS_DIR / "final_model_metadata.json",
        {
            "best_model": best_model_name,
            "selection_metric": "validation_f1_macro",
            "validation_f1_macro": validation_scores[best_model_name],
            "feature_columns": feature_columns,
            "dropped_training_only_features": dropped_training_only_features,
            "class_labels": CLASS_LABELS,
        },
    )
    print(f"[OK] Model metadata saved to: {MODELS_DIR / 'final_model_metadata.json'}")

    # 5. Predictions
    if WRITE_FINAL_PREDICTIONS:
        print("\n" + "="*60)
        print("Generating final predictions")
        print("="*60)
        best_model = models[best_model_name]
        for split_name, x_split, y_split in [
            ("validation", x_validation, y_validation),
            ("internal_test", x_test, y_test),
            ("external", x_external, y_external),
        ]:
            print(f"  Generating predictions for {split_name} split...")
            y_pred, y_proba, _ = predict_with_timing(best_model, x_split)
            write_predictions(best_model_name, split_name, y_split, y_pred, y_proba)
            print(f"  [OK] Predictions saved to: {RESULTS_DIR / f'predictions_{best_model_name}_{split_name}.csv'}")

if __name__ == "__main__":
    main()
