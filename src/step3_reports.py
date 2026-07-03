# ==============================================================================
# PIPELINE STEP 3: GENERATING MARKDOWN REPORTS
# ==============================================================================
# This script loads execution statistics, data profiles, and split outputs
# to construct the 6 curriculum deliverables.
# ==============================================================================

import sys
import pandas as pd
from pathlib import Path

# Ensure proper module resolution
sys.path.append(str(Path(__file__).parent))

from config import REPORTS_DIR, RESULTS_DIR, MODELS_DIR
from run_pipeline import write_markdown_reports
from data_utils import read_json

def main():
    print("="*60)
    print("PIPELINE STEP 3: GENERATING MARKDOWN REPORTS")
    print("="*60)

    # --------------------------------------------------------------------------
    # SUBSTEP 3.1: Deliverables Verification
    # --------------------------------------------------------------------------
    # Check that metadata and evaluation summaries are available on disk.
    metadata_path = REPORTS_DIR / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Run metadata not found at {metadata_path}. Run step1_preprocess.py first.")

    metrics_path = RESULTS_DIR / "metrics_summary.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics summary not found at {metrics_path}. Run step2_train.py first.")

    # --------------------------------------------------------------------------
    # SUBSTEP 3.2: Record Parser
    # --------------------------------------------------------------------------
    # Parse the metadata dictionaries and model metric lists.
    metadata = read_json(metadata_path)
    all_metrics = pd.read_csv(metrics_path).to_dict(orient="records")

    # Read final model selection
    metadata_model_path = MODELS_DIR / "final_model_metadata.json"
    if metadata_model_path.exists():
        best_model_name = read_json(metadata_model_path)["best_model"]
    else:
        best_model_name = "unknown"

    # --------------------------------------------------------------------------
    # SUBSTEP 3.3: Write Markdown Reports
    # --------------------------------------------------------------------------
    # Invoke markdown generator function to build all 6 report deliverables:
    # 1. Dataset Analysis Report
    # 2. Temporal Split Report
    # 3. Leakage Prevention Report
    # 4. Comparative Evaluation Report
    # 5. Balancing Report
    # 6. Generalization Study Report
    write_markdown_reports(metadata, all_metrics, best_model_name)
    print("[OK] All reports generated successfully!")

if __name__ == "__main__":
    main()
