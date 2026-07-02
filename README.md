# Binary HTTP Flood Attack Detection Pipeline

This project implements a robust, temporal-aware binary machine learning pipeline designed to detect HTTP-based flooding and Denial of Service (DoS) attacks using network flow features.

---

## 1. Project Architecture

The model is trained as a **binary classifier** separating legitimate web traffic from malicious application-layer attacks:
*   **Class 0 (Benign)**: Normal network traffic.
*   **Class 1 (Attack)**: Composed of volumetric HTTP floods (`DoS attacks-Hulk`, `DDOS attack-HOIC`) and other DoS vectors.

### Key Implementation Details
*   **Per-Class Temporal Splitting**: To avoid disjoint class distributions across splits (arising from chronological attack recordings), data splitting in [data_utils.py](file:///d:/HTTP%20flood%20attack/src/data_utils.py#L93-L125) is performed temporally *within* each class (70% train, 10% validation, 20% test) before merging, preserving chronological integrity without losing class representation.
*   **Zero Leakage**: All scaling, imputing, and preprocessing steps are fit exclusively on the training split and applied to the validation, test, and external sets.
*   **Oversampling-Free Balancing**: The training dataset uses 150,000 unique `Benign` rows and 150,000 unique `Attack` rows (sampled strictly without replacement) to avoid introducing duplicate rows into the learning loop.
*   **Pre-processed Datasets**: The training and external datasets are provided in compressed format and must be extracted before running the pipeline. No dataset merging or preprocessing is required.

---

## 2. Dataset Overview

The datasets are provided in compressed format in the [datasets/](file:///d:/HTTP%20flood%20attack/datasets/) directory. The `.rar` files are tracked in version control, while the extracted `.csv` files are gitignored and must be extracted locally before running the pipeline:

1.  **Training & Internal Validation**: `training_binary.csv` (300,000 rows, balanced 150,000 Benign vs. 150,000 Attack (composed of 75,000 unique Hulk and 75,000 unique HOIC flows)).
    - Source: CSE-CIC-IDS2018 (February 15-21, 2018)
    - Time range: 2018-02-15 01:00:01 to 2018-02-21 10:42:39
    - Compressed file: `datasets/training_binary.rar`

2.  **External Generalization**: `DDos_pcap_binary_external.csv` (918,437 rows, containing Wednesday and Friday afternoon PCAP flows, with all attacks mapped to the `Attack` class).
    - Source: CICIDS2017 (Wednesday + Friday afternoon PCAPs)
    - Contains LOIC DDoS and slow-rate attacks for generalization testing
    - Compressed files: `datasets/DDos_pcap_binary_external.part1.rar` and `part2.rar`

---

## 3. How to Run

### Step 1: Extract the Datasets
The compressed dataset files are located in the `datasets/` directory. Extract them in-place before running the pipeline:

**Windows:**
```powershell
# Extract training dataset
& "C:\Program Files\WinRAR\WinRAR.exe" x "datasets/training_binary.rar" "datasets/"

# Extract external dataset (multi-part archive)
& "C:\Program Files\WinRAR\WinRAR.exe" x "datasets/DDos_pcap_binary_external.part1.rar" "datasets/"
```

**Or use 7-Zip:**
```powershell
& "C:\Program Files\7-Zip\7z.exe" x "datasets/training_binary.rar" -odatasets\
& "C:\Program Files\7-Zip\7z.exe" x "datasets/DDos_pcap_binary_external.part1.rar" -odatasets\
```

After extraction, you should have:
- `datasets/training_binary.csv` (300,000 rows)
- `datasets/DDos_pcap_binary_external.csv` (918,437 rows)

### Step 2: Train and Evaluate the Models
Run the training pipeline to train the baseline and advanced classifiers, save model artifacts, and generate comparative reports:
```powershell
& 'C:\Users\ashok\AppData\Local\Programs\Python\Python310\python.exe' src/run_pipeline.py
```

Outputs are written to:
*   [models/](file:///d:/HTTP%20flood%20attack/models/): Saved `.joblib` model binaries and metadata.
*   [results/](file:///d:/HTTP%20flood%20attack/results/): Predictions and raw metric csv summaries.
*   [reports/](file:///d:/HTTP%20flood%20attack/reports/): Markdown reports detailing dataset analyses, splits, and evaluations.

---

## 4. Execution & Generalization Results

The comparative results of the trained models are summarized below (detailed in [comparative_evaluation.md](file:///d:/HTTP%20flood%20attack/reports/comparative_evaluation.md)):

| Model | Split | Samples | Accuracy | Macro F1 | Status |
| :--- | :--- | ---: | :---: | :---: | :--- |
| **LightGBM** | **Validation** | 29,998 | 0.999800 | **0.999800** | **Selected Best Model** |
| **LightGBM** | **Internal Test** | 60,002 | 0.999950 | **0.999950** | Passed |
| **LightGBM** | **External (Generalization)** | 918,437 | 0.747131 | **0.706019** | **Best Stable Gen** |
| XGBoost | External (Generalization) | 918,437 | 0.746584 | 0.704558 | Similar Generalization |
| TCN (Temporal CNN) | External (Generalization) | 918,437 | 0.738991 | 0.685603 | Sequence-based Generalization |

### Key Generalization Insights
*   **Tree Generalization Limits**: Both LightGBM and XGBoost generalized consistently well on unseen attacks (around **0.706 Macro F1**), providing a strong balance of high precision (low False Positives, ~4%) with moderate detection rate.
*   **Sequential Sequence Power**: The TCN (Temporal Convolutional Network) introduces a sliding window sequence mapping that analyzes sequential windows of consecutive flows, allowing it to capture temporal network behavior during domain shifts.

---

## 5. Loading and Running a Trained Model (Python Example)

To run inference on new flow feature samples, you can load a trained model directly from the [models/](file:///d:/HTTP%20flood%20attack/models/) directory:

```python
import joblib
import pandas as pd

# 1. Load the trained best binary model pipeline (includes the preprocessor steps)
model = joblib.load("models/final_model.joblib")

# 2. Prepare new flow features as a pandas DataFrame (with correct aligned columns)
# new_data = pd.DataFrame(...) 

# 3. Predict malicious vs. benign status
# returns 0 (Benign) or 1 (Attack)
predictions = model.predict(new_data)

# 4. Predict probabilities
# returns [probability_Benign, probability_Attack]
probabilities = model.predict_proba(new_data)
```

