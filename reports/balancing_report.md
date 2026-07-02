# Data Balancing Report

The implementation uses cost-sensitive learning where supported instead of synthetic oversampling.

- Logistic Regression uses `class_weight='balanced'`.
- Random Forest uses `class_weight='balanced_subsample'`.
- HistGradientBoosting is trained without synthetic resampling.

SMOTE/ADASYN are not applied by default because the dataset is temporal and flow-based; generating synthetic minority flows before careful analysis could distort the traffic chronology and feature relationships.