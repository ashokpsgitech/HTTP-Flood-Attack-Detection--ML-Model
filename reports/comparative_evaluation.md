# Comparative Evaluation Report

Selected final model: `xgboost` based on validation macro F1.

| model | split | samples | accuracy | precision_macro | recall_macro | f1_macro | inference_latency_ms_per_sample | training_time_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decision_tree | external | 918437 | 0.791661 | 0.825259 | 0.758286 | 0.767083 | 0.002032 | 9.040155 |
| lightgbm | external | 918437 | 0.747131 | 0.796479 | 0.703762 | 0.706019 | 0.003096 | 6.159665 |
| xgboost | external | 918437 | 0.751625 | 0.822013 | 0.704702 | 0.705736 | 0.002645 | 10.666718 |
| catboost | external | 918437 | 0.694121 | 0.685335 | 0.672938 | 0.675599 | 0.002489 | 10.308170 |
| logistic_regression | external | 918437 | 0.450342 | 0.329466 | 0.393554 | 0.346033 | 0.002505 | 5.257642 |
| catboost | internal_test | 60002 | 0.999967 | 0.999967 | 0.999967 | 0.999967 | 0.002360 | 10.308170 |
| lightgbm | internal_test | 60002 | 0.999950 | 0.999950 | 0.999950 | 0.999950 | 0.003572 | 6.159665 |
| logistic_regression | internal_test | 60002 | 0.999833 | 0.999833 | 0.999833 | 0.999833 | 0.002608 | 5.257642 |
| xgboost | internal_test | 60002 | 0.999817 | 0.999817 | 0.999817 | 0.999817 | 0.002656 | 10.666718 |
| decision_tree | internal_test | 60002 | 0.502700 | 0.718419 | 0.502700 | 0.339654 | 0.002107 | 9.040155 |
| xgboost | validation | 29998 | 0.999867 | 0.999867 | 0.999867 | 0.999867 | 0.002821 | 10.666718 |
| catboost | validation | 29998 | 0.999833 | 0.999833 | 0.999833 | 0.999833 | 0.002488 | 10.308170 |
| lightgbm | validation | 29998 | 0.999800 | 0.999800 | 0.999800 | 0.999800 | 0.007208 | 6.159665 |
| logistic_regression | validation | 29998 | 0.999800 | 0.999800 | 0.999800 | 0.999800 | 0.002197 | 5.257642 |
| decision_tree | validation | 29998 | 0.994566 | 0.994623 | 0.994566 | 0.994566 | 0.002829 | 9.040155 |