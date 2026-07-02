# Comparative Evaluation Report

Selected final model: `catboost` based on validation macro F1.

| model | split | samples | accuracy | precision_macro | recall_macro | f1_macro | inference_latency_ms_per_sample | training_time_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | external | 918437 | 0.747131 | 0.796479 | 0.703762 | 0.706019 | 0.002276 | 3.767310 |
| xgboost | external | 918437 | 0.746584 | 0.798153 | 0.702634 | 0.704558 | 0.001779 | 4.299160 |
| catboost | external | 918437 | 0.694121 | 0.685335 | 0.672938 | 0.675599 | 0.002060 | 6.577700 |
| catboost | internal_test | 60002 | 0.999967 | 0.999967 | 0.999967 | 0.999967 | 0.001845 | 6.577700 |
| lightgbm | internal_test | 60002 | 0.999950 | 0.999950 | 0.999950 | 0.999950 | 0.002487 | 3.767310 |
| xgboost | internal_test | 60002 | 0.999400 | 0.999401 | 0.999400 | 0.999400 | 0.002031 | 4.299160 |
| catboost | validation | 29998 | 0.999833 | 0.999833 | 0.999833 | 0.999833 | 0.002202 | 6.577700 |
| lightgbm | validation | 29998 | 0.999800 | 0.999800 | 0.999800 | 0.999800 | 0.002769 | 3.767310 |
| xgboost | validation | 29998 | 0.999667 | 0.999667 | 0.999667 | 0.999667 | 0.002002 | 4.299160 |