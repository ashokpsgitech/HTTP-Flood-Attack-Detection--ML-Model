# Comparative Evaluation Report

Selected final model: `catboost` based on validation macro F1.

| model | split | samples | accuracy | precision_macro | recall_macro | f1_macro | inference_latency_ms_per_sample | training_time_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | external | 918437 | 0.747131 | 0.796479 | 0.703762 | 0.706019 | 0.002272 | 4.015692 |
| xgboost | external | 918437 | 0.746584 | 0.798153 | 0.702634 | 0.704558 | 0.001691 | 4.358169 |
| catboost | external | 918437 | 0.694121 | 0.685335 | 0.672938 | 0.675599 | 0.001692 | 6.200279 |
| catboost | internal_test | 60002 | 0.999967 | 0.999967 | 0.999967 | 0.999967 | 0.001990 | 6.200279 |
| lightgbm | internal_test | 60002 | 0.999950 | 0.999950 | 0.999950 | 0.999950 | 0.002316 | 4.015692 |
| xgboost | internal_test | 60002 | 0.999400 | 0.999401 | 0.999400 | 0.999400 | 0.001970 | 4.358169 |
| catboost | validation | 29998 | 0.999833 | 0.999833 | 0.999833 | 0.999833 | 0.002070 | 6.200279 |
| lightgbm | validation | 29998 | 0.999800 | 0.999800 | 0.999800 | 0.999800 | 0.002310 | 4.015692 |
| xgboost | validation | 29998 | 0.999667 | 0.999667 | 0.999667 | 0.999667 | 0.001974 | 4.358169 |