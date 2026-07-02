# Comparative Evaluation Report

Selected final model: `catboost` based on validation macro F1.

| model | split | samples | accuracy | precision_macro | recall_macro | f1_macro | inference_latency_ms_per_sample | training_time_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tcn | external | 918437 | 0.857521 | 0.881094 | 0.834379 | 0.845586 | 0.013035 | 21.772062 |
| lightgbm | external | 918437 | 0.747131 | 0.796479 | 0.703762 | 0.706019 | 0.002165 | 4.162951 |
| xgboost | external | 918437 | 0.746584 | 0.798153 | 0.702634 | 0.704558 | 0.001921 | 4.650927 |
| catboost | external | 918437 | 0.694121 | 0.685335 | 0.672938 | 0.675599 | 0.001987 | 6.599021 |
| catboost | internal_test | 60002 | 0.999967 | 0.999967 | 0.999967 | 0.999967 | 0.001992 | 6.599021 |
| lightgbm | internal_test | 60002 | 0.999950 | 0.999950 | 0.999950 | 0.999950 | 0.002489 | 4.162951 |
| xgboost | internal_test | 60002 | 0.999400 | 0.999401 | 0.999400 | 0.999400 | 0.001970 | 4.650927 |
| tcn | internal_test | 60002 | 0.966984 | 0.969025 | 0.966984 | 0.966948 | 0.012652 | 21.772062 |
| catboost | validation | 29998 | 0.999833 | 0.999833 | 0.999833 | 0.999833 | 0.002209 | 6.599021 |
| lightgbm | validation | 29998 | 0.999800 | 0.999800 | 0.999800 | 0.999800 | 0.002615 | 4.162951 |
| xgboost | validation | 29998 | 0.999667 | 0.999667 | 0.999667 | 0.999667 | 0.002023 | 4.650927 |
| tcn | validation | 29998 | 0.998700 | 0.998700 | 0.998700 | 0.998700 | 0.018444 | 21.772062 |