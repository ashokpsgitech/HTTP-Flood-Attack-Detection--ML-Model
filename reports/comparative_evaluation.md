# Comparative Evaluation Report

Selected final model: `lightgbm` based on validation macro F1.

| model | split | samples | accuracy | precision_macro | recall_macro | f1_macro | inference_latency_ms_per_sample | training_time_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | external | 918437 | 0.747131 | 0.796479 | 0.703762 | 0.706019 | 0.001969 | 3.571719 |
| xgboost | external | 918437 | 0.746584 | 0.798153 | 0.702634 | 0.704558 | 0.001807 | 4.543637 |
| tcn | external | 918437 | 0.738991 | 0.819405 | 0.688661 | 0.685603 | 0.011940 | 21.649581 |
| lightgbm | internal_test | 60002 | 0.999950 | 0.999950 | 0.999950 | 0.999950 | 0.002264 | 3.571719 |
| xgboost | internal_test | 60002 | 0.999400 | 0.999401 | 0.999400 | 0.999400 | 0.001868 | 4.543637 |
| tcn | internal_test | 60002 | 0.975717 | 0.976839 | 0.975717 | 0.975703 | 0.012387 | 21.649581 |
| lightgbm | validation | 29998 | 0.999800 | 0.999800 | 0.999800 | 0.999800 | 0.002239 | 3.571719 |
| xgboost | validation | 29998 | 0.999667 | 0.999667 | 0.999667 | 0.999667 | 0.001884 | 4.543637 |
| tcn | validation | 29998 | 0.999033 | 0.999034 | 0.999033 | 0.999033 | 0.018022 | 21.649581 |