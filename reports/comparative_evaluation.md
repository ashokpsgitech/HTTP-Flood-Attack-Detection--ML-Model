# Comparative Evaluation Report

Selected final model: `lightgbm` based on validation macro F1.

| model | split | samples | accuracy | precision_macro | recall_macro | f1_macro | inference_latency_ms_per_sample | training_time_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | external | 918437 | 0.747131 | 0.796479 | 0.703762 | 0.706019 | 0.010728 | 16.075995 |
| xgboost | external | 918437 | 0.746584 | 0.798153 | 0.702634 | 0.704558 | 0.003123 | 19.355125 |
| sequential_markov | external | 918437 | 0.603821 | 0.731256 | 0.523284 | 0.422401 | 0.066842 | 18.746156 |
| lightgbm | internal_test | 60002 | 0.999950 | 0.999950 | 0.999950 | 0.999950 | 0.012223 | 16.075995 |
| xgboost | internal_test | 60002 | 0.999400 | 0.999401 | 0.999400 | 0.999400 | 0.004438 | 19.355125 |
| sequential_markov | internal_test | 60002 | 0.951368 | 0.951368 | 0.951368 | 0.951368 | 0.071481 | 18.746156 |
| lightgbm | validation | 29998 | 0.999800 | 0.999800 | 0.999800 | 0.999800 | 0.011788 | 16.075995 |
| xgboost | validation | 29998 | 0.999667 | 0.999667 | 0.999667 | 0.999667 | 0.007939 | 19.355125 |
| sequential_markov | validation | 29998 | 0.998700 | 0.998700 | 0.998700 | 0.998700 | 0.070983 | 18.746156 |