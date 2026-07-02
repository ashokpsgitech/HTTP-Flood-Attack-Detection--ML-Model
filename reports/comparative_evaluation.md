# Comparative Evaluation Report

Selected final model: `logistic_regression_baseline` based on validation macro F1.

| model | split | samples | accuracy | precision_macro | recall_macro | f1_macro | inference_latency_ms_per_sample | training_time_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression_baseline | external | 918437 | 0.711091 | 0.711282 | 0.717646 | 0.709018 | 0.001518 | 85.670578 |
| hist_gradient_boosting | external | 918437 | 0.746162 | 0.792876 | 0.703267 | 0.705622 | 0.003192 | 8.594630 |
| random_forest | external | 918437 | 0.650337 | 0.791321 | 0.579318 | 0.524544 | 0.003100 | 13.131189 |
| logistic_regression_baseline | internal_test | 60002 | 0.999933 | 0.999933 | 0.999933 | 0.999933 | 0.001510 | 85.670578 |
| hist_gradient_boosting | internal_test | 60002 | 0.999800 | 0.999800 | 0.999800 | 0.999800 | 0.003377 | 8.594630 |
| random_forest | internal_test | 60002 | 0.994567 | 0.994625 | 0.994567 | 0.994567 | 0.003415 | 13.131189 |
| logistic_regression_baseline | validation | 29998 | 0.999767 | 0.999767 | 0.999767 | 0.999767 | 0.001698 | 85.670578 |
| hist_gradient_boosting | validation | 29998 | 0.999733 | 0.999733 | 0.999733 | 0.999733 | 0.003922 | 8.594630 |
| random_forest | validation | 29998 | 0.996600 | 0.996623 | 0.996600 | 0.996600 | 0.003767 | 13.131189 |