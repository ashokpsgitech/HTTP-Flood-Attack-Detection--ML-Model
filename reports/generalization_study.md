# Generalization Study Report

External dataset: `D:\HTTP flood attack\datasets\DDos_pcap_binary_external.csv`
External SHA256: `6f4eed76e05971dc4b8145950ad0fc14ed0d85fd441d7a8838354f4a48d00a1a`

The external dataset is evaluated after normalizing labels and filtering to the predefined class taxonomy. Feature names are normalized into the training schema before inference.

| model | samples | accuracy | precision_macro | recall_macro | f1_macro | inference_latency_ms_per_sample |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression_baseline | 918437 | 0.711091 | 0.711282 | 0.717646 | 0.709018 | 0.001518 |
| hist_gradient_boosting | 918437 | 0.746162 | 0.792876 | 0.703267 | 0.705622 | 0.003192 |
| random_forest | 918437 | 0.650337 | 0.791321 | 0.579318 | 0.524544 | 0.003100 |