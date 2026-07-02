# Generalization Study Report

External dataset: `D:\HTTP flood attack\datasets\DDos_pcap_binary_external.csv`
External SHA256: `6f4eed76e05971dc4b8145950ad0fc14ed0d85fd441d7a8838354f4a48d00a1a`

The external dataset is evaluated after normalizing labels and filtering to the predefined class taxonomy. Feature names are normalized into the training schema before inference.

| model | samples | accuracy | precision_macro | recall_macro | f1_macro | inference_latency_ms_per_sample |
| --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 918437 | 0.747131 | 0.796479 | 0.703762 | 0.706019 | 0.010728 |
| xgboost | 918437 | 0.746584 | 0.798153 | 0.702634 | 0.704558 | 0.003123 |
| sequential_markov | 918437 | 0.603821 | 0.731256 | 0.523284 | 0.422401 | 0.066842 |