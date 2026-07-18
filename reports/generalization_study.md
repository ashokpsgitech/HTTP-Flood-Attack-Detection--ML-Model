# Generalization Study Report

External dataset: `D:\Project\HTTP-Flood-Attack-Detection--ML-Model\datasets\DDos_pcap_binary_external.csv`
External SHA256: `6f4eed76e05971dc4b8145950ad0fc14ed0d85fd441d7a8838354f4a48d00a1a`

The external dataset is evaluated after normalizing labels and filtering to the predefined class taxonomy. Feature names are normalized into the training schema before inference.

| model | samples | accuracy | precision_macro | recall_macro | f1_macro | inference_latency_ms_per_sample |
| --- | --- | --- | --- | --- | --- | --- |
| decision_tree | 918437 | 0.791661 | 0.825259 | 0.758286 | 0.767083 | 0.002032 |
| lightgbm | 918437 | 0.747131 | 0.796479 | 0.703762 | 0.706019 | 0.003096 |
| xgboost | 918437 | 0.751625 | 0.822013 | 0.704702 | 0.705736 | 0.002645 |
| catboost | 918437 | 0.694121 | 0.685335 | 0.672938 | 0.675599 | 0.002489 |
| logistic_regression | 918437 | 0.450342 | 0.329466 | 0.393554 | 0.346033 | 0.002505 |