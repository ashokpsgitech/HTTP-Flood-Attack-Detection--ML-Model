# Dataset Analysis Report

## Class Taxonomy

| Class ID | Label |
|---:|---|
| 0 | Benign |
| 1 | Attack |

## Training Dataset

- Path: `D:\HTTP flood attack\datasets\training_binary.csv`
- SHA256: `2be43ce8befa2840b33a16ccf2716842f36b545c3304a467757f77444756948c`
- Rows after filtering: 300000
- Time range: 2018-02-15 01:00:01 to 2018-02-21 10:42:39
- Exact duplicate rows after filtering: 1

## External Dataset

- Path: `D:\HTTP flood attack\datasets\DDos_pcap_binary_external.csv`
- SHA256: `6f4eed76e05971dc4b8145950ad0fc14ed0d85fd441d7a8838354f4a48d00a1a`
- Rows after filtering: 918437
- Time range: None to None
- Exact duplicate rows after filtering: 87287

## Feature Alignment

- Compatible model features: 77
- Dropped training-only features: Protocol

The external dataset is filtered to labels present in the predefined class taxonomy. Rows outside the taxonomy are excluded before inference and are not used for training, tuning, or preprocessing. External feature names are normalized through the configured CICIDS alias map before schema alignment.