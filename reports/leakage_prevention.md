# Leakage Prevention Documentation

- Preprocessing is fitted only on the temporal training split.
- Validation, internal test, and external datasets are transformed with the fitted preprocessing pipeline.
- Feature selection through variance filtering is learned only from the training split.
- Normalization statistics are learned only from the training split.
- Training duplicates are removed before model fitting.
- Rows duplicated from training are removed from validation, internal test, and external evaluation.
- External dataset labels are filtered before inference according to the predefined taxonomy.
- External data is not used for model selection, threshold tuning, balancing, or hyperparameter tuning.