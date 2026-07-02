# Temporal Splitting Documentation

The training dataset is sorted by `Timestamp` using stable ordering. No random shuffle or stratified split is used.

- Training split: 70%
- Validation split: 10%
- Internal test split: 20%

| Split | Rows | Timestamp Min | Timestamp Max |
|---|---:|---|---|
| train | 210000 | 2018-02-15 01:00:01 | 2018-02-21 02:16:49 |
| validation | 29998 | 2018-02-16 01:47:21 | 2018-02-21 02:21:21 |
| internal_test | 60002 | 2018-02-21 02:18:06 | 2018-02-21 10:42:39 |
| external | 918437 | None | None |

## Split Label Distribution

Because strict temporal order is preserved, some later splits may not contain every class. This is reported explicitly instead of using stratified sampling, which would violate the project guideline.

### train

| Label | Count |
|---|---:|
| Benign | 105000 |
| Attack | 105000 |

### validation

| Label | Count |
|---|---:|
| Benign | 14999 |
| Attack | 14999 |

### internal_test

| Label | Count |
|---|---:|
| Benign | 30001 |
| Attack | 30001 |

### external

| Label | Count |
|---|---:|
| Benign | 537749 |
| Attack | 380688 |
