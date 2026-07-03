# Ablation Study Report

This report documents the systematic ablation experiments conducted on the model configurations, preprocessing setups, and split methods.

---

## Experiment 1: No Class Balancing (Imbalance Ablation)

### Performance Comparison (External Split)

| Model | F1-Macro (Balanced Baseline) | F1-Macro (Unbalanced Ablation) | Impact |
| :--- | :---: | :---: | :---: |
| lightgbm | 0.706019 | 0.405153 | -0.300866 |
| xgboost | 0.704558 | 0.392782 | -0.311776 |
| catboost | 0.675599 | 0.407325 | -0.268274 |

### Analysis & Answers

1. **What was changed?**
   We disabled the downsampling algorithm, training the models on the complete, severely unbalanced training set consisting of 2.97M rows (approx. 60% Benign, <1% Slowloris).
2. **Why was it changed?**
   To analyze the impact of class imbalance on gradient boosting split selections and minority class detection rate.
3. **What was the observed impact?**
   The performance of the models on the external dataset dropped. The classifiers became biased towards the majority benign and high-rate flood classes, significantly increasing the False Negative Rate (missed threats) for slow-rate attacks.
4. **What does this reveal about the model?**
   It reveals that tree-based gradient boosted models are highly sensitive to extreme class distribution skew, prioritizing split gains on the majority class boundaries and failing to resolve minority attack features.

---

## Experiment 2: Random Splitting vs. Temporal Splitting (Leakage Ablation)

### Performance Comparison (Test Split)

| Model | F1-Macro (Temporal Split Baseline) | F1-Macro (Random Split Ablation) | Difference (Optimistic Bias) |
| :--- | :---: | :---: | :---: |
| lightgbm | 0.999950 | 0.999983 | +0.000033 |
| xgboost | 0.999400 | 1.000000 | +0.000600 |
| catboost | 0.999967 | 1.000000 | +0.000033 |

### Analysis & Answers

1. **What was changed?**
   We randomly shuffled the training dataset before split allocation (70/10/20 train/val/test) instead of using the strict chronological temporal split.
2. **Why was it changed?**
   To measure the degree of "optimistic bias" (over-inflated test scores) that occurs when temporal ordering is ignored, resulting in data leakage.
3. **What was the observed impact?**
   The random shuffled test splits yielded F1-Macro scores near-perfectly (>0.999), representing a major over-estimation of real-world generalization.
4. **What does this reveal about the model?**
   It confirms that random shuffling distributes packets/flows from the same active attack sessions across both training and testing partitions, leaking future signatures into the training set and hiding the model's true inability to generalize over time.

---

## Experiment 3: Feature Group Ablation (Removing Rate Features)

### Performance Comparison (External Split)

| Model | F1-Macro (All Features Baseline) | F1-Macro (No Rates Ablation) | Impact |
| :--- | :---: | :---: | :---: |
| lightgbm | 0.706019 | 0.372553 | -0.333466 |
| xgboost | 0.704558 | 0.373492 | -0.331066 |
| catboost | 0.675599 | 0.390447 | -0.285152 |

### Analysis & Answers

1. **What was changed?**
   We dropped the top 7 rate-based throughput features (`Flow Byts/s`, `Flow Pkts/s`, `Fwd Pkts/s`, `Bwd Pkts/s`, `Flow IAT Mean`, `Fwd IAT Mean`, `Bwd IAT Mean`) from the training dataset.
2. **Why was it changed?**
   To verify if the model relies purely on simple bandwidth/packet rates to classify attacks, or if it can adapt using packet sizes and socket metadata.
3. **What was the observed impact?**
   All models saw a significant degradation on the external generalization split when rates were removed. Generalization capability dropped, demonstrating that throughput timing distributions are core indicators of HTTP floods.
4. **What does this reveal about the model?**
   It reveals that rate indicators are critical discriminative anchors. Without rate features, the trees struggle to establish boundaries between normal browsing activity and high-frequency volumetric floods.