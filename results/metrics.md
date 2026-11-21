| model | pr_auc | roc_auc | precision_at_k | recall_at_1pct_fpr | best_f1 | fit_s | score_s |
|---|---|---|---|---|---|---|---|
| RandomForest (supervised reference) | 0.8132 | 0.9331 | 0.8176 | 0.8456 | 0.8550 | 69.9734 | 0.3743 |
| Local Outlier Factor | 0.5516 | 0.9432 | 0.6284 | 0.8041 | 0.6445 | 3.7977 | 4.6881 |
| One-Class SVM | 0.3310 | 0.9371 | 0.3986 | 0.8041 | 0.4457 | 0.6273 | 8.1436 |
| Autoencoder (PyTorch) | 0.2397 | 0.9252 | 0.3446 | 0.6892 | 0.3481 | 52.9121 | 0.0558 |
| Isolation Forest | 0.1630 | 0.9429 | 0.2568 | 0.6014 | 0.2676 | 11.2005 | 2.5646 |
