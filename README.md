# Fraud Anomaly Detection

Benchmark of **unsupervised and semi-supervised anomaly detectors** for credit-card fraud, evaluated with metrics that
make sense under extreme class imbalance (PR-AUC, precision@k, recall at a fixed false-positive rate), not accuracy.

![CI](https://github.com/ahmadabdul592/fraud-anomaly-detection/actions/workflows/ci.yml/badge.svg)

## Why this project
Fraud is rare (~0.2–1% of transactions) and labels are scarce and delayed. This repo asks a practical question:
**how much of a supervised model's performance can label-free detectors recover, and how much does it help to fit them on known-normal data only?**

## Approach
| Component | Detail |
|---|---|
| Detectors | Isolation Forest, One-Class SVM, Local Outlier Factor (novelty mode), **Autoencoder** (PyTorch, with a scikit-learn fallback) |
| Two training regimes | `unsupervised`: fit on all training data, no labels. `semi`: fit on normal transactions only |
| Reference | Random Forest *with* fraud labels, shown only as an upper bound |
| Features | Cyclical hour encoding, `log1p(Amount)`, standardised V-features (scaler fit on training rows only, so no leakage) |
| Metrics | PR-AUC (headline), ROC-AUC, precision@k, recall @ 1% FPR, best F1 |
| Score convention | Every detector returns a score where higher = more anomalous |

## Quick start
```bash
git clone https://github.com/ahmadabdul592/fraud-anomaly-detection && cd fraud-anomaly-detection
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ".[torch]" to use the PyTorch autoencoder
pytest -q
python -m fraudad.train --mode unsupervised
python -m fraudad.train --mode semi
```
Outputs go to `results/`: `metrics.csv`, `metrics.md`, `pr_curves.png`, `run_config.json`.

### Use the real dataset
Download the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) to
`data/creditcard.csv` and rerun. The code detects it automatically. With no file present it runs on a synthetic dataset with the identical schema.

## Results
> The table below comes from the **synthetic** dataset (60k rows, 1% fraud) with the scikit-learn autoencoder.
> **Replace it with your numbers from the real Kaggle data** and add your `results/pr_curves.png`.

| Model | PR-AUC (unsupervised) | PR-AUC (semi: normal-only fit) |
|---|---|---|
| Autoencoder (sklearn) | 0.479 | **0.712** |
| Local Outlier Factor | 0.404 | 0.589 |
| One-Class SVM | 0.109 | 0.336 |
| Isolation Forest | 0.050 | 0.064 |
| *RF with labels (reference)* | *0.962* | *0.962* |

**Takeaways (on this synthetic data)**
- Fitting on normal-only data helped every density/reconstruction-based model. Contamination in the training set blurs what "normal" means.
- Reconstruction error (autoencoder) and neighbourhood density (LOF) beat Isolation Forest here, because the fraud differs in *correlated* feature combinations rather than along single axes.
- A large gap to the supervised reference remains, which is what labelled fraud is worth.
- Random baseline PR-AUC equals the fraud rate (0.01), so every model is far above chance.

## Limitations & next steps
- Synthetic results show the pipeline works. They are not evidence about real-world fraud. Run on real data before drawing conclusions.
- Single train/test split. Add repeated stratified CV and confidence intervals.
- Add cost-sensitive thresholding (cost of missed fraud vs. cost of a false alert).
- Explore a variational autoencoder and SHAP explanations for flagged transactions.

## Project layout
```
src/fraudad/  data.py · autoencoder.py · models.py · metrics.py · train.py
tests/        schema, leakage-safe scaling, metrics, autoencoder behaviour, end-to-end runs
```
MIT licensed.
