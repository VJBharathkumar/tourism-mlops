---
license: mit
tags:
- tabular-classification
- scikit-learn
- random-forest
---

# Tourism Package Prediction Model

This repository contains a scikit-learn Pipeline trained to predict `ProdTaken` (whether a customer will take a tourism package).

## Metrics (test set)
- Accuracy: 0.9068
- F1: 0.7159
- ROC-AUC: 0.9597

## Artifacts
- `model.joblib`: saved scikit-learn Pipeline (preprocessing + model)
- `metrics.json`: evaluation metrics + best hyperparameters
