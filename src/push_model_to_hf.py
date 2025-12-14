import os
from huggingface_hub import HfApi

HF_MODEL_REPO = "VJBharathkumar/tourism-prod-prediction"  

def main():
    api = HfApi()

    # Ensure artifacts exist
    if not os.path.exists("artifacts/model.joblib"):
        raise FileNotFoundError("artifacts/model.joblib not found. Run: python src/train.py")
    if not os.path.exists("artifacts/metrics.json"):
        raise FileNotFoundError("artifacts/metrics.json not found. Run: python src/train.py")

    # Upload model + metrics
    api.upload_file(
        path_or_fileobj="artifacts/model.joblib",
        path_in_repo="model.joblib",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
    )

    api.upload_file(
        path_or_fileobj="artifacts/metrics.json",
        path_in_repo="metrics.json",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
    )

    # Optional: upload a quick README
    readme_text = """---
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
"""
    with open("README_MODEL.md", "w", encoding="utf-8") as f:
        f.write(readme_text)

    api.upload_file(
        path_or_fileobj="README_MODEL.md",
        path_in_repo="README.md",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
    )

    print("Uploaded model.joblib, metrics.json, and README.md to Hugging Face Model Hub")
    print("Model repo:", HF_MODEL_REPO)

if __name__ == "__main__":
    main()