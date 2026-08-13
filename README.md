# Credit Card Fraud Detection

This repository is an exploratory notebook built on the public Kaggle/ULB
credit-card fraud dataset. It is retained as historical coursework and is not
currently a production ML service.

## What Is Actually Here

- One Google Colab/Jupyter notebook
- A PDF report and presentation
- No API, serialized model, CI pipeline, tests, or deployment configuration

The current notebook uses Isolation Forest. It does not implement the
XGBoost, SMOTE, ROC-AUC, model-serving, or drift-monitoring claims that may
appear in older portfolio copy. No performance metric should be cited without
re-running the notebook and recording the exact dataset split and output.

## Run Locally

Open `Credit_Card_Fraud_Detectio_googlecolab.ipynb` in Jupyter or Google Colab
and provide the dataset through the notebook's documented input path.

## Roadmap Before Portfolio Use

1. Replace the notebook with a reproducible `src/` pipeline.
2. Split before fitting preprocessing and resampling.
3. Add tests, a pinned environment, evaluation JSON, model card, and data card.
4. Add a FastAPI inference endpoint and a documented deployment.

Until those steps are complete, this project should not be presented as a
production fraud-detection system.
