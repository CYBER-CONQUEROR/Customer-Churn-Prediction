# Customer Churn Prediction and Retention Analytics

Complete, executed customer churn analysis with reproducible model development, explainability and capacity-constrained retention simulation.

## Submission

- [Executed notebook](01_data_science_notebook.ipynb)
- [Validated cleaned dataset](02_cleaned_dataset.csv)
- [Complete fitted model pipeline](03_model_artifact.joblib)
- [Seven-page business report](04_business_report.pdf)
- [Detailed setup, methodology and limitations](05_README.md)

Gradient Boosting achieved held-out ROC-AUC **0.880** and average precision **0.979**, against a positive prevalence of **0.863**. The top 48 of 240 test customers captured 48 churners: precision **100%**, recall **23.2%**, and lift **1.159x** over random outreach. Financial results are hypothetical and require controlled intervention experiments.

## Reproduce

Use Python 3.12 from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe build_project.py
.\.venv\Scripts\python.exe verify.py
```

Source data, assignment PDF, reusable Python modules, pinned dependencies, generated figures and evaluation results are included. The virtual environment, machine-specific kernel configuration, caches and temporary PDF review renders are excluded.

See [05_README.md](05_README.md) for inference examples, schema, validation design, assumptions and limitations. Load the model only from a trusted source and keep `churn_features.py` importable.
