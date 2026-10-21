# Customer Churn Prediction and Retention Analytics

Predict next-cycle churn and evaluate a 20% retention contact queue. Source assignment: `DS 01.pdf`; source data: `customer_churn_advanced_dataset.csv`. All numerical conclusions are in the executed notebook, `results/summary.json`, and the seven-page business report.

## Install and reproduce (Windows PowerShell, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe build_project.py
.\.venv\Scripts\python.exe verify.py
```

`build_project.py` rebuilds the notebook from readable analysis sections and executes every cell in a fresh kernel, regenerating CSV, figures, tuning tables, model, report and summary, then runs QA. Use the project root as the working directory. The original sources must be present. No hidden notebook state or network access is required after installation. Python 3.12 was used; exact direct package versions are pinned in requirements.txt and shown in the notebook. Seed 42 is used throughout randomized analysis. CPU searches run serially on five stratified folds.

## Files

- `01_data_science_notebook.ipynb`: complete executed A-H analysis, tables and inline figures.
- `02_cleaned_dataset.csv`: all 1,200 validated records and 20 original columns; missing values retained; no rows dropped, caps or learned imputations.
- `03_model_artifact.joblib`: complete fitted raw-feature transformer, ColumnTransformer and classifier.
- `04_business_report.pdf`: business decision brief with observed metrics and hypothetical economics.
- `05_README.md`: setup and methodology.
- `analysis_cells.py`, `build_project.py`, `churn_features.py`, `report.py`, `verify.py`: reusable source and QA.
- `figures/`: publication-resolution EDA, evaluation and explanation charts.
- `results/`: audit, billing investigations, split membership, complete search tables, comparisons, predictions, ranking, sensitivity and summary.

## Data and methodology

The schema has a unique customer_id, ten numeric predictors (age, tenure, monthly/total charges, complaints, late payments, satisfaction, monthly usage, discount and senior status), eight categorical predictors (contract, internet, region, payment, paperless, dependents and two support flags), and binary churn. Churn=1 is the majority: 1,036 of 1,200. Missing counts are 30 monthly charges, 24 total charges, 48 satisfaction and 36 usage. No exact duplicates were observed.

One stratified 80/20 split is locked at the beginning. EDA uses training labels. Feature engineering is deterministic inside the pipeline; numeric median imputation, categorical mode imputation, scaling and one-hot encoding are learned per fold. Explicit column allowlists exclude ID and churn. Logistic Regression, Random Forest and Gradient Boosting have baseline and efficient grid searches; DummyClassifier gives context. Five-fold training CV AP selects the family, with ROC, recall, F1 and top-20 precision reported. A training-only engineered-feature ablation selects inclusion. CV fold variation is the robustness check. Selection uses shared folds and is not nested; CV and selected-model OOF utility can be optimistic.

Training OOF probabilities choose a fixed threshold by hypothetical net utility. The final test is evaluated after decisions are locked. Held-out permutation importance is explanatory only. Local explanations replace individual attributes by training medians/modes and recompute derived features; these are reference sensitivities, not additive or causal attributions. The saved model remains fitted on the 80% training data so test prediction parity is verifiable.

## Raw-row inference

Keep `churn_features.py` importable and use the pinned environment. Load only trusted joblib artifacts because deserialization can execute code.

```python
import pandas as pd
import joblib
model = joblib.load('03_model_artifact.joblib')
new = pd.read_csv('customer_churn_advanced_dataset.csv').drop(columns='churn').head(3)
probability = model.predict_proba(new)[:, 1]
print(probability)
```

Replace `new` with valid new raw customer rows using the same named predictor columns; ID is optional and ignored. Column order is immaterial. Unknown categories and numeric missing values are supported. Operational fixed classification uses the threshold in `results/summary.json`, not the classifier's default predict threshold.

## Targeting assumptions and limitations

Edit `CFG` in analysis_cells.py: contact cost $5, retained value $200, success rate 25%, capacity 20%. Ranking selects exactly ceil(capacity*N), breaking equal probabilities by customer_id. The simulation uses actual test prevalence for the uniform-random benchmark and varies both success and retained value. These are hypothetical effects and values, not measured ROI.

Validate target definition, sample provenance, feature availability at scoring, timing, fairness, data drift and external/temporal performance. The cohort has unusually high churn, and only 33 test non-churners. Correlated inputs complicate attribution. Billing anomalies need invoice review. Randomized outreach trials must establish incremental retention effects before making financial claims.

QA checks schema, binary target, preserved records, independent split membership, feature exclusion, unknown/missing input execution, column order, fresh-process model reload parity, saved probabilities, exact capacity, capture/precision/recall/lift arithmetic, executed notebook cells, PDF page count and mandatory file existence.
