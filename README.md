# Cardio Catch Disease — Cardiovascular Risk Screening

> Balanced binary classification · Clinical risk screening · Sensitivity/specificity

## Business Problem

A clinic wants to flag patients at elevated risk of cardiovascular disease from a routine
examination, so physicians can prioritize follow-up tests and lifestyle intervention. The decision
the model informs is **which patients to escalate** for closer cardiac assessment.

The cost of error is clinical, not symmetric. A false negative (missing a real case) is the most
serious outcome — an untreated patient at risk — while a false positive sends a healthy patient for
an unnecessary follow-up, costing time and reassurance. The screening threshold therefore trades
**sensitivity against specificity**, and the tool is positioned as decision support, never a
diagnosis. Because the classes are roughly balanced (~50%), accuracy is informative but the
sensitivity/specificity pair is what clinicians read.

A fixed rule ("flag everyone with high blood pressure") captures the dominant signal but misses
the way age, cholesterol and body composition combine, which the model uses to rank borderline cases.

## Dataset

[Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

| Property | Value |
|----------|-------|
| Rows | 70,000 examinations |
| Target | `cardio` (1 = cardiovascular disease present) |
| Positive rate | 50.0% (balanced) |
| Key inputs | blood pressure, age, cholesterol, weight/height, glucose, lifestyle flags |

## Solution Strategy

1. **Acquisition** — pull the dataset from Kaggle on demand (semicolon-separated); a versioned stratified sample backs an offline run.
2. **Data quality** — the raw blood-pressure fields contain impossible entries (negative and 16,000+); these are clipped to plausible clinical bounds inside the pipeline.
3. **Leakage control** — every measurement is taken at the examination, before any diagnosis, so there is no target leakage; `id` is dropped.
4. **Feature engineering** — age in years, BMI, pulse pressure and a hypertension flag, all inside the model `Pipeline` so serving reuses the exact transform.
5. **Model selection** — `StratifiedKFold` cross-validation compares a logistic baseline, random forest and histogram gradient boosting on ROC AUC; the winner is tuned with `RandomizedSearchCV`.
6. **Evaluation** — ROC AUC and average precision on a stratified holdout, plus sensitivity, specificity and accuracy at the decision threshold, and ROC AUC by patient segment.

## Top Insights & Hypotheses

- **Systolic blood pressure dominates** (permutation importance 0.17): it carries far more screening signal than any other input, confirming hypertension as the leading marker.
- **Age and cholesterol follow**, with risk rising steeply after the mid-fifties and at the highest cholesterol band.
- **The model discriminates least well among patients with very high cholesterol** (ROC AUC 0.66 vs ~0.80 overall), because that group is already mostly positive — a limitation flagged in Next Steps.
- **Lifestyle flags (smoking, alcohol, activity) add little** on their own once blood pressure and body composition are in the model.

## Engineered Features

| Feature | Formula | Business signal |
|---------|---------|-----------------|
| age_years | `age_days / 365.25` | Age on a human scale; risk rises non-linearly with it. |
| bmi | `weight / (height_m)^2` (clipped 10-80) | Body composition; obesity raises cardiovascular risk. |
| pulse_pressure | `ap_hi - ap_lo` | Arterial stiffness marker beyond either pressure alone. |
| high_blood_pressure | `ap_hi >= 140 or ap_lo >= 90` | Clinical hypertension threshold as a binary flag. |

## Model

A histogram gradient boosting classifier (selected by cross-validation, tuned with randomized
search) inside a `Pipeline` that owns the clipping, engineering and scaling. The logistic baseline
sets the bar.

| Model | CV ROC AUC | Holdout ROC AUC | Holdout AP |
|-------|-----------:|----------------:|-----------:|
| Logistic baseline | 0.794 | 0.790 | — |
| Random forest | 0.774 | — | — |
| **Hist gradient boosting (final)** | **0.803** | **0.800** | **0.781** |

## Business Results

At the default 0.5 threshold on the holdout:

| Metric | Value |
|--------|------:|
| Sensitivity (true cases flagged) | 69% |
| Specificity (healthy cleared) | 78% |
| Accuracy | 73% |
| ROC AUC | 0.80 |

The model flags **69% of true cardiovascular cases while clearing 78% of healthy patients**. The
threshold is adjustable: lowering it raises sensitivity (catch more disease) at the cost of more
false alarms, which the app lets a clinician explore directly.

## How to Run

1. **Clone**
   ```
   git clone https://github.com/pmusachio/cardio-catch-disease.git
   cd cardio-catch-disease
   ```
2. **Environment**
   ```
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Kaggle access** — place a Kaggle API token at `~/.kaggle/`; the pipeline falls back to the versioned sample if none is present.
4. **Run the pipeline**
   ```
   python -m src.pipeline
   ```
5. **Tests**
   ```
   pytest tests/
   ```
6. **App (local)**
   ```
   streamlit run app/streamlit_app.py
   ```
7. **Live app** — [cardio-catch-disease.onrender.com](https://cardio-catch-disease.onrender.com) — assess a patient and explore the threshold trade-off.

## Next Steps

- Calibrate the probabilities so the displayed risk can be read as an absolute likelihood, which matters for a clinical tool; deferred pending validation against real outcomes.
- Add laboratory and family-history features to improve discrimination in the high-cholesterol group, where the current inputs separate cases poorly.
- Set the operating threshold from the clinic's true cost of a missed case versus an unnecessary follow-up, rather than the default 0.5; deferred until those costs are agreed with clinicians.
