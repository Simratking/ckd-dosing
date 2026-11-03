# CKD detection + renal dose adjustment

A model detects chronic kidney disease (CKD) from labs and vitals. eGFR and CKD stage come from the CKD-EPI 2021 formula. An assistant then adjusts doses for 9 common drugs from a rule table and flags drugs to avoid.

> Prototype decision support, not a prescriber. `rules/dose_rules.json` is a **draft**: verify every row against the product label and KDIGO before relying on it.

## How the pieces fit

```
labs, age, sex, weight, drugs
   |-> Model A (trained: you + AutoScientist) -> CKD probability
   |-> CKD-EPI 2021 eGFR + Cockcroft-Gault CrCl -> stage G1..G5
   |-> lookup_dose_rule / check_contraindication -> rules/dose_rules.json -> adjusted dose, warnings
   `-> explanation (template, or an LLM whose numbers are checked against the result)
```

Doses never come from an LLM. The agent code is deterministic; an LLM may only rephrase, and its text is thrown away if it contains a number that is not in the result.

## Setup (Windows, macOS, Linux)

```
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate
pip install -r requirements.txt
python -m pytest            # 36 tests, should all pass
```

Tabular training runs fine on CPU. You do not need the GPU.

## Run order

1. **Get the real data**: `python scripts/download_data.py` (UCI CKD, 400 rows). If it fails, download the CSV from UCI or Kaggle and save it as `data/raw/ckd.csv`. Both column naming styles work.
2. **Prepare**: `python -m src.data_prep` writes `data/processed/{train,val,test}.csv` (stratified 70/15/15, seed 42) plus `results/missing_values.csv` and `results/data_summary.json`.
3. **Baselines**: `python -m src.train` scores logistic regression, random forest and gradient boosting on the validation set, writes `results/baseline.json`, and saves the best one as `models/model.joblib` so the demo already works.
4. **AutoScientist run** (the main event): give it `data/processed/train.csv` and `val.csv`. The target column is `target` (1 = CKD). Optimize recall or F1 for the CKD class. Imputation: the files contain missing values, so use the platform's imputation or a pipeline that includes one. Record every setting and keep screenshots, and log each run in `results/run_log.md`.
5. **Use its model**: if it exports a scikit-learn model, save it with `joblib.dump(model, "models/model.joblib")`. If it exports something else, have it predict probabilities for `data/processed/test.csv` and use the CSV in step 6.
6. **Evaluate once**: `python -m src.evaluate` (or `python -m src.evaluate --proba-csv probs.csv`). It reports test metrics, a creatinine-ablation, random-masking tests (10/20/30%), 5-fold CV, and writes the confusion matrix and calibration plots. It refuses to run a second time without `--force`, so the test set is not tuned against. The ablation, masking and CV need a loadable model.
7. **Demo**: `streamlit run app.py`. Three made-up patients are built in (healthy, G3b, G5).
8. **Deploy**: Streamlit Community Cloud or Hugging Face Spaces. Commit `models/model.joblib` (or retrain at startup) and `rules/dose_rules.json`.

## Smoke testing without data

`python scripts/make_synthetic.py` writes **invented** data to `data/raw/synthetic_ckd.csv`, and `python -m src.data_prep --raw data/raw/synthetic_ckd.csv` runs the pipeline on it. Never report numbers from it. Delete `data/processed`, `models` and `results` before the real run.

## Optional LLM explanation

`pip install anthropic`, set `ANTHROPIC_API_KEY` (and optionally `CKD_LLM_MODEL`). Without a key the app uses a fixed template.

## Honest limits (put these in your write-up)

- UCI CKD is small (400 rows) and easy to separate, so headline scores will be near-perfect. The ablation and masking results are the real evidence of quality.
- eGFR stage is a formula, not machine learning. The trained model detects CKD.
- A G-stage from eGFR alone is not a diagnosis; real diagnosis needs persistence over 3 months and/or markers of kidney damage.
- Dosing rules are a draft and cover 9 drugs, adults only, no dialysis, no drug interactions.

## Layout

```
app.py                  Streamlit demo
rules/dose_rules.json   dosing bands per drug (DRAFT)
src/features.py         schema, cleaning, Model A inference
src/data_prep.py        clean + split
src/train.py            baselines
src/evaluate.py         test metrics + robustness
src/egfr.py             CKD-EPI 2021, Cockcroft-Gault, stages
src/dose_rules.py       lookup_dose_rule, check_contraindication
src/agent.py            assess() + explain()
src/cases.py            3 demo patients
scripts/                download_data.py, make_synthetic.py
tests/                  36 tests
results/run_log.md      fill this in as you go
```
