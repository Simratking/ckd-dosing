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
pip install -r requirements-dev.txt
python -m pytest            # 43 tests, should all pass
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

## What we trained with AutoScientist

AutoScientist fine-tunes language models on prompt and completion text, so `scripts/make_llm_dataset.py` turns each patient into a text prompt with the answer "CKD" or "not CKD" (`data/llm/`). The platform's data-enhancement step changed 37 of 175 CKD labels to "not CKD", so we rejected it and trained on the original columns (`results/evolved_rows_check.json`). The model is Qwen3-4B-Base with LoRA; every setting is in `results/run_log.md`. Score its answers on the held-out test set with `python scripts/score_llm.py outputs.csv --baseline`. The full write-up is in `SUBMISSION.md`.

## Smoke testing without data

`python scripts/make_synthetic.py` writes **invented** data to `data/raw/synthetic_ckd.csv`, and `python -m src.data_prep --raw data/raw/synthetic_ckd.csv` runs the pipeline on it. Never report numbers from it. Delete `data/processed`, `models` and `results` before the real run.

## Changes over time

`assess_history(visits, drugs)` in `src/history.py` takes dated visits (each with that visit's creatinine, weight and age), runs the normal assessment on each, and reports: per-drug dose changes between visits, stage changes, the eGFR slope per year, and a rapid-decline flag (falling by more than 5 per year). It warns that a dose change should be confirmed with a repeat creatinine. It also gives the dose needed at every visit for every drug.

From a file: one row per visit with columns `date` (YYYY-MM-DD), `age`, `sex` (M/F), `sc` (creatinine mg/dL), and optionally `weight` and `patient_id` (several patients can share one file). See `examples/visits.csv`.

```
python -m src.history examples/visits.csv --drugs metformin,gabapentin,ibuprofen
```

This prints each patient's dose changes and writes the full per-visit timeline to `results/dose_timeline.csv`. The app's "Changes over time" section accepts the same CSV upload, or an editable visit table. The trend is a straight-line fit, so treat it as rough with only 2 or 3 visits. It does not forecast future eGFR.

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
src/history.py          dose changes across dated visits + eGFR trend
src/cases.py            3 demo patients
scripts/                download_data.py, make_synthetic.py, make_llm_dataset.py, score_llm.py, check_evolved*.py
examples/visits.csv     made-up sample visits for 2 patients
tests/                  43 tests
results/run_log.md      fill this in as you go
```
