# Submission: CKD detection + renal dose adjustment

**Demo:** **[FILL: live link after deploying]**  **Code:** https://github.com/Simratking/ckd-dosing

Every number below comes from a file in `results/`. Items marked **[FILL]** depend on the AutoScientist model's result and are left blank on purpose.

## Problem
Patients with chronic kidney disease (CKD) clear many drugs more slowly. A dose that is fine for a healthy adult can cause harm (metformin below eGFR 30, NSAIDs, nitrofurantoin). This project detects CKD with a trained model, estimates kidney function with a standard formula, and adjusts doses for 9 common drugs from a rule table. It is decision support for a prototype, not a prescriber.

## Data
UCI Chronic Kidney Disease dataset (Rubini, Soundarapandian and Eswaran, 2015; check the exact citation on the UCI page): 400 patients, 24 features, 10.5% of cells missing (`results/data_summary.json`). Stratified 70/15/15 split with seed 42: 280 train / 60 validation / 60 test (CKD cases 175 / 38 / 37). Missing values are imputed inside the model pipeline, fitted on the training split only. The test set was evaluated once.

## Model setup and training process
**Baselines** (validation set, `results/baseline.json`): logistic regression, random forest and gradient boosting scored recall 1.000 / 1.000 / 0.974 and F1 1.000 / 1.000 / 0.987. Logistic regression was selected.

**AutoScientist run** (full step-by-step record in `results/run_log.md`, with screenshots):
- AutoScientist fine-tunes language models on prompt and completion data, so I converted each patient into a text prompt (labs and findings, missing values written as "not recorded") with the answer "CKD" or "not CKD" (`scripts/make_llm_dataset.py`).
- The platform's data-enhancement step raised its own quality score from 6.0 to 8.5, but a row-by-row check showed it **changed 37 of 175 CKD labels to "not CKD"** (21%), had emptied the enhanced prompts, and dropped 2 rows. A model trained on that would learn to miss real cases, so I rejected the enhanced columns and trained on the original prompt and completion columns (`results/evolved_rows_check.json`, `results/screenshots/adaption_flipped_label.png`).
- Configuration: Instruction Tuning, all model sizes, base model Qwen3-4B-Base, LoRA (r 8, alpha 16, dropout 0, q/k/v/o projections), supervised fine-tuning, 2 epochs, learning rate 5e-5, cosine schedule, warmup 0.05, gradient clipping 1, weight decay 0.01, 1x H100. The platform requires at least 1,000 rows, so I added 750 general-purpose rows with no CKD content (no unverifiable labels) and no domain-specific synthetic rows. Cost: 8 credits (training 0 via voucher, 8 for the 750 general-purpose rows).
- Result of the run: finished after 1 hour 51 minutes (52 training steps). Training loss 1.158 at step 1 and 1.134 at step 52; validation loss 1.143 at step 12 and 1.068 at step 52 (`results/autoscientist_trainer_state.json`). Only 278 of the 1,028 training rows were patient data, so these loss values say little about CKD accuracy by themselves.

## Results (test set, n = 60, evaluated once)
| Model | Recall (CKD) | Precision | F1 | ROC-AUC | Brier |
|---|---|---|---|---|---|
| Logistic regression baseline | 0.973 | 1.000 | 0.986 | 1.000 | 0.0096 |
| Fine-tuned Qwen3-4B (AutoScientist) | not scored on the test set | | | n/a (text answers) | n/a |

The baseline missed 1 CKD case out of 37 and raised 0 false alarms (confusion matrix `[[23, 0], [1, 36]]`). 

**Fine-tuned model, validation set** (n = 60: 38 CKD, 22 not CKD; `results/autoscientist_val.json`): it answered "CKD" for all 60 patients (confusion matrix `[[0, 22], [0, 38]]`, 0 unparseable answers). Its recall of 1.000 comes only from never answering "not CKD"; precision is 0.633 and F1 is 0.776, the same as always answering CKD. The baseline scored F1 1.000 on this validation set. A separate check of the model's score margin, log P("CKD") minus log P("not CKD"), gives ROC-AUC 0.963 (`results/autoscientist_margin_auc.json`; code in `scripts/colab_score_val.py`): the model ranks CKD above not-CKD fairly well. Its greedy answers were "CKD" for all 60 patients, and all 60 margins are positive; the margin compares summed token log-probabilities of two answers of different length, so its zero point is not a calibrated threshold and only the ranking (AUC) is used. I therefore report this run as not better than the baseline. Possible contributors, which I did not test: only 278 of 1,028 training rows were patient data, training lasted 52 steps, and the patient rows were 175 CKD to 103 not CKD.

With only 60 test rows, one case moves recall by about 3 points, so read differences with the confidence intervals.

**Robustness of the baseline** (`results/metrics.json`):
- Without serum creatinine: recall 0.973, ROC-AUC 1.000 (no change), so the model is relying on other markers as well.
- With 10 / 20 / 30% of values randomly masked: ROC-AUC 0.9995 / 0.9984 / 0.9934; recall 0.962 / 0.962 / 0.924.
- 5-fold cross-validation on train and validation: F1 0.995 ± 0.010, recall 0.991 ± 0.019.

Scores are near-perfect because this public dataset is small and its classes are well separated. These numbers show the pipeline works; they do not predict performance on real patients.

## Dosing agent
eGFR (CKD-EPI 2021) and creatinine clearance (Cockcroft-Gault) are computed by formula, not learned. `lookup_dose_rule` and `check_contraindication` read `rules/dose_rules.json` (9 drugs, adults). Doses never come from an LLM; an optional LLM may only rephrase the result, and its text is discarded if it contains a number that is not in the result. The CKD probability in the demo comes from the logistic regression baseline (`models/model.joblib`), because the fine-tuned model returns text answers, not probabilities, and did not beat the baseline.

Demo cases (made-up patients, checked in `results/run_log.md`):
- Healthy adult (45 y, creatinine 0.90): CKD probability 2.7%, eGFR 107.3, stage G1, usual doses for all drugs.
- Stage G3b (68 y, creatinine 1.90): eGFR 37.9; metformin reduced to a maximum of about 1000 mg/day and not to be started, gabapentin 400 to 1400 mg/day, ibuprofen flagged for caution.
- Stage G5 (72 y, creatinine 3.60): eGFR 12.9; warnings first for metformin (contraindicated), ibuprofen (avoid) and nitrofurantoin (avoid); gabapentin 100 to 300 mg/day.

## Limitations
- Small public dataset, not real clinical data; a 60-row test set.
- The dosing table is a draft covering 9 adult drugs, with no dialysis and no drug interactions. Verify every row against the product label and KDIGO before any real use.
- An eGFR stage alone is not a diagnosis.
- Decision support only. A clinician or pharmacist must confirm any dose.

## Evidence
`results/run_log.md`, `results/screenshots/`, `results/baseline.json`, `results/metrics.json`, `results/confusion_matrix.png`, `results/calibration.png`, `results/evolved_rows_check.json`, `results/autoscientist_val.json`, `results/autoscientist_val_predictions.csv`, `results/autoscientist_margin_auc.json`, `results/autoscientist_margins.csv`, `results/autoscientist_trainer_state.json`, and the passing `pytest` run (43 tests).

---

## 2-minute demo script
1. (10s) "CKD patients need drug doses adjusted to kidney function. This tool detects CKD and adjusts the doses."
2. (30s) Healthy adult: probability 2.7%, eGFR 107.3, usual doses, no warnings.
3. (40s) Stage G3b: metformin reduced, NSAID flagged. Point at the "Based on" column and the note.
4. (30s) Stage G5: metformin and nitrofurantoin avoided, gabapentin reduced. Show the warnings at the top.
5. (10s) Show the row-level label check that rejected 37 flipped labels, then the results table.
