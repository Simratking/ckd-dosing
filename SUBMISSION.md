# Submission: CKD detection + renal dose adjustment

Fill every `[...]` from your own result files. Do not write a number that is not in `results/`.

**Demo:** [link]  **Code:** [repo link]

## Problem
Patients with chronic kidney disease clear many drugs more slowly. Doses that are fine for a healthy adult can cause harm (metformin below eGFR 30, NSAIDs, nitrofurantoin). The project detects CKD with a trained model, estimates kidney function, and adjusts doses from a rule table.

## Data
UCI Chronic Kidney Disease dataset: [400] patients, [24] features, [X]% of cells missing (`results/data_summary.json`). Stratified 70/15/15 split, seed 42: [train/val/test sizes]. Imputation is fitted on the training split only, inside the model pipeline.

## Model setup
- Baselines: logistic regression, random forest, gradient boosting. Best on validation: [name], recall [x], F1 [x] (`results/baseline.json`).
- AutoScientist: [what you gave it: files, target `target`, metric], [search space / model families it tried], [number of trials], [time budget], [number of runs]. Winner chosen on the validation set: [config], validation [metric] = [x].
- Why it beat or matched the baseline: [one honest sentence].

## Results (test set, evaluated once)
| Metric | Value |
|---|---|
| Recall (CKD) | [ ] |
| F1 | [ ] |
| ROC-AUC | [ ] |
| PR-AUC | [ ] |
| Brier score | [ ] |

Robustness (`results/metrics.json`):
- Without serum creatinine: recall [ ], ROC-AUC [ ]
- With 10 / 20 / 30% of values masked: ROC-AUC [ ] / [ ] / [ ]
- 5-fold CV (train+val): F1 [mean ± sd]

Scores on this small dataset are high for most models, so the ablation and masking results are the better evidence of quality.

## Dosing agent
eGFR (CKD-EPI 2021) and CrCl (Cockcroft-Gault) are computed by formula. For each drug, `lookup_dose_rule` reads `rules/dose_rules.json` and `check_contraindication` flags drugs to avoid. Doses never come from an LLM; an optional LLM only rephrases the result, and its text is rejected if it contains a number that is not in the result. Example: [patient] has eGFR [ ], stage [ ]; metformin goes from [standard] to [adjusted].

## Limitations
Small public dataset, not real clinical data. Dosing table is a draft covering 9 adult drugs, no dialysis, no interactions. eGFR staging alone is not a diagnosis. Decision support only.

## Evidence
`results/run_log.md`, AutoScientist screenshots, `baseline.json`, `metrics.json`, `confusion_matrix.png`, `calibration.png`, passing `pytest` output.

---

## 2-minute demo script
1. (10s) "CKD patients need drug doses adjusted to kidney function. This tool detects CKD and adjusts doses."
2. (30s) Healthy adult: probability [ ]%, eGFR [ ], every drug at the usual dose, no warnings.
3. (40s) Switch to stage G3b: metformin reduced, NSAID flagged. Point at the "Based on" column and the note.
4. (30s) Switch to G5: metformin and nitrofurantoin avoided, gabapentin reduced. Show the warnings at the top.
5. (10s) Show `results/metrics.json` (ablation and masking) and say how AutoScientist was used.
