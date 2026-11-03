"""Turn the processed CKD CSVs into prompt/completion files for an LLM fine-tuning platform.

    python scripts/make_llm_dataset.py

Writes data/llm/{train,val,test}.csv and .jsonl with columns: prompt, completion.
Completion is exactly "CKD" or "not CKD". Missing values are written as "not recorded".
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NUM = {"age": "age (years)", "bp": "blood pressure (mm Hg)", "sg": "urine specific gravity",
       "al": "urine albumin (0-5)", "su": "urine sugar (0-5)", "bgr": "random blood glucose (mg/dL)",
       "bu": "blood urea (mg/dL)", "sc": "serum creatinine (mg/dL)", "sod": "sodium (mEq/L)",
       "pot": "potassium (mEq/L)", "hemo": "hemoglobin (g/dL)", "pcv": "packed cell volume",
       "wc": "white blood cell count (cells/cumm)", "rc": "red blood cell count (millions/cmm)"}
BIN = {"rbc": ("red blood cells in urine", "abnormal", "normal"), "pc": ("pus cells", "abnormal", "normal"),
       "pcc": ("pus cell clumps", "present", "not present"), "ba": ("bacteria", "present", "not present"),
       "htn": ("hypertension", "yes", "no"), "dm": ("diabetes mellitus", "yes", "no"),
       "cad": ("coronary artery disease", "yes", "no"), "appet": ("appetite", "poor", "good"),
       "pe": ("pedal edema", "yes", "no"), "ane": ("anemia", "yes", "no")}
HEAD = "Based on these lab results and findings, does this patient have chronic kidney disease? Answer CKD or not CKD.\n"


def prompt(row):
    parts = []
    for c, label in NUM.items():
        parts.append(f"{label}: {'not recorded' if pd.isna(row[c]) else row[c]}")
    for c, (label, pos, neg) in BIN.items():
        parts.append(f"{label}: {'not recorded' if pd.isna(row[c]) else (pos if row[c] == 1 else neg)}")
    return HEAD + "\n".join(parts)


out = ROOT / "data" / "llm"
out.mkdir(parents=True, exist_ok=True)
for name in ("train", "val", "test"):
    df = pd.read_csv(ROOT / "data" / "processed" / f"{name}.csv")
    rows = [{"prompt": prompt(r), "completion": "CKD" if r["target"] == 1 else "not CKD"} for _, r in df.iterrows()]
    pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(name, len(rows))
