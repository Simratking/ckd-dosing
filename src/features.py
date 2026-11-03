"""Shared schema, cleaning for the UCI / Kaggle CKD data, and Model A inference."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"

NUMERIC = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
# text column -> the value that becomes 1.0 (abnormal / present / yes / poor)
BINARY = {"rbc": "abnormal", "pc": "abnormal", "pcc": "present", "ba": "present",
          "htn": "yes", "dm": "yes", "cad": "yes", "appet": "poor", "pe": "yes", "ane": "yes"}
FEATURES = NUMERIC + list(BINARY)
TARGET = "target"  # 1 = ckd, 0 = notckd


def clean(df):
    """Raw UCI or Kaggle CKD table -> numeric FEATURES + TARGET. Missing values stay NaN."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"wbcc": "wc", "rbcc": "rc"})  # UCI names vs Kaggle names
    label = "classification" if "classification" in df else "class"
    need = FEATURES + [label]
    if any(c not in df for c in need):
        raise ValueError(f"missing columns: {[c for c in need if c not in df]}")
    df = df[need].astype(object)
    df = df.apply(lambda s: s.map(lambda v: v.strip() if isinstance(v, str) else v))  # stray tabs/spaces
    df = df.mask(df.isin(["?", ""]))
    out = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in NUMERIC})
    for c, pos in BINARY.items():
        out[c] = df[c].map(lambda v: float(v.lower() == pos) if isinstance(v, str) else np.nan)
    out[TARGET] = df[label].map(lambda v: {"ckd": 1, "notckd": 0}.get(v.lower()) if isinstance(v, str) else None)
    return out.dropna(subset=[TARGET]).astype({TARGET: int})


def load_split(name):
    """name in {'train','val','test'} -> (X, y)."""
    df = pd.read_csv(ROOT / "data" / "processed" / f"{name}.csv")
    return df[FEATURES], df[TARGET]


def load_model(path=MODEL_PATH):
    return joblib.load(path) if Path(path).exists() else None


def ckd_probability(model, patient):
    """Probability of CKD from a dict of feature values (missing -> NaN, the pipeline imputes)."""
    row = pd.DataFrame([{f: np.nan if patient.get(f) is None else patient[f] for f in FEATURES}])
    return float(model.predict_proba(row)[0, 1])
