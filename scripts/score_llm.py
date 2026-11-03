"""Score a fine-tuned LLM's "CKD" / "not CKD" answers on the held-out test set, with bootstrap 95% CIs.

    python scripts/score_llm.py outputs.csv               # score the LLM
    python scripts/score_llm.py outputs.csv --baseline    # also score models/model.joblib and test the difference

outputs.csv holds one model answer per test prompt, in a column named prediction / output / response /
completion (otherwise the last column is used). If the file also has a 'prompt' column, rows are matched to
data/llm/test.csv by prompt; if not, they must be in the same order (60 rows).

An answer counts as "not CKD" or "CKD" if it starts with that text (case and punctuation ignored, anything the
model keeps writing after it is ignored). Anything else is reported as unparseable and counted as "not CKD",
which is the safe way to count it for recall. Run this once, on the test set: it is the final number.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse(text):
    t = re.sub(r"[^a-z ]", "", str(text).lower()).strip()
    if t.startswith("not ckd"):
        return 0
    if t.startswith("ckd"):
        return 1
    return None


def counts(y, p):
    """tp, fp, fn along the last axis (works on one sample or a stack of bootstrap resamples)."""
    return (y * p).sum(-1), ((1 - y) * p).sum(-1), (y * (1 - p)).sum(-1)


def metrics(y, p):
    tp, fp, fn = (np.asarray(x, dtype=float) for x in counts(y, p))
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"recall": tp / (tp + fn), "precision": np.where(tp + fp > 0, tp / (tp + fp), 0.0), "f1": 2 * tp / (2 * tp + fp + fn)}


def ci(a):
    lo, hi = np.nanpercentile(a, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs")
    ap.add_argument("--baseline", action="store_true", help="also score models/model.joblib on data/processed/test.csv")
    ap.add_argument("--resamples", type=int, default=2000)
    a = ap.parse_args()

    test = pd.read_csv(ROOT / "data" / "llm" / "test.csv")
    y = (test["completion"].str.strip().str.lower() == "ckd").astype(int).to_numpy()
    out = pd.read_csv(a.outputs)
    col = next((c for c in ("prediction", "output", "response", "completion") if c in out), out.columns[-1])
    if "prompt" in out:
        truth = dict(zip(test["prompt"].str.strip(), y))
        missing = [p for p in out["prompt"].str.strip() if p not in truth]
        if missing or len(out) != len(test):
            sys.exit(f"prompts do not match data/llm/test.csv ({len(missing)} unknown, {len(out)} vs {len(test)} rows)")
        order = {p: i for i, p in enumerate(test["prompt"].str.strip())}
        out = out.assign(_i=out["prompt"].str.strip().map(order)).sort_values("_i")
    elif len(out) != len(test):
        sys.exit(f"expected {len(test)} answers, got {len(out)}")
    parsed = [parse(t) for t in out[col]]
    unparseable = sum(p is None for p in parsed)
    pred = np.array([p or 0 for p in parsed])

    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(y), (a.resamples, len(y)))
    boot = {"llm": metrics(y[idx], pred[idx])}
    point = {"llm": metrics(y, pred)}
    cm = {"llm": [[int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())],
                  [int(((y == 1) & (pred == 0)).sum()), int(((y == 1) & (pred == 1)).sum())]]}

    if a.baseline:
        import joblib
        from src.features import MODEL_PATH, load_split
        X, yb = load_split("test")
        assert (yb.to_numpy() == y).all(), "data/processed/test.csv rows do not line up with data/llm/test.csv"
        base = (joblib.load(MODEL_PATH).predict_proba(X)[:, 1] >= 0.5).astype(int)
        boot["baseline"], point["baseline"] = metrics(y[idx], base[idx]), metrics(y, base)
        cm["baseline"] = [[int(((y == 0) & (base == 0)).sum()), int(((y == 0) & (base == 1)).sum())],
                          [int(((y == 1) & (base == 0)).sum()), int(((y == 1) & (base == 1)).sum())]]

    result = {"n": int(len(y)), "unparseable_answers": unparseable, "bootstrap_resamples": a.resamples}
    for name in boot:
        result[name] = {k: {"value": round(float(point[name][k]), 3), "ci95": ci(boot[name][k])} for k in ("recall", "precision", "f1")}
        result[name]["confusion_matrix"] = cm[name]
    if a.baseline:
        d = boot["llm"]["f1"] - boot["baseline"]["f1"]
        result["f1_llm_minus_baseline"] = {"value": round(float(point["llm"]["f1"] - point["baseline"]["f1"]), 3), "ci95": ci(d)}
        result["note"] = "If the CI of the difference contains 0, the two models are not distinguishable on this test set."

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "llm_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
