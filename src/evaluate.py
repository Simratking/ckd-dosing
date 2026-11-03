"""Final evaluation on the held-out test set, plus robustness checks. Run this ONCE.

    python -m src.evaluate                       # uses models/model.joblib
    python -m src.evaluate --model path.joblib   # another sklearn-compatible model
    python -m src.evaluate --proba-csv p.csv     # model can't be loaded: CSV of test-set probabilities

Writes results/metrics.json, confusion_matrix.png, calibration.png. Refuses to run twice
unless --force, so the test set is not tuned against.
"""
import argparse
import json
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import CalibrationDisplay  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, brier_score_loss,  # noqa: E402
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402

from .features import MODEL_PATH, ROOT, load_split  # noqa: E402

RES = ROOT / "results"


def score(y, p, thr=0.5):
    p = np.asarray(p)
    pred = (p >= thr).astype(int)
    return {"f1": f1_score(y, pred), "recall": recall_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "confusion_matrix": confusion_matrix(y, pred).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--proba-csv")
    ap.add_argument("--drop", default="sc", help="feature removed in the ablation")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if (RES / "metrics.json").exists() and not a.force:
        sys.exit("Test set already evaluated (results/metrics.json exists). Use --force to rerun.")
    Xte, yte = load_split("test")
    model = None
    if a.proba_csv:
        p = pd.read_csv(a.proba_csv).iloc[:, -1].to_numpy()
        assert len(p) == len(yte), f"expected {len(yte)} probabilities, got {len(p)}"
    else:
        model = joblib.load(a.model)
        p = model.predict_proba(Xte)[:, 1]
    out = {"split": "test", "n": len(yte), "test": score(yte, p)}

    RES.mkdir(exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(yte, (p >= 0.5).astype(int), display_labels=["not CKD", "CKD"])
    plt.savefig(RES / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    CalibrationDisplay.from_predictions(yte, p, n_bins=5)  # few bins: the test set is small
    plt.savefig(RES / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close()

    if model is not None:
        Xtr, ytr = load_split("train")
        Xva, yva = load_split("val")
        Xall, yall = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
        try:  # 1. ablation: retrain without a key lab
            m = clone(model).fit(Xtr.drop(columns=a.drop), ytr)
            out["ablation"] = {"dropped": a.drop, "test": score(yte, m.predict_proba(Xte.drop(columns=a.drop))[:, 1])}
        except Exception as e:  # e.g. model type that sklearn can't clone
            out["ablation"] = {"skipped": str(e)}
        rng = np.random.default_rng(42)  # 2. mask random cells, 5 repeats each
        out["masking"] = {}
        for frac in (0.1, 0.2, 0.3):
            runs = [score(yte, model.predict_proba(Xte.mask(rng.random(Xte.shape) < frac))[:, 1]) for _ in range(5)]
            out["masking"][str(frac)] = {k: float(np.mean([r[k] for r in runs])) for k in ("f1", "recall", "roc_auc")}
        try:  # 3. 5-fold CV on train+val (test stays untouched)
            cv = cross_validate(clone(model), Xall, yall, scoring=["f1", "recall", "roc_auc"],
                                cv=StratifiedKFold(5, shuffle=True, random_state=42))
            out["cv"] = {k: {"mean": float(cv[f"test_{k}"].mean()), "std": float(cv[f"test_{k}"].std())}
                         for k in ("f1", "recall", "roc_auc")}
        except Exception as e:
            out["cv"] = {"skipped": str(e)}

    (RES / "metrics.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "test"} | {"test": {k: v for k, v in out["test"].items() if k != "confusion_matrix"}}, indent=2, default=float))


if __name__ == "__main__":
    main()
