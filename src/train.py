"""Baselines. Imputation lives inside each Pipeline, so it is fitted on train only.

    python -m src.train

Scores on the validation set, saves results/baseline.json, and saves the best baseline to
models/model.joblib so the demo works before the AutoScientist model is ready. To use the
AutoScientist model instead, overwrite models/model.joblib with its export (see README).
"""
import json

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import score
from .features import MODEL_PATH, ROOT, load_split


def candidates():
    imp = lambda: SimpleImputer(strategy="median")  # noqa: E731
    return {
        "logreg": make_pipeline(imp(), StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": make_pipeline(imp(), RandomForestClassifier(300, random_state=42)),
        "gradient_boosting": make_pipeline(imp(), GradientBoostingClassifier(random_state=42)),
    }


def main():
    Xtr, ytr = load_split("train")
    Xva, yva = load_split("val")
    results, fitted = {}, {}
    for name, model in candidates().items():
        fitted[name] = model.fit(Xtr, ytr)
        results[name] = score(yva, model.predict_proba(Xva)[:, 1])
        print(f"{name:18s} val recall={results[name]['recall']:.3f} f1={results[name]['f1']:.3f} auc={results[name]['roc_auc']:.3f}")
    best = max(results, key=lambda n: (results[n]["recall"], results[n]["f1"], results[n]["roc_auc"]))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], MODEL_PATH)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "baseline.json").write_text(
        json.dumps({"split": "val", "best": best, "models": results}, indent=2, default=float), encoding="utf-8")
    print("best baseline:", best, "-> saved to", MODEL_PATH)


if __name__ == "__main__":
    main()
