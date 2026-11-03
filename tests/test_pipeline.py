"""Cleaning + agent checks on synthetic data (smoke test only, no metrics are meaningful)."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.agent import assess, explain, numbers_ok
from src.cases import CASES
from src.features import FEATURES, TARGET, clean
from src.synthetic import make_synthetic


def _data():
    df = clean(make_synthetic())
    return df[FEATURES], df[TARGET]


def test_clean_handles_messy_values():
    X, y = _data()
    assert list(X.columns) == FEATURES and set(y) == {0, 1} and len(X) == 400
    assert X.dtypes.map(lambda t: np.issubdtype(t, np.number)).all()
    assert 0.03 < X.isna().mean().mean() < 0.2  # '?' became NaN, nothing else did


def test_uci_column_names_are_accepted():
    df = make_synthetic().rename(columns={"wc": "wbcc", "rc": "rbcc", "classification": "class"})
    assert len(clean(df)) == 400


def test_agent_end_to_end():
    X, y = _data()
    model = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(max_iter=500)).fit(X, y)
    r = assess(CASES["CKD stage G5"], ["metformin", "gabapentin", "nitrofurantoin", "nobody"], model)
    assert r["stage"] == "G5" and 0 <= r["p_ckd_pct"] <= 100
    assert r["drugs"][0]["action"] == "avoid" and r["drugs"][1]["action"] == "reduce"
    assert any("no rule on file" in w for w in r["warnings"])
    assert "Decision support only" in explain(r)


def test_missing_creatinine_does_not_crash():
    r = assess(dict(CASES["Healthy adult (eGFR >= 90)"], sc=None), ["metformin"])
    assert r["egfr"] is None and r["drugs"] == [] and "creatinine" in r["warnings"][0]


def test_three_demo_cases_hit_their_stages():
    stages = [assess(c, [])["stage"] for c in CASES.values()]
    assert stages == ["G1", "G3b", "G5"]


def test_llm_text_with_invented_number_is_rejected():
    r = assess(CASES["CKD stage G3b"], ["metformin"])
    assert numbers_ok(f"eGFR is {r['egfr']}", r)
    assert not numbers_ok("Give 750 mg", r)
