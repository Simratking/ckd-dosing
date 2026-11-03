import pytest

import pandas as pd

from src.features import ROOT
from src.history import assess_history, history_from_df


def test_csv_with_two_patients():
    res = history_from_df(pd.read_csv(ROOT / "examples" / "visits.csv"), ["metformin", "gabapentin"])
    assert list(res) == ["P1", "P2"]
    p1, p2 = res["P1"], res["P2"]
    assert len(p1["timeline"]) == 4 * 2 and {t["drug"] for t in p1["timeline"]} == {"metformin", "gabapentin"}
    assert any(c["drug"] == "metformin" for c in p1["dose_changes"]) and p1["egfr_slope_per_year"] < 0
    assert p2["dose_changes"] == [] and p2["stage_changes"] == []


def test_csv_missing_columns_raises():
    with pytest.raises(ValueError, match="missing columns"):
        history_from_df(pd.DataFrame({"date": ["2025-01-01"], "sc": [1.0]}), ["metformin"])

DRUGS = ["metformin", "ibuprofen", "gabapentin"]
BASE = dict(age=60, sex="M", weight=80)


def _visits(*pairs):
    return [dict(BASE, date=d, sc=sc) for d, sc in pairs]


def test_worsening_kidneys_trigger_dose_changes_and_rapid_decline():
    h = assess_history(_visits(("2026-01-01", 2.4), ("2025-01-01", 1.0), ("2025-07-01", 1.6)), DRUGS)  # unsorted on purpose
    assert [v["stage"] for v in h["visits"]] == ["G2", "G3a", "G3b"]
    met = [c for c in h["dose_changes"] if c["drug"] == "metformin"]
    assert len(met) == 1 and met[0]["from_dose"] == "Usual dose" and met[0]["to_action"] == "reduce"
    assert met[0]["from_date"] == "2025-07-01" and met[0]["to_date"] == "2026-01-01"
    assert h["egfr_slope_per_year"] < -5 and h["rapid_decline"]
    assert len(h["stage_changes"]) == 2
    assert any("repeat creatinine" in w for w in h["warnings"])


def test_stable_patient_has_no_changes():
    h = assess_history(_visits(("2025-01-01", 1.0), ("2025-07-01", 1.0), ("2026-01-01", 1.0)), DRUGS)
    assert h["dose_changes"] == [] and h["stage_changes"] == [] and not h["rapid_decline"] and h["warnings"] == []
    assert h["egfr_slope_per_year"] == pytest.approx(0, abs=0.1)


def test_single_visit_has_no_trend():
    h = assess_history(_visits(("2025-01-01", 1.0)), DRUGS)
    assert h["egfr_slope_per_year"] is None and h["dose_changes"] == []


def test_visit_without_creatinine_is_skipped_not_fatal():
    visits = _visits(("2025-01-01", 1.0), ("2025-07-01", None), ("2026-01-01", 1.0))
    h = assess_history(visits, DRUGS)
    assert len(h["visits"]) == 2 and "skipped" in h["warnings"][0]


def test_bad_date_raises():
    with pytest.raises(ValueError):
        assess_history(_visits(("not-a-date", 1.0)), DRUGS)
