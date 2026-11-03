"""Dose changes over time: run assess() on every visit and compare neighbouring visits."""
from datetime import date

import argparse

import numpy as np
import pandas as pd

from .agent import assess
from .dose_rules import list_drugs
from .features import ROOT

RAPID = -5.0  # KDIGO: eGFR falling by more than 5 mL/min/1.73 m2 per year = rapid progression


def _d(v):
    return date.fromisoformat(str(v["date"])[:10])  # ValueError on a bad date


def assess_history(visits, drugs, model=None):
    """visits: dicts like assess()'s patient plus 'date' (YYYY-MM-DD), each with that visit's creatinine/weight/age."""
    seen = []
    for v in sorted(visits, key=_d):
        r = assess(v, drugs, model)
        if r["egfr"] is not None:
            seen.append((_d(v), r))
    out = {"visits": [{"date": d.isoformat(), "egfr": r["egfr"], "crcl": r["crcl"], "stage": r["stage"]} for d, r in seen],
           "timeline": [{"date": d.isoformat(), "egfr": r["egfr"], "stage": r["stage"], "drug": x["drug"],
                         "dose": x.get("dose", x.get("message")), "action": x.get("action", "")}
                        for d, r in seen for x in r["drugs"]],
           "dose_changes": [], "stage_changes": [], "egfr_slope_per_year": None, "rapid_decline": False, "warnings": []}
    if len(seen) < len(visits):
        out["warnings"].append(f"{len(visits) - len(seen)} visit(s) skipped: creatinine, age or sex missing.")

    for (d0, a), (d1, b) in zip(seen, seen[1:]):
        if a["stage"] != b["stage"]:
            out["stage_changes"].append({"from_date": d0.isoformat(), "to_date": d1.isoformat(), "from": a["stage"], "to": b["stage"]})
        for x, y in zip(a["drugs"], b["drugs"]):  # same drug order in every visit
            if x["found"] and y["found"] and x["dose"] != y["dose"]:
                out["dose_changes"].append({"drug": y["drug"], "from_date": d0.isoformat(), "to_date": d1.isoformat(),
                                            "from_dose": x["dose"], "to_dose": y["dose"], "to_action": y["action"], "egfr": b["egfr"]})

    # ponytail: straight-line fit over all visits; noisy with few visits, use a mixed model if you need confidence intervals
    if len(seen) >= 2 and (seen[-1][0] - seen[0][0]).days >= 30:
        t = [(d - seen[0][0]).days / 365.25 for d, _ in seen]
        slope = float(np.polyfit(t, [r["egfr"] for _, r in seen], 1)[0])
        out["egfr_slope_per_year"] = round(slope, 1)
        out["rapid_decline"] = slope < RAPID
    if out["dose_changes"]:
        out["warnings"].append("Dose change suggested. Confirm with a repeat creatinine first: one reading can be skewed by acute illness or dehydration.")
    if out["rapid_decline"]:
        out["warnings"].append("eGFR is falling faster than 5 per year (rapid progression). Review with a clinician.")
    return out


def history_from_df(df, drugs, model=None):
    """One row per visit. Columns: date, age, sex, sc (+ weight, patient_id). Returns {patient_id: assess_history(...)}."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in ("date", "age", "sex", "sc") if c not in df]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if "patient_id" not in df:
        df["patient_id"] = "patient"
    return {str(pid): assess_history([{k: (None if pd.isna(x) else x) for k, x in row.items()}
                                      for row in g.drop(columns="patient_id").to_dict("records")], drugs, model)
            for pid, g in df.groupby("patient_id", sort=False)}


def main():
    ap = argparse.ArgumentParser(description="Dose needed at each visit, from a CSV of visits.")
    ap.add_argument("csv")
    ap.add_argument("--drugs", default=",".join(list_drugs()), help="comma-separated")
    ap.add_argument("--out", default=str(ROOT / "results" / "dose_timeline.csv"))
    a = ap.parse_args()
    res = history_from_df(pd.read_csv(a.csv), [d.strip() for d in a.drugs.split(",")])
    pd.DataFrame([{"patient_id": pid, **t} for pid, h in res.items() for t in h["timeline"]]).to_csv(a.out, index=False)
    for pid, h in res.items():
        print(f"== {pid}: {len(h['visits'])} visits, eGFR slope {h['egfr_slope_per_year']} per year" + (" (RAPID DECLINE)" if h["rapid_decline"] else ""))
        for c in h["dose_changes"]:
            print(f"   {c['from_date']} -> {c['to_date']}  {c['drug']}: {c['from_dose']}  ->  {c['to_dose']}")
        if not h["dose_changes"]:
            print("   no dose changes")
        for w in h["warnings"]:
            print("   WARNING:", w)
    print("full timeline written to", a.out)


if __name__ == "__main__":
    main()
