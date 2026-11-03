"""Demo UI.  streamlit run app.py"""
import pandas as pd
import streamlit as st

from src.agent import assess, explain
from src.cases import CASES
from src.dose_rules import list_drugs
from src.features import BINARY, NUMERIC, load_model
from src.history import assess_history, history_from_df

st.set_page_config(page_title="CKD dosing assistant", layout="wide")
st.title("CKD detection and renal dose adjustment")
st.warning("Prototype decision support, not a prescriber. Doses come from a draft rule table that has not been clinically validated.")


@st.cache_resource
def get_model():
    return load_model()


model = get_model()
if model is None:
    st.info("No trained model at models/model.joblib yet (run `python -m src.train`). eGFR and dosing still work.")

case = st.sidebar.selectbox("Example patient (made up)", list(CASES))
P = CASES[case]


def num(box, name, label, **kw):
    return box.number_input(label, value=float(P[name]), key=f"{case}-{name}", **kw)


a, b, c, d = st.columns(4)
patient = {
    "age": num(a, "age", "Age (years)", min_value=1.0, max_value=110.0),
    "sex": b.selectbox("Sex", ["M", "F"], index=["M", "F"].index(P["sex"]), key=f"{case}-sex"),
    "weight": num(c, "weight", "Weight (kg)", min_value=20.0, max_value=250.0),
    "sc": num(d, "sc", "Serum creatinine (mg/dL)", min_value=0.1, max_value=20.0, step=0.1),
}
with st.expander("Other values used by the CKD model"):
    cols = st.columns(4)
    for i, f in enumerate([f for f in NUMERIC if f not in ("age", "sc")]):
        patient[f] = num(cols[i % 4], f, f)
    st.caption("Tick = abnormal / present / yes / poor")
    for i, f in enumerate(BINARY):
        patient[f] = float(cols[i % 4].checkbox(f, value=bool(P[f]), key=f"{case}-{f}"))

drugs = st.multiselect("Drugs", list_drugs(), default=list_drugs())
r = assess(patient, drugs, model)

for w in r["warnings"]:
    st.error(w)
m = st.columns(4)
m[0].metric("CKD probability (model)", "n/a" if r["p_ckd_pct"] is None else f"{r['p_ckd_pct']}%")
m[1].metric("eGFR (CKD-EPI 2021)", r["egfr"] if r["egfr"] is not None else "n/a")
m[2].metric("CrCl (Cockcroft-Gault)", r["crcl"] if r["crcl"] is not None else "n/a")
m[3].metric("Stage", r["stage"] or "n/a")

def show_table(rows, index=None):
    """Static table: long dose text wraps onto several lines instead of being cut off."""
    df = pd.DataFrame(rows).fillna("").astype(str)
    if index and index in df.columns:
        df = df.set_index(index)
    else:
        df.index = range(1, len(df) + 1)
    st.table(df)


if r["drugs"]:
    show_table([{"Drug": x["drug"], "Standard dose": x.get("standard", ""), "Adjusted dose": x.get("dose", x.get("message")),
                 "Action": x.get("action", ""), "Note": x.get("note", ""), "Based on": x.get("basis", "")}
                for x in r["drugs"]], index="Drug")
st.subheader("Explanation")
st.write(explain(r))
st.caption(r["disclaimer"])

st.divider()
st.subheader("Changes over time")
st.caption("Upload a visits CSV (columns: date, age, sex, sc; optional weight, patient_id), or edit the table below "
           "(dates as YYYY-MM-DD; sex and other values then come from the patient above).")
start = pd.DataFrame([{"date": d, "age": patient["age"], "weight": patient["weight"], "sc": round(patient["sc"] * f, 2)}
                      for d, f in (("2025-11-03", 0.7), ("2026-05-03", 0.85), ("2026-11-03", 1.0))])
up = st.file_uploader("Visits CSV", type="csv")
try:
    if up is not None:
        results = history_from_df(pd.read_csv(up), drugs)
        h = results[st.selectbox("Patient", list(results))]
    else:
        edited = st.data_editor(start, num_rows="dynamic", key=f"hist-{case}", hide_index=True)
        rows = [{**patient, **{k: (None if pd.isna(x) else x) for k, x in v.items()}}
                for v in edited.dropna(subset=["date", "sc", "age"]).to_dict("records")]
        h = assess_history(rows, drugs)
except (ValueError, KeyError) as e:
    st.error(f"Check the visit data: {e}")
else:
    c = st.columns(3)
    c[0].metric("eGFR trend (per year)", "n/a" if h["egfr_slope_per_year"] is None else h["egfr_slope_per_year"])
    c[1].metric("Rapid decline", "yes" if h["rapid_decline"] else "no")
    c[2].metric("Dose changes", len(h["dose_changes"]))
    for w in h["warnings"]:
        st.error(w)
    if h["visits"]:
        st.line_chart(pd.DataFrame(h["visits"]).set_index("date")["egfr"])
    if h["timeline"]:
        st.markdown("**Dose needed at each visit**")
        show_table(pd.DataFrame(h["timeline"]).pivot_table(index="drug", columns="date", values="dose", aggfunc="first").reset_index(), index="drug")
    if h["dose_changes"]:
        st.markdown("**Dose changes**")
        show_table(h["dose_changes"])
    else:
        st.success("No dose changes across these visits.")
