"""Demo UI.  streamlit run app.py"""
import streamlit as st

from src.agent import assess, explain
from src.cases import CASES
from src.dose_rules import list_drugs
from src.features import BINARY, NUMERIC, load_model

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

if r["drugs"]:
    st.dataframe([{"Drug": x["drug"], "Standard dose": x.get("standard", ""), "Adjusted dose": x.get("dose", x.get("message")),
                   "Action": x.get("action", ""), "Note": x.get("note", ""), "Based on": x.get("basis", "")}
                  for x in r["drugs"]], use_container_width=True, hide_index=True)
st.subheader("Explanation")
st.write(explain(r))
st.caption(r["disclaimer"])
