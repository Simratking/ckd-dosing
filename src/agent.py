"""Dosing assistant.

Orchestration is deterministic, so every dose comes from rules/dose_rules.json. An LLM may
optionally rephrase the result, but its text is rejected (and the template is used) if it
contains any number that is not in the result. No API key needed for the app to work.
"""
import json
import os
import re

from .dose_rules import check_contraindication, lookup_dose_rule
from .egfr import cockcroft_gault, ckd_epi_2021, stage
from .features import ckd_probability

DISCLAIMER = "Decision support only. Confirm with a clinician or pharmacist."
NUM = re.compile(r"\d+(?:\.\d+)?")


def assess(patient, drugs, model=None):
    """patient: dict with age, sex ('M'/'F'), weight (kg), sc (mg/dL) + optional model features."""
    out = {"p_ckd_pct": None, "egfr": None, "crcl": None, "stage": None,
           "warnings": [], "drugs": [], "disclaimer": DISCLAIMER}
    if model is not None:
        out["p_ckd_pct"] = round(100 * ckd_probability(model, patient), 1)
    sc, age, sex, wt = (patient.get(k) for k in ("sc", "age", "sex", "weight"))
    if not (sc and age and sex):
        out["warnings"].append("Serum creatinine, age and sex are needed to estimate kidney function, so no doses were adjusted.")
        return out
    egfr = round(ckd_epi_2021(sc, age, sex), 1)  # rounded once, so shown value = value used
    crcl = round(cockcroft_gault(sc, age, wt, sex), 1) if wt else None
    out.update(egfr=egfr, crcl=crcl, stage=stage(egfr))
    for d in drugs:
        r = lookup_dose_rule(d, egfr, crcl)
        out["drugs"].append(r)
        warn = check_contraindication(d, egfr, crcl) if r["found"] else f"{d}: no rule on file."
        if warn:
            out["warnings"].append(warn)
    return out


def _template(r):
    lines = []
    if r["p_ckd_pct"] is not None:
        lines.append(f"The model estimates a {r['p_ckd_pct']}% probability of CKD.")
    if r["egfr"] is None:
        return " ".join(lines + r["warnings"] + [r["disclaimer"]])
    lines.append(f"Estimated eGFR is {r['egfr']} (stage {r['stage']})" + (f" and CrCl is {r['crcl']} mL/min." if r["crcl"] else "."))
    for d in r["drugs"]:
        if d["found"]:
            lines.append(f"{d['drug'].capitalize()}: {d['dose']} (standard: {d['standard']}). Based on {d['basis']}." + (f" {d['note']}." if d["note"] else ""))
        else:
            lines.append(f"{d['drug']}: no rule on file.")
    return "\n".join(lines + [r["disclaimer"]])


def numbers_ok(text, result):
    """True if every number in text also appears in the result (no invented figures)."""
    allowed = set(NUM.findall(json.dumps(result)))
    return all(n in allowed for n in NUM.findall(text))


def explain(result):
    """Plain-language explanation. Uses an LLM only if ANTHROPIC_API_KEY is set and the output passes the number check."""
    text = _template(result)
    if os.getenv("ANTHROPIC_API_KEY") and result["egfr"] is not None:
        try:
            import anthropic
            prompt = ("Explain this renal dosing result to a clinician in under 150 words. Use ONLY the JSON. "
                      "Copy every number exactly as written. Do not add drugs, doses or advice. End with the disclaimer.\n"
                      + json.dumps(result))
            msg = anthropic.Anthropic().messages.create(
                model=os.getenv("CKD_LLM_MODEL", "claude-sonnet-5-5"), max_tokens=600,
                messages=[{"role": "user", "content": prompt}])
            llm = msg.content[0].text
            if numbers_ok(llm, result):
                return llm
        except Exception:
            pass  # fall back to the template
    return text
