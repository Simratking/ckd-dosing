"""The two agent tools. Every dose the system shows comes from rules/dose_rules.json."""
import json

from .features import ROOT

_DATA = json.loads((ROOT / "rules" / "dose_rules.json").read_text(encoding="utf-8"))
RULES = _DATA["drugs"]
_ALIAS = {a: key for key, d in RULES.items() for a in d["aliases"]}


def list_drugs():
    return list(RULES)


def lookup_dose_rule(drug, egfr, crcl=None):
    """Return the standard dose and the adjusted dose for this kidney function."""
    key = _ALIAS.get(drug.strip().lower())
    if key is None:
        return {"drug": drug, "found": False, "message": "no rule on file"}
    d = RULES[key]
    use_crcl = d["metric"] == "crcl" and crcl is not None
    value = crcl if use_crcl else egfr
    basis = "CrCl" if use_crcl else "eGFR"
    if d["metric"] == "crcl" and not use_crcl:
        basis = "eGFR (CrCl unavailable: weight missing)"
    for b in d["bands"]:
        if value >= b["min"] and (b["max"] is None or value < b["max"]):
            return {"drug": key, "found": True, "standard": d["standard"], "action": b["action"],
                    "dose": b["dose"], "note": b["note"], "basis": basis, "value": value,
                    "source": d["source"]}
    return {"drug": key, "found": False, "message": "no rule on file for this value"}


def check_contraindication(drug, egfr, crcl=None):
    """Return a warning string if the drug should be avoided, else None."""
    r = lookup_dose_rule(drug, egfr, crcl)
    if r.get("action") == "avoid":
        return f"{r['drug']}: avoid at this kidney function. {r['note']}".strip()
    return None
