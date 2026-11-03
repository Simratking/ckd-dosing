"""Kidney-function formulas. Units: creatinine mg/dL, weight kg, age years, sex 'M' or 'F'.

Stage here is the G-category from eGFR alone. A real CKD diagnosis also needs
persistence over 3 months and/or markers of kidney damage.
"""


def _female(sex):
    return sex.strip().upper().startswith("F")


def ckd_epi_2021(scr, age, sex):
    """CKD-EPI 2021 creatinine equation (race-free). Returns mL/min/1.73 m2."""
    female = _female(sex)
    k, a = (0.7, -0.241) if female else (0.9, -0.302)
    r = scr / k
    return 142 * min(r, 1) ** a * max(r, 1) ** -1.2 * 0.9938 ** age * (1.012 if female else 1)


def cockcroft_gault(scr, age, weight, sex):
    """Creatinine clearance in mL/min (what most drug labels use)."""
    crcl = (140 - age) * weight / (72 * scr)
    return crcl * 0.85 if _female(sex) else crcl


STAGES = [(90, "G1"), (60, "G2"), (45, "G3a"), (30, "G3b"), (15, "G4"), (0, "G5")]


def stage(egfr):
    return next(name for lo, name in STAGES if egfr >= lo)
