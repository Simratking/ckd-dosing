"""SYNTHETIC data in the messy Kaggle CKD format. For smoke tests ONLY.

Never report results from this data: it is invented and far easier to separate than real patients.
"""
import numpy as np
import pandas as pd

from .features import BINARY, NUMERIC

# name: (mean if ckd, mean if not, sd)
_NUM = {"age": (58, 46, 14), "bp": (78, 72, 10), "sg": (1.012, 1.021, 0.004), "al": (2.0, 0.1, 1.2),
        "su": (0.8, 0.1, 1.0), "bgr": (150, 110, 45), "bu": (60, 28, 25), "sc": (4.0, 1.0, 2.5),
        "sod": (136, 141, 5), "pot": (4.7, 4.3, 0.8), "hemo": (11, 14.5, 2), "pcv": (34, 44, 6),
        "wc": (8500, 7800, 2000), "rc": (3.9, 5.0, 0.7)}
_P = {"rbc": (.5, .02), "pc": (.5, .03), "pcc": (.2, .01), "ba": (.1, .01), "htn": (.6, .02),
      "dm": (.45, .03), "cad": (.1, .01), "appet": (.25, .02), "pe": (.3, .01), "ane": (.25, .01)}
_NEG = {"rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent", "htn": "no", "dm": "no",
        "cad": "no", "appet": "good", "pe": "no", "ane": "no"}


def make_synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    ckd = rng.random(n) < 0.62
    df = pd.DataFrame({"id": range(n)})
    for c in NUMERIC:
        m1, m0, sd = _NUM[c]
        v = rng.normal(np.where(ckd, m1, m0), sd).clip(min=0.4 if c == "sc" else 0)
        df[c] = v.round(3 if c == "sg" else 1)
    for c, pos in BINARY.items():
        p1, p0 = _P[c]
        df[c] = np.where(rng.random(n) < np.where(ckd, p1, p0), pos, _NEG[c])
    df["classification"] = np.where(ckd, "ckd", "notckd")
    df = df.astype(object)
    for c in NUMERIC + list(BINARY):  # mimic the real file: '?' for missing, stray tabs/spaces
        df.loc[rng.random(n) < 0.08, c] = "?"
        pad = (rng.random(n) < 0.05) & (df[c] != "?")
        df.loc[pad, c] = "\t" + df.loc[pad, c].astype(str)
    df.loc[rng.random(n) < 0.1, "classification"] = "ckd\t"
    df.loc[df["classification"] == "notckd", "classification"] = "notckd"
    return df.rename(columns={"wc": "wc", "rc": "rc"})
