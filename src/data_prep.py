"""Clean the raw CKD CSV and write a stratified 70/15/15 split.

    python -m src.data_prep --raw data/raw/ckd.csv
"""
import argparse
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from .features import ROOT, TARGET, clean

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw" / "ckd.csv"))
    a = ap.parse_args()

    df = clean(pd.read_csv(a.raw, dtype=str, keep_default_na=False))
    train, rest = train_test_split(df, test_size=0.30, stratify=df[TARGET], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest[TARGET], random_state=SEED)

    out, res = ROOT / "data" / "processed", ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(out / f"{name}.csv", index=False)

    miss = df.drop(columns=TARGET).isna().sum()
    pd.DataFrame({"missing": miss, "pct": (100 * miss / len(df)).round(1)}).to_csv(res / "missing_values.csv")
    summary = {"rows": len(df), "features": df.shape[1] - 1,
               "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
               "ckd_counts": {n: int(p[TARGET].sum()) for n, p in (("train", train), ("val", val), ("test", test))},
               "missing_cells_pct": round(100 * float(miss.sum()) / (len(df) * (df.shape[1] - 1)), 1)}
    (res / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
