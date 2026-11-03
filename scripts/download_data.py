"""Download the real UCI CKD dataset (id 336) into data/raw/ckd.csv.

    pip install ucimlrepo
    python scripts/download_data.py

If this fails (network), download the CSV from Kaggle ("Chronic Kidney Disease dataset")
or UCI and save it as data/raw/ckd.csv. Both formats are handled by src/features.py.
"""
from pathlib import Path

from ucimlrepo import fetch_ucirepo

root = Path(__file__).resolve().parent.parent
d = fetch_ucirepo(id=336)
df = d.data.features.copy()
df["classification"] = d.data.targets.iloc[:, 0]
(root / "data" / "raw").mkdir(parents=True, exist_ok=True)
df.to_csv(root / "data" / "raw" / "ckd.csv", index=False)
print("saved", df.shape, "-> data/raw/ckd.csv")
