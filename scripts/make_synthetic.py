"""Write INVENTED data to data/raw/synthetic_ckd.csv to smoke-test the pipeline offline.

    python scripts/make_synthetic.py
    python -m src.data_prep --raw data/raw/synthetic_ckd.csv

Never report results from it. Delete data/processed, models and results before the real run.
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from src.synthetic import make_synthetic  # noqa: E402

(root / "data" / "raw").mkdir(parents=True, exist_ok=True)
make_synthetic().to_csv(root / "data" / "raw" / "synthetic_ckd.csv", index=False)
print("wrote data/raw/synthetic_ckd.csv (SYNTHETIC)")
