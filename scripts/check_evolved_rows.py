"""Row-by-row check of an Adaption output that has prompt, completion, enhanced_prompt, enhanced_completion.

    python scripts/check_evolved_rows.py data/llm/train_evolved.csv

Writes results/evolved_rows_check.json and prints one sample enhanced prompt.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
orig = pd.read_csv(ROOT / "data" / "llm" / "train.csv")
evo = pd.read_csv(sys.argv[1])
NUM = re.compile(r"\d+(?:\.\d+)?")
strip = lambda s: str(s).strip()  # noqa: E731

evo["prompt"] = evo["prompt"].map(strip)
orig["prompt"] = orig["prompt"].map(strip)
out = {"rows_original": len(orig), "rows_adapted": len(evo)}

# 1. which original rows are missing, and is the original text in the file untouched?
out["original_text_in_file_matches_train_csv"] = int(evo["prompt"].isin(set(orig["prompt"])).sum())
missing = orig[~orig["prompt"].isin(set(evo["prompt"]))]
out["original_rows_dropped"] = len(missing)
out["dropped_row_labels"] = missing["completion"].value_counts().to_dict()
out["duplicate_original_prompts_in_adapted_file"] = int(evo["prompt"].duplicated().sum())

# 2. labels: enhanced completion vs the original completion of the same row
out["enhanced_completion_not_exact_label"] = int((~evo["enhanced_completion"].map(strip).isin({"CKD", "not CKD"})).sum())
same = evo["completion"].map(strip) == evo["enhanced_completion"].map(strip)
out["enhanced_completion_differs_from_original_completion"] = int((~same).sum())
lab = dict(zip(orig["prompt"], orig["completion"].map(strip)))
chk = evo[evo["prompt"].isin(lab)]
out["adapted_label_differs_from_train_csv_label"] = int((chk["enhanced_completion"].map(strip) != chk["prompt"].map(lab)).sum())

# 3. lab values: every number in the original prompt must still be in the enhanced prompt, and none added
lost, added, nr_diff = [], [], []
for i, r in evo.iterrows():
    a, b = Counter(NUM.findall(r["prompt"])), Counter(NUM.findall(str(r["enhanced_prompt"])))
    if a - b:
        lost.append(i)
    if b - a:
        added.append(i)
    if r["prompt"].count("not recorded") != str(r["enhanced_prompt"]).count("not recorded"):
        nr_diff.append(i)
out["rows_where_an_original_number_is_missing_from_enhanced_prompt"] = len(lost)
out["rows_where_enhanced_prompt_has_extra_numbers"] = len(added)
out["rows_where_not_recorded_count_differs"] = len(nr_diff)
out["example_row_indexes_with_lost_numbers"] = lost[:5]

# 4. is the enhanced_prompt column actually filled in?
out["enhanced_prompt_empty_rows"] = int(evo["enhanced_prompt"].isna().sum())
flipped = evo[evo["completion"].map(strip) != evo["enhanced_completion"].map(strip)]
out["flipped_labels_original_to_enhanced"] = {f"{k[0]} -> {k[1]}": int(v) for k, v in flipped.groupby([flipped["completion"].map(strip), flipped["enhanced_completion"].map(strip)]).size().items()}

out["avg_words_original_prompt"] = round(evo["prompt"].str.split().str.len().mean(), 1)
ep = evo["enhanced_prompt"].dropna().astype(str)
out["avg_words_enhanced_prompt"] = round(ep.str.split().str.len().mean(), 1) if len(ep) else None
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "evolved_rows_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
print("\n--- sample enhanced_prompt (row 0, first 1500 chars) ---")
print(str(ep.iloc[0])[:1500] if len(ep) else "(enhanced_prompt is empty in every row)")
