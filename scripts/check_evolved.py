"""Compare an Adaption-adapted dataset with the original train.csv.

    python scripts/check_evolved.py data/llm/train_evolved.csv

Checks: row counts, duplicate prompts in the original, allowed completion values,
whether every adapted prompt is identical to an original prompt, and whether the
label for each prompt is unchanged. Prints a summary; writes results/evolved_check.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {"CKD", "not CKD"}

orig = pd.read_csv(ROOT / "data" / "llm" / "train.csv")
evo = pd.read_csv(sys.argv[1])
print("adapted file columns:", list(evo.columns))


def pick(words, avoid=()):
    cols = [c for c in evo.columns if all(w in c.lower() for w in words[:1])]
    for pref in ("enhanced", "adapt", "evolved", "new"):
        for c in cols:
            if pref in c.lower() and not any(a in c.lower() for a in avoid):
                return c
    return cols[0] if cols else None


comp_col = pick(["completion"])
prompt_col = pick(["prompt"])
print("using completion column:", comp_col, "| prompt column:", prompt_col)
if comp_col is None or prompt_col is None:
    sys.exit("Could not find prompt/completion columns - send me the column list above.")

dup = int(orig.prompt.duplicated().sum())
out = {"original_rows": len(orig), "adapted_rows": len(evo), "duplicate_prompts_in_original": dup}
comp = evo[comp_col].astype(str).str.strip()
out["completions_not_exact_CKD_or_not_CKD"] = int((~comp.isin(ALLOWED)).sum())
out["bad_completion_examples"] = comp[~comp.isin(ALLOWED)].head(5).tolist()
lookup = dict(zip(orig.prompt.str.strip(), orig.completion.str.strip()))
p = evo[prompt_col].astype(str).str.strip()
known = p.isin(lookup)
out["adapted_prompts_identical_to_an_original"] = int(known.sum())
out["adapted_prompts_changed_or_unknown"] = int((~known).sum())
out["labels_changed_vs_original"] = int((p[known].map(lookup) != comp[known]).sum())
out["original_prompts_missing_from_adapted"] = int((~orig.prompt.str.strip().isin(set(p))).sum())
Path(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "evolved_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
