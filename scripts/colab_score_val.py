"""Google Colab code used to score the fine-tuned AutoScientist model on the VALIDATION set.

This is a record of the notebook cells, in the order they were used. It is meant to be run in Colab
(T4 GPU), not locally: it expects /content/adapter.zip (the adapter downloaded from AutoScientist,
the same files as adapter/ in this project) and /content/val.csv (data/llm/val.csv) to be uploaded first.

It produced these files in results/:
  autoscientist_val.json, autoscientist_val_predictions.csv   (cells 1 and 2)
  autoscientist_margin_auc.json, autoscientist_margins.csv    (cells 3 and 4)

Notes:
- Only the validation set was used here. The test set was not scored with this code.
- In cell 2, an answer that is not parsed as CKD / not CKD is counted as WRONG (the opposite of the true
  label). scripts/score_llm.py instead counts it as "not CKD". No answer was unparseable in this run
  ("unparseable_outputs": 0), so both rules give the same result here.
- The margin in cell 3 compares the summed token log-probabilities of the two answers, which have
  different lengths, so its zero point is not a calibrated threshold. Only its ranking (ROC-AUC) is used.
"""

# ---------------------------------------------------------------- cell 1: load model, generate answers
# Notebook shell lines, run first in Colab:
#   !pip uninstall -y -q torchao
#   !pip install -q peft accelerate
#   !unzip -q -o /content/adapter.zip -d /content

import json, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix

tok = AutoTokenizer.from_pretrained("/content/adapter")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Base", torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, "/content/adapter")
model.eval()

val = pd.read_csv("/content/val.csv")
print(len(val), "rows; columns:", list(val.columns))


def build(prompt):
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


print(build(val["prompt"][0])[-120:])   # check the prompt ends with the assistant/think tags

outs = []
for p in val["prompt"]:
    ids = tok(build(p), return_tensors="pt").to(model.device)
    with torch.no_grad():
        o = model.generate(**ids, max_new_tokens=8, do_sample=False)
    outs.append(tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())

val["raw_output"] = outs
print(val["raw_output"].value_counts())

# ---------------------------------------------------------------- cell 2: score the answers, save results
def parse(t):
    t = str(t).strip().lower()
    if t.startswith("not ckd"): return 0
    if t.startswith("ckd"): return 1
    return None


val["true"] = val["completion"].apply(parse)
val["pred"] = val["raw_output"].apply(parse)
unparseable = int(val["pred"].isna().sum())
val["pred"] = val.apply(lambda r: (1 - r["true"]) if pd.isna(r["pred"]) else r["pred"], axis=1)

y, p = val["true"].astype(int), val["pred"].astype(int)
result = {
    "n": len(val),
    "unparseable_outputs": unparseable,
    "recall_ckd": round(recall_score(y, p), 3),
    "precision": round(precision_score(y, p), 3),
    "f1": round(f1_score(y, p), 3),
    "confusion_matrix": confusion_matrix(y, p).tolist(),
    "prompt_format": "chat template from adapter, enable_thinking=False, greedy decoding",
}
print(result)
json.dump(result, open("/content/autoscientist_val.json", "w"), indent=2)
val.to_csv("/content/val_predictions.csv", index=False)

from google.colab import files
files.download("/content/autoscientist_val.json")
files.download("/content/val_predictions.csv")

# ---------------------------------------------------------------- cell 3: margin AUC diagnostic
from sklearn.metrics import roc_auc_score


def margin(prompt):
    text = build(prompt)
    n = tok(text, return_tensors="pt")["input_ids"].shape[1]

    def lp(ans):
        full = tok(text + ans, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**full).logits[0, n - 1:-1].float()
        lps = torch.log_softmax(logits, -1)
        ids = full["input_ids"][0, n:]
        return lps[range(len(ids)), ids].sum().item()

    return lp("CKD") - lp("not CKD")


val["margin"] = [margin(p) for p in val["prompt"]]
print("ROC-AUC from margin:", roc_auc_score(val["true"].astype(int), val["margin"]))

# ---------------------------------------------------------------- cell 4: save the margin results
auc = float(roc_auc_score(val["true"].astype(int), val["margin"]))
json.dump({"n": len(val), "roc_auc_from_margin": round(auc, 4),
           "method": "log P('CKD') - log P('not CKD') for each prompt, same chat template, no threshold chosen"},
          open("/content/autoscientist_margin_auc.json", "w"), indent=2)
val[["margin", "true"]].to_csv("/content/autoscientist_margins.csv", index=False)
files.download("/content/autoscientist_margin_auc.json")
files.download("/content/autoscientist_margins.csv")
