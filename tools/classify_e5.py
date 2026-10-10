"""Inference with the validation-selected E5 linear classifier; trusted local files."""
import argparse
import json
from pathlib import Path

import joblib
import torch

from clipcompass.research_data import embed_texts

p = argparse.ArgumentParser()
p.add_argument("--text", required=True)
p.add_argument("--encoder", default="data/models/multilingual-e5-small")
p.add_argument("--classifier", default="artifacts/research/qwen-lora/e5_logistic.joblib")
p.add_argument("--output")
a = p.parse_args()
torch.set_num_threads(4)
state = joblib.load(a.classifier)
features, _ = embed_texts([a.text], a.encoder)
probabilities = state["classifier"].predict_proba(features)[0]
predicted = int(state["classifier"].predict(features)[0])
result = {"input_text":a.text, "model":"frozen multilingual E5 + validation-selected logistic regression", "predicted_category":state["classes"][predicted],
          "scope":"8-category text classifier; not video understanding", "scores_are_uncalibrated":True,
          "category_scores":{state['classes'][int(c)]:float(score) for c,score in zip(state['classifier'].classes_, probabilities)}}
serialized = json.dumps(result, ensure_ascii=False, indent=2)
if a.output:
    Path(a.output).write_text(serialized, encoding="utf-8")
print(serialized)
