"""Apply the trained real Qwen LoRA adapter to an unseen caption."""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from clipcompass.research_lora import collate, response_logits, tokenized_samples

p = argparse.ArgumentParser()
p.add_argument("--text", required=True)
p.add_argument("--model", default="data/models/qwen2.5-0.5b")
p.add_argument("--experiment", default="artifacts/research/qwen-lora")
p.add_argument("--output")
a = p.parse_args()
torch.set_num_threads(4)
directory = Path(a.experiment)
report = json.loads((directory / "lora_results.json").read_text(encoding="utf-8"))
categories = report["classes"]
tokenizer = AutoTokenizer.from_pretrained(a.model)
tokenizer.pad_token = tokenizer.eos_token
base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, attn_implementation="sdpa", trust_remote_code=False)
model = PeftModel.from_pretrained(base, directory / "adapter").eval()
# The category field below only satisfies dataset packing; its target is unused at inference.
rows, codes, ids = tokenized_samples(tokenizer, {"inference": [{"text": a.text, "category": categories[0], "video_id": -1}]}, categories)
inputs, _ = collate(rows["inference"], tokenizer)
with torch.inference_mode():
    logits = response_logits(model, inputs)[0, ids]
    probabilities = logits.softmax(-1)
result = {"input_text": a.text, "predicted_category": categories[int(probabilities.argmax())], "scope": "8-category constrained classifier; not open-ended generation",
          "scores_are_uncalibrated": True,
          "category_scores": {category: float(score) for category, score in zip(categories, probabilities)}}
serialized = json.dumps(result, ensure_ascii=False, indent=2)
if a.output:
    Path(a.output).write_text(serialized, encoding="utf-8")
print(serialized)
