import argparse
import json

from clipcompass.data import write_json
from clipcompass.understand import analyze

p = argparse.ArgumentParser()
p.add_argument("--frames", nargs="+", required=True)
p.add_argument("--context", default="")
p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
p.add_argument("--output", default="data/analysis.json")
a = p.parse_args()
write_json(a.output, analyze(a.frames, a.context, a.base_url, a.model))
print(a.output)
