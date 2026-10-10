"""Diagnostic controls only: cold ID models tie, so order can affect reported metrics."""
import json
from pathlib import Path

import numpy as np

from clipcompass.data import write_json
from clipcompass.research_data import load, training_matrix
from clipcompass.research_eval import candidate_matrices, evaluate
from clipcompass.research_train import observed_negatives

out = Path("artifacts/research/kuairec")
data = load("data/research/kuairec")
x, neg = training_matrix(data), observed_negatives(data)
eligible = (np.diff(x.indptr)>0) & (np.diff(neg.indptr)>0)
candidates, truth = candidate_matrices(data, 2, True)
results = {}
for name, values in [("ascending_id_tie", -np.arange(data["shape"][1])), ("descending_id_tie", np.arange(data["shape"][1]))]:
    results[name] = evaluate(np.broadcast_to(values, data["shape"]), candidates, truth, eligible)
for seed in [42, 2024, 2025]:
    scores = np.random.default_rng(seed).random(data["shape"])
    results[f"random_seed{seed}"] = evaluate(scores, candidates, truth, eligible)
write_json(out / "cold_tie_audit.json", {"purpose": "post-evaluation diagnostic; no model/hyperparameter changes; not a tuned baseline", "results": results})
lines = ["", "## Cold ID模型的平分诊断", "", "无行为内容的ID模型分数相同，主协议按ID稳定打破平分。因此其cold结果不是有效的个性化能力；ID排序仍可能利用数据中的隐含结构。以下补充随机与反向排序，作为诊断，不改动主结果。", "", "| 诊断 | Cold Recall@20 | Cold NDCG@20 |", "|---|---:|---:|"]
for name, r in results.items():
    lines.append(f"| {name} | {r['recall@20']:.5f} | {r['ndcg@20']:.5f} |")
with (out / "REAL_REPORT.md").open("a", encoding="utf-8") as file:
    file.write("\n".join(lines))
print(json.dumps({n: {k:v for k,v in r.items() if k != 'per_user'} for n,r in results.items()}, indent=2))
