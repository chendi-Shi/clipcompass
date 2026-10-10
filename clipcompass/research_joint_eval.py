"""Joint warm+cold exposure ranking to reveal score-scale tradeoffs between cohorts."""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from .data import write_json
from .research_data import load, training_matrix
from .research_eval import candidate_matrices, evaluate, paired_bootstrap
from .research_train import history_features, observed_negatives


@threadpool_limits.wrap(limits=2)
def run(directory, output):
    out = Path(output)
    main = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    data = load(directory)
    x = training_matrix(data)
    neg = observed_negatives(data)
    eligible = (np.diff(x.indptr)>0) & (np.diff(neg.indptr)>0)
    warm, cold = candidate_matrices(data, 2, False), candidate_matrices(data, 2, True)
    candidates, truth = warm[0]+cold[0], warm[1]+cold[1]
    results = {}
    def record(name, scores):
        results[name] = evaluate(scores, candidates, truth, eligible)
    popularity = np.asarray(x.sum(0)).reshape(-1)
    record("popularity", np.broadcast_to(popularity, data["shape"]))
    mask = (data["split"]==0) & ~data["cold"][data["item"]]
    exposure = np.bincount(data["item"][mask], minlength=data["shape"][1])
    rate = (popularity+20*popularity.sum()/max(1, exposure.sum()))/(exposure+20)
    record("smoothed_longwatch_rate", np.broadcast_to(rate, data["shape"]))
    tfidf = sp.load_npz(Path(directory)/"tfidf.npz")
    record("tfidf_history", (normalize(x@tfidf, axis=1)@tfidf.T).toarray())
    e5 = np.load(Path(directory)/"e5.npy")
    record("e5_history", normalize(history_features(x, e5), axis=1)@e5.T)
    gram = (x.T@x).toarray().astype(np.float64)
    gram.flat[::len(gram)+1] += main["validation"]["ease_selected_lambda"]
    precision = np.linalg.inv(gram)
    b = -precision/np.diag(precision)[None, :]
    np.fill_diagonal(b, 0)
    record("ease", np.asarray(x@b, dtype=np.float32))
    for kind in ["bpr_mf", "lightgcn", "content_gate", "content_ssl"]:
        for seed in main["seeds"]:
            name = f"{kind}_seed{seed}"
            state = torch.load(out / f"{name}.pt", map_location="cpu", weights_only=True)
            record(name, (state["user_vectors"]@state["item_vectors"].T).numpy())
    summary = {}
    for kind in main["summary"]:
        names = [kind] if kind in results else [f"{kind}_seed{s}" for s in main["seeds"]]
        summary[kind] = {metric: {"mean": float(np.mean([results[n][metric] for n in names])),
            "std": float(np.std([results[n][metric] for n in names], ddof=1)) if len(names)>1 else 0.} for metric in ["recall@20", "ndcg@20"]}
    def mean_users(kind):
        indexed = [{r["user"]: r["ndcg"] for r in results[f"{kind}_seed{s}"]["per_user"]} for s in main["seeds"]]
        return [{"user":u, "ndcg":float(np.mean([m[u] for m in indexed]))} for u in indexed[0]]
    write_json(out / "joint_results.json", {"protocol": "all warm+cold observed test exposures ranked jointly; no model selection on these test results",
               "summary": summary, "results": results,
               "paired_content_gate_vs_lightgcn": paired_bootstrap(mean_users("content_gate"), mean_users("lightgcn"))})
    lines = ["", "## Warm/Cold 混合候选的整体排序", "", "用于检验不同候选组的分数是否可比较；未用于选择模型或修改超参。", "",
             "| 方法 | Joint Recall@20 | Joint NDCG@20 |", "|---|---:|---:|"]
    for kind, values in summary.items():
        lines.append(f"| {kind} | {values['recall@20']['mean']:.5f} ± {values['recall@20']['std']:.5f} | {values['ndcg@20']['mean']:.5f} ± {values['ndcg@20']['std']:.5f} |")
    lines += ["", "冷启动分组内改善不自动等于整体排序改善。若混合候选出现退化，需要独立验证分数校准/分支归一化方案，不能只展示冷启动提升。"]
    with (out / "REAL_REPORT.md").open("a", encoding="utf-8") as file:
        file.write("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/research/kuairec")
    p.add_argument("--output", default="artifacts/research/kuairec")
    a = p.parse_args()
    run(a.data, a.output)
