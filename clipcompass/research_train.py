import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from threadpoolctl import threadpool_limits
from torch.nn import functional as F

from .data import write_json
from .research_data import load, training_matrix
from .research_eval import candidate_matrices, evaluate, paired_bootstrap
from .research_model import GraphContent, normalized_graph, semantic_alignment


def history_features(x, features):
    h = np.asarray(x @ features)
    h /= np.maximum(np.asarray(x.sum(1)), 1)
    if sp.issparse(h):
        h = h.toarray()
    return h.astype(np.float32)


def observed_negatives(data):
    mask = (data["split"] == 0) & ~data["cold"][data["item"]] & (data["watch_ratio"] < 1)
    return sp.csr_matrix((np.ones(int(mask.sum()), dtype=np.float32), (data["user"][mask], data["item"][mask])), shape=data["shape"])


def draw_pairs(positive, negative, eligible, rng, per_user=128):
    users = np.repeat(np.flatnonzero(eligible), per_user)
    def draw(matrix):
        length = np.diff(matrix.indptr)[users]
        offsets = (rng.random(len(users))*length).astype(np.int64)
        return matrix.indices[matrix.indptr[users]+offsets]
    return torch.tensor(users), torch.tensor(draw(positive)), torch.tensor(draw(negative))


def fit(data, x, features, graph, candidates, truths, kind, seed, epochs=60):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    negatives = observed_negatives(data)
    eligible = (np.diff(x.indptr) > 0) & (np.diff(negatives.indptr) > 0)
    active = torch.tensor(np.asarray(x.sum(0)).reshape(-1) > 0)
    text = torch.tensor(features)
    history = torch.tensor(history_features(x, features))
    model = GraphContent(*data["shape"], features.shape[1], dim=32, layers=0 if kind == "bpr_mf" else 2,
                         content=kind.startswith("content"))
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    best_metric, best_state, curve = -1, None, []
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        users, positive, negative = draw_pairs(x, negatives, eligible, rng)
        uv, iv, alignment = model.vectors(graph, text, history, active)
        margin = (uv[users] * (iv[positive] - iv[negative])).sum(-1)
        bpr = F.softplus(-margin).mean()
        regularization = model.nodes.weight.square().mean()
        loss = bpr + .001*regularization
        ssl = torch.tensor(0.)
        if kind == "content_ssl":
            ssl = semantic_alignment(*alignment, active, generator)
            loss += .02*ssl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                uv, iv, _ = model.vectors(graph, text, history, active)
                scores = (uv @ iv.T).numpy()
            metric = evaluate(scores, candidates, truths, eligible)["ndcg@20"]
            curve.append({"epoch": epoch, "loss": float(loss.detach()), "bpr": float(bpr.detach()),
                          "ssl": float(ssl.detach()), "validation_warm_ndcg@20": metric})
            print(f"{kind} seed={seed} epoch={epoch} val_ndcg={metric:.5f}", flush=True)
            if metric > best_metric:
                best_metric = metric
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        uv, iv, _ = model.vectors(graph, text, history, active)
    return model, (uv @ iv.T).detach().numpy(), curve, uv.detach(), iv.detach(), eligible


@threadpool_limits.wrap(limits=4)
def run(directory, output, epochs=60, seeds=(42, 2024, 2025)):
    torch.set_num_threads(4)
    data = load(directory)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    x = training_matrix(data)
    negatives = observed_negatives(data)
    eligible = (np.diff(x.indptr)>0) & (np.diff(negatives.indptr)>0)
    e5 = np.load(Path(directory) / "e5.npy")
    tfidf = sp.load_npz(Path(directory) / "tfidf.npz")
    graph = normalized_graph(x)
    val = candidate_matrices(data, 1, False)
    warm, cold = candidate_matrices(data, 2, False), candidate_matrices(data, 2, True)
    results, curves, validations = {}, {}, {}
    def record(name, scores):
        results[name] = {"warm": evaluate(scores, *warm, eligible), "cold": evaluate(scores, *cold, eligible)}
    popularity = np.asarray(x.sum(0)).reshape(-1)
    record("popularity", np.broadcast_to(popularity, data["shape"]))
    # Strong item performance prior; denominators use train exposures only.
    m = (data["split"] == 0) & ~data["cold"][data["item"]]
    exposure = np.bincount(data["item"][m], minlength=data["shape"][1])
    rate = (popularity + 20*popularity.sum()/max(1, exposure.sum())) / (exposure+20)
    record("smoothed_longwatch_rate", np.broadcast_to(rate, data["shape"]))
    # Content cosine profiles use positive histories only.
    counts = np.maximum(np.asarray(x.sum(1)), 1)
    htf = x @ tfidf
    from sklearn.preprocessing import normalize
    htf = normalize(htf, axis=1)
    record("tfidf_history", (htf @ tfidf.T).toarray())
    he5 = normalize(history_features(x, e5), axis=1)
    record("e5_history", he5 @ e5.T)
    # EASE is a competitive closed-form collaborative baseline; select lambda on validation.
    gram = (x.T @ x).toarray().astype(np.float64)
    best, ease_scores, ease_lambda = -1, None, None
    for ridge in [100., 500., 1000.]:
        g = gram.copy()
        g.flat[::len(g)+1] += ridge
        precision = np.linalg.inv(g)
        b = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(b, 0)
        scores = np.asarray(x @ b, dtype=np.float32)
        value = evaluate(scores, *val, eligible)["ndcg@20"]
        validations[f"ease_lambda_{ridge}"] = value
        print(f"EASE lambda={ridge} validation={value:.5f}", flush=True)
        if value > best:
            best, ease_scores, ease_lambda = value, scores, ridge
    record("ease", ease_scores)
    validations["ease_selected_lambda"] = ease_lambda
    for kind in ["bpr_mf", "lightgcn", "content_gate", "content_ssl"]:
        for seed in seeds:
            name = f"{kind}_seed{seed}"
            start = time.perf_counter()
            model, scores, curve, uv, iv, elig = fit(data, x, e5, graph, *val, kind, seed, epochs)
            record(name, scores)
            results[name]["training_seconds"] = time.perf_counter()-start
            results[name]["parameters"] = sum(p.numel() for p in model.parameters())
            curves[name] = curve
            torch.save({"state": model.state_dict(), "user_vectors": uv, "item_vectors": iv,
                        "kind": kind, "seed": seed, "source": "KuaiRec", "catalog": data["catalog"]}, out / f"{name}.pt")
            # Save intermediate progress so a completed run remains recoverable.
            write_json(out / "results.json", results)
            write_json(out / "curves.json", curves)
    summary = {}
    for kind in ["popularity", "smoothed_longwatch_rate", "tfidf_history", "e5_history", "ease", "bpr_mf", "lightgcn", "content_gate", "content_ssl"]:
        names = [kind] if kind in results else [f"{kind}_seed{s}" for s in seeds]
        summary[kind] = {}
        for group in ["warm", "cold"]:
            summary[kind][group] = {}
            for metric in ["recall@20", "ndcg@20"]:
                values = [results[n][group][metric] for n in names]
                summary[kind][group][metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values)>1 else 0.}
    # Choose based ONLY on mean warm validation score, never on test/cold metrics.
    winner = max(["bpr_mf", "lightgcn", "content_gate", "content_ssl"], key=lambda kind: np.mean([
        max(c["validation_warm_ndcg@20"] for c in curves[f"{kind}_seed{s}"]) for s in seeds]))
    comparisons = {}
    for group in ["warm", "cold"]:
        # Collapse model seeds per user before bootstrap; seed observations are not independent users.
        def averaged_rows(kind):
            indexed = [{r["user"]: r["ndcg"] for r in results[f"{kind}_seed{s}"][group]["per_user"]} for s in seeds]
            return [{"user": u, "ndcg": float(np.mean([m[u] for m in indexed]))} for u in indexed[0]]
        comparisons[group] = paired_bootstrap(averaged_rows("content_gate"), averaged_rows("lightgcn"))
    write_json(out / "summary.json", {"protocol": data["protocol"], "features": json.loads((Path(directory)/"feature_manifest.json").read_text()),
               "epochs": epochs, "seeds": list(seeds), "summary": summary, "validation": validations,
               "selected_neural_kind_by_validation": winner, "paired_content_gate_vs_lightgcn": comparisons,
               "versions": {"torch": str(torch.__version__), "numpy": np.__version__}})
    lines = ["# KuaiRec 真实数据实验", "", f"用户 {data['shape'][0]}，内容 {data['shape'][1]}，原始交互 {data['protocol']['raw_rows']:,}。",
             f"神经模型 seeds={list(seeds)}，每次 {epochs} epochs。验证集选择的神经模型：{winner}。", "",
             "候选：用户在各测试分组中真实曝光的全部内容；非采样负例。以下为均值±跨seed标准差。", "",
             "| 方法 | Warm Recall@20 | Warm NDCG@20 | Cold Recall@20 | Cold NDCG@20 |", "|---|---:|---:|---:|---:|"]
    for kind, groups in summary.items():
        values = [groups[g][m] for g in ["warm", "cold"] for m in ["recall@20", "ndcg@20"]]
        lines.append("| " + kind + " | " + " | ".join(f"{v['mean']:.5f} ± {v['std']:.5f}" for v in values) + " |")
    lines += ["", "这是观看行为上的真实离线推荐实验，不是线上 A/B，不是创作转化实验。此推荐实验没有视频视觉特征，也未使用LLM微调输出；独立Qwen LoRA实验见另一报告。",
              "冷启动内容的全部训练标签被移除，其验证标签不参与模型选择；冷启动不是随机用户拆分。",
              "协议/来源校验/排除用户/候选规模/每用户指标/置信区间详见 summary.json、results.json。"]
    (out / "REAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/research/kuairec")
    p.add_argument("--output", default="artifacts/research/kuairec")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 2025])
    a = p.parse_args()
    run(a.data, a.output, a.epochs, tuple(a.seeds))
