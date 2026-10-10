import numpy as np
import scipy.sparse as sp


def candidate_matrices(data, split, cold):
    mask = (data["split"] == split) & (data["cold"][data["item"]] == cold)
    shape = data["shape"]
    candidate = sp.csr_matrix((np.ones(int(mask.sum()), dtype=bool), (data["user"][mask], data["item"][mask])), shape=shape)
    positive_mask = mask & (data["watch_ratio"] >= 2)
    truth = sp.csr_matrix((np.ones(int(positive_mask.sum()), dtype=bool), (data["user"][positive_mask], data["item"][positive_mask])), shape=shape)
    return candidate, truth


def evaluate(scores, candidates, truth, eligible, k=20):
    scores = np.asarray(scores)
    per_user, recommended = [], set()
    for u in np.flatnonzero(eligible):
        pool = candidates.indices[candidates.indptr[u]:candidates.indptr[u+1]]
        pos = truth.indices[truth.indptr[u]:truth.indptr[u+1]]
        if len(pos) == 0 or len(pool) == 0:
            continue
        # Stable tie-break on internal item ID; identical across all methods.
        top = pool[np.lexsort((pool, -scores[u, pool]))[:k]]
        hits = np.isin(top, pos)
        dcg = float(np.sum(hits / np.log2(np.arange(len(top))+2)))
        idcg = float(np.sum(1 / np.log2(np.arange(min(k, len(pos)))+2)))
        per_user.append({"user": int(u), "recall": float(hits.sum()/len(pos)), "ndcg": dcg/idcg,
                         "candidates": len(pool), "positives": len(pos)})
        recommended.update(top.tolist())
    return {f"recall@{k}": float(np.mean([r["recall"] for r in per_user])) if per_user else 0,
            f"ndcg@{k}": float(np.mean([r["ndcg"] for r in per_user])) if per_user else 0,
            "evaluated_users": len(per_user), "eligible_users": int(eligible.sum()),
            "mean_candidates": float(np.mean([r["candidates"] for r in per_user])) if per_user else 0,
            "mean_positive_items": float(np.mean([r["positives"] for r in per_user])) if per_user else 0,
            "recommended_unique_items": len(recommended), "per_user": per_user}


def paired_bootstrap(a, b, samples=2000):
    amap, bmap = {r["user"]: r["ndcg"] for r in a}, {r["user"]: r["ndcg"] for r in b}
    ids = sorted(amap.keys() & bmap.keys())
    if not ids:
        return {"users": 0}
    differences = np.array([amap[u]-bmap[u] for u in ids])
    rng = np.random.default_rng(20261010)
    means = np.array([rng.choice(differences, len(ids), replace=True).mean() for _ in range(samples)])
    return {"users": len(ids), "mean_ndcg_difference": float(differences.mean()),
            "95_percent_ci": [float(x) for x in np.quantile(means, [.025, .975])],
            "resamples": samples, "unit": "user; not causal/online significance"}
