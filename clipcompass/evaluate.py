import numpy as np


def ranking_metrics(scores, truth, k=10):
    if not truth:
        return None
    order = np.argsort(-np.asarray(scores), kind="stable")
    top = [int(i) for i in order if np.isfinite(scores[i])][:k]
    hits = np.asarray([int(i in truth) for i in top])
    dcg = float((hits / np.log2(np.arange(len(hits)) + 2)).sum())
    ideal = float((1 / np.log2(np.arange(min(len(truth), k)) + 2)).sum())
    return {"recall@10": float(hits.sum()/len(truth)), "ndcg@10": dcg/ideal}


def evaluate(scores, train, validation, test, catalog):
    """Full-catalog ranking at test-start; mask earlier exposures and future items."""
    cutoff = int(test[:, 2].min())
    published = np.asarray([c.get("published_at") if c.get("published_at") is not None else -1 for c in catalog])
    seen = {}
    for u, i, *_ in np.concatenate([train, validation]):
        seen.setdefault(int(u), set()).add(int(i))
    results, groups, recommendations = [], {}, set()
    positive = test[test[:, 3] == 1]
    for u in np.unique(test[:, 0]):
        s = np.array(scores[u], copy=True)
        s[list(seen.get(int(u), set()))] = -np.inf
        s[published > cutoff] = -np.inf
        truth = set(int(i) for i in positive[positive[:, 0] == u, 1] if np.isfinite(s[i]))
        value = ranking_metrics(s, truth)
        if value is None:
            continue
        results.append(value)
        recommendations.update(int(i) for i in np.argsort(-s)[:10] if np.isfinite(s[i]))
        # Language-conditioned positives, same global ranking (not language-only candidate pools).
        for language in sorted({catalog[i]["language"] for i in truth}):
            gt = {i for i in truth if catalog[i]["language"] == language}
            groups.setdefault(language, []).append(ranking_metrics(s, gt))
    def average(rows):
        return {key: float(np.mean([r[key] for r in rows])) for key in ("recall@10", "ndcg@10")} if rows else {}
    return {**average(results), "evaluated_users": len(results),
            "excluded_users_without_eligible_positives": len(np.unique(test[:, 0])) - len(results),
            "catalog_coverage@10": len(recommendations)/len(catalog),
            "by_positive_language": {lang: {**average(rows), "users": len(rows)} for lang, rows in groups.items()}}
