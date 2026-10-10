"""Frozen offline model service with exact retrieval over the real catalog."""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

from .data import write_json
from .research_data import load


class ResearchEngine:
    def __init__(self, directory, checkpoint):
        self.data = load(directory)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.users = state["user_vectors"].numpy()
        self.items = state["item_vectors"].numpy()
        self.kind = state["kind"]
        if self.users.shape[0] != self.data["shape"][0] or self.items.shape[0] != self.data["shape"][1]:
            raise ValueError("checkpoint/data shape mismatch")
        self.seen = [set() for _ in range(len(self.users))]
        prior = self.data["split"] < 2
        for u, i in zip(self.data["user"][prior], self.data["item"][prior]):
            self.seen[int(u)].add(int(i))

    def recommend(self, user=0, k=10, cold_only=False):
        if not 0 <= user < len(self.users) or not 1 <= k <= 100:
            raise ValueError("user or k out of range")
        scores = self.items @ self.users[user]
        scores[list(self.seen[user])] = -np.inf
        if cold_only:
            scores[~self.data["cold"]] = -np.inf
        pool = np.flatnonzero(np.isfinite(scores))
        ids = pool[np.lexsort((pool, -scores[pool]))[:k]]
        return {"internal_user_id": user, "source_user_id": int(self.data["original_users"][user]), "model": self.kind,
                "data_source": "KuaiRec", "retrieval": "exact dot product over frozen full catalog; no online updates",
                "items": [{"video_id": self.data["catalog"][i]["video_id"], "caption": self.data["catalog"][i]["caption"],
                           "category": self.data["catalog"][i]["category"], "cold": bool(self.data["cold"][i]),
                           "ranking_score": float(scores[i]), "score_is_probability": False} for i in ids]}


def benchmark(engine, output):
    from threadpoolctl import threadpool_limits
    rng = np.random.default_rng(42)
    with threadpool_limits(limits=1):
        for u in range(min(10, len(engine.users))):
            engine.recommend(u)
        times = []
        for u in rng.integers(0, len(engine.users), 300):
            start = time.perf_counter()
            engine.recommend(int(u))
            times.append((time.perf_counter()-start)*1000)
    result = {"queries": len(times), "catalog_items": len(engine.items), "dimension": engine.items.shape[1],
              "p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
              "vector_memory_bytes": int(engine.users.nbytes+engine.items.nbytes),
              "measured_scope": "single CPU BLAS thread, exact scoring + seen mask + topk + JSON object construction; excludes HTTP/encoder/model load",
              "scale": "actual KuaiRec catalog only; not a million-item service benchmark"}
    write_json(Path(output) / "serving_benchmark.json", result)
    write_json(Path(output) / "real_recommendations.json", engine.recommend(0))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/research/kuairec")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--output", default="artifacts/research/kuairec")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    engine = ResearchEngine(a.data, a.checkpoint)
    if a.benchmark:
        benchmark(engine, a.output)
    else:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path != "/recommend":
                    self.send_response(404); self.end_headers(); return
                query = parse_qs(parsed.query)
                try:
                    result = engine.recommend(int(query.get("user", [0])[0]), int(query.get("k", [10])[0]), query.get("cold", ["false"])[0] == "true")
                    body = json.dumps(result, ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                except (ValueError, IndexError) as error:
                    body = json.dumps({"error": str(error)}).encode()
                    self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers(); self.wfile.write(body)
        print(f"Serving at http://127.0.0.1:{a.port}/recommend?user=0&k=10", flush=True)
        ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
