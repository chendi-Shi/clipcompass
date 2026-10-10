import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from clipcompass.research_data import load, load_captions, prepare, training_matrix
from clipcompass.research_eval import candidate_matrices, evaluate, paired_bootstrap
from clipcompass.research_model import GraphContent, normalized_graph
from clipcompass.research_train import draw_pairs, fit, observed_negatives


class ResearchTests(unittest.TestCase):
    def test_caption_unquoted_newline_repair_is_audited(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "captions.csv"
            path.write_text("video_id,caption,manual_cover_text\n0,hello\ncontinued,UNKNOWN\n1,normal,title\n", encoding="utf-8")
            frame, audit = load_captions(path)
            self.assertEqual(audit["repaired_physical_line_breaks"], 1)
            self.assertEqual(frame.iloc[0].caption, "hello continued")
            self.assertEqual(len(frame), 2)

    def test_real_protocol_removes_all_cold_edges_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            rows = [[u, i, i*3+u, 2.5 if i%2 else .2] for u in range(3) for i in range(40)]
            rows.append([0, 0, 1000, 20])  # future label must not overwrite first exposure
            pd.DataFrame(rows, columns=["user_id", "video_id", "timestamp", "watch_ratio"]).to_csv(p / "small_matrix.csv", index=False)
            pd.DataFrame({"video_id": range(40), "caption": [f"视频{i}" for i in range(40)]}).to_csv(p / "kuairec_caption_category.csv", index=False)
            (p / "download_manifest.json").write_text("{}")
            prepare(p, p / "out")
            data = load(p / "out")
            self.assertEqual(len(data["user"]), 120)
            x = training_matrix(data)
            self.assertEqual(x[:, data["cold"]].nnz, 0)
            original0 = (data["user"]==0) & (data["item"]==0)
            self.assertAlmostEqual(float(data["watch_ratio"][original0][0]), .2, places=5)
            neg = observed_negatives(data)
            self.assertEqual(neg[:, data["cold"]].nnz, 0)
            self.assertEqual(x.multiply(neg).nnz, 0)

    def test_graph_normalization_and_cold_vectors(self):
        matrix = sp.csr_matrix([[1., 0.], [1., 0.]], dtype=np.float32)
        graph = normalized_graph(matrix)
        dense = graph.to_dense().numpy()
        self.assertTrue(np.allclose(dense, dense.T))
        self.assertTrue(np.allclose(dense[3], 0))
        model = GraphContent(2, 2, 3, layers=2, content=False)
        _, items, _ = model.vectors(graph, torch.randn(2, 3), torch.randn(2, 3), torch.tensor([True, False]))
        self.assertTrue(torch.all(items[1]==0))

    def test_pair_sampler_never_uses_unknown_or_positive_negative(self):
        pos = sp.csr_matrix([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
        neg = sp.csr_matrix([[0., 1., 0.], [1., 0., 0.]], dtype=np.float32)
        u, p, n = draw_pairs(pos, neg, np.ones(2, dtype=bool), np.random.default_rng(42), 20)
        for uid, positive, negative in zip(u.tolist(), p.tolist(), n.tolist()):
            self.assertEqual(pos[uid, positive], 1)
            self.assertEqual(neg[uid, negative], 1)
            self.assertNotEqual(positive, negative)

    def test_full_exposure_candidate_metric(self):
        scores = np.array([[100., 2., 1., 3.]])
        candidates = sp.csr_matrix([[0, 1, 1, 1]], dtype=bool)
        truth = sp.csr_matrix([[0, 0, 0, 1]], dtype=bool)
        result = evaluate(scores, candidates, truth, np.array([True]), k=1)
        self.assertEqual(result["ndcg@1"], 1)
        self.assertEqual(result["mean_candidates"], 3)

    def test_paired_bootstrap_exact_constant_difference(self):
        a = [{"user": i, "ndcg": .3} for i in range(5)]
        b = [{"user": i, "ndcg": .1} for i in range(5)]
        result = paired_bootstrap(a, b, samples=20)
        self.assertTrue(np.allclose(result["95_percent_ci"], [.2, .2]))

    def test_training_is_invariant_to_future_test_labels(self):
        torch.set_num_threads(2)
        item = np.repeat(np.arange(12), 4)
        data = {"user": np.tile(np.arange(4), 12), "item": item,
                "watch_ratio": np.where(item%2==0, 3., .2), "split": np.where(item<8, 0, np.where(item<10, 1, 2)),
                "cold": np.array([False]*11+[True]), "shape": (4, 12)}
        x = training_matrix(data)
        graph = normalized_graph(x)
        features = np.random.default_rng(1).normal(size=(12, 6)).astype(np.float32)
        val = candidate_matrices(data, 1, False)
        a, *_ = fit(data, x, features, graph, *val, "content_gate", 42, epochs=2)
        changed = {**data, "watch_ratio": data["watch_ratio"].copy()}
        changed["watch_ratio"][changed["split"]==2] = 20.
        b, *_ = fit(changed, x, features, graph, *val, "content_gate", 42, epochs=2)
        self.assertTrue(all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()))


if __name__ == "__main__":
    unittest.main()
