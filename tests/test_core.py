import csv
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch

from clipcompass.data import hash_text, histories, import_kuairec, load_dataset, make_demo, temporal_split
from clipcompass.evaluate import evaluate, ranking_metrics
from clipcompass.model import Recommender, alignment_loss, pairwise_loss
from clipcompass.recommend import recommend
from clipcompass.train import train_suite
from clipcompass.understand import analyze, validate_analysis
from tools.export_sft import export


class CoreTests(unittest.TestCase):
    def test_temporal_boundaries_and_history(self):
        events = np.array([[0, t % 3, t, t % 2, 1] for t in range(20)])
        train, val, test = temporal_split(events)
        self.assertLess(train[:, 2].max(), val[:, 2].min())
        self.assertLess(val[:, 2].max(), test[:, 2].min())
        h = histories(train, np.eye(3, dtype=np.float32), 2)
        self.assertTrue(np.allclose(h[1], 0))
        edited = test.copy()
        edited[:, 3] = 1-edited[:, 3]
        changed_train, _, _ = temporal_split(np.concatenate([train, val, edited]))
        self.assertTrue(np.array_equal(h, histories(changed_train, np.eye(3, dtype=np.float32), 2)))

    def test_ranking_known_answer(self):
        result = ranking_metrics(np.array([.8, .9, .1]), {0, 1}, k=2)
        self.assertEqual(result["recall@10"], 1)
        self.assertEqual(result["ndcg@10"], 1)
        self.assertIsNone(ranking_metrics([1, 2], set()))
        self.assertEqual(ranking_metrics(np.array([1., 2., 3.]), {0}, k=1)["ndcg@10"], 0)

    def test_mask_seen_and_future(self):
        catalog = [{"language": "en", "published_at": 0}, {"language": "en", "published_at": 0},
                   {"language": "es", "published_at": 100}]
        result = evaluate(np.array([[10., 2., 20.]]), np.array([[0, 0, 0, 1, 1]]),
                          np.empty((0, 5), dtype=int), np.array([[0, 1, 10, 1, 1], [0, 2, 11, 1, 1]]), catalog)
        self.assertEqual(result["recall@10"], 1)
        self.assertEqual(result["catalog_coverage@10"], 1/3)

    def test_hash_is_stable_and_nonempty(self):
        self.assertTrue(np.array_equal(hash_text("猫咪 cat"), hash_text("猫咪 cat")))
        self.assertAlmostEqual(float(np.linalg.norm(hash_text("猫咪 cat"))), 1, places=5)

    def test_alignment_gradients_and_missing_visuals(self):
        torch.manual_seed(1)
        model = Recommender(8, 4, 2, dim=4)
        tx, vi = torch.randn(3, 8), torch.randn(3, 4)
        loss = alignment_loss(model, tx, vi, torch.tensor([True, True, False]))
        loss.backward()
        self.assertTrue(torch.isfinite(model.text[0].weight.grad).all())
        self.assertEqual(float(alignment_loss(model, tx, vi, torch.zeros(3, dtype=torch.bool))), 0)

    def test_pairwise_uses_same_user_and_handles_no_pairs(self):
        u = torch.tensor([[1., 0.]])
        items = torch.tensor([[1., 0.], [-1., 0.]], requires_grad=True)
        events = torch.tensor([[0, 0, 0, 1, 1], [0, 1, 1, 0, 0]])
        loss = pairwise_loss(u, items, events, torch.Generator().manual_seed(1))
        self.assertLess(float(loss.detach()), .01)
        loss.backward()
        self.assertTrue(torch.isfinite(items.grad).all())
        self.assertEqual(float(pairwise_loss(u, items, events[:1], torch.Generator()).detach()), 0)

    def test_attach_reviewed_real_frames(self):
        from PIL import Image
        from tools.attach_content import attach
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            make_demo(p / "data", users=4, items=30)
            frame = p / "frame.png"
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(frame)
            rows = [{"item_id": 0, "reviewed": True, "frames": [str(frame)], "analysis": {
                "language": "en", "summary": "Red image", "topics": ["color"],
                "evidence": [{"frame_index": 0, "observation": "red background"}]}}]
            manifest = p / "manifest.json"
            manifest.write_text(json.dumps(rows))
            attach(p / "data", manifest, p / "attached")
            result = load_dataset(p / "attached")
            self.assertEqual(result["catalog"][0]["caption"], "Red image")
            self.assertTrue(np.array_equal(result["visual"][0].reshape(16, 3), np.tile([1, 0, 0], (16, 1))))

    def test_kuairec_csv_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            with (p / "captions.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["video_id", "caption"])
                w.writerows([[i, f"视频 {i}"] for i in range(12)])
            with (p / "matrix.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["user_id", "video_id", "timestamp", "watch_ratio"])
                w.writerows([[0, i, i, 2.5 if i % 2 else .2] for i in range(12)])
            import_kuairec(p / "matrix.csv", p / "captions.csv", p / "out")
            data = load_dataset(p / "out")
            self.assertEqual(len(data["catalog"]), 12)
            self.assertFalse(data["visual_available"].any())
            self.assertEqual(int(data["interactions"][:, 3].sum()), 6)

    def test_analysis_schema_rejects_hallucinated_frame(self):
        a = {"language": "en", "summary": "A cat.", "topics": ["pets"],
             "evidence": [{"frame_index": 2, "observation": "cat"}]}
        with self.assertRaises(ValueError):
            validate_analysis(a, 1)

    def test_local_vlm_transport(self):
        answer = {"language": "en", "summary": "A cat.", "topics": ["pets"],
                  "evidence": [{"frame_index": 0, "observation": "cat"}]}
        observed = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                observed.append(payload)
                body = json.dumps({"choices": [{"message": {"content": json.dumps(answer)}}]}).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(body)
            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                frame = Path(d) / "f.png"; frame.write_bytes(b"mock-png-bytes")
                result = analyze([frame], base_url=f"http://127.0.0.1:{server.server_port}/v1")
                self.assertEqual(result["summary"], "A cat.")
                self.assertEqual(observed[0]["messages"][0]["content"][1]["type"], "image_url")
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_sft_review_and_video_split(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); frame = p / "f.png"; frame.write_bytes(b"mock")
            answer = {"language": "en", "summary": "cat", "topics": ["pets"],
                      "evidence": [{"frame_index": 0, "observation": "cat"}]}
            rows = [{"video_id": i, "reviewed": True, "split": split, "frames": [str(frame)], "answer": answer}
                    for i, split in enumerate(["train", "eval"])]
            file = p / "labels.json"; file.write_text(json.dumps(rows))
            export(file, p / "sft")
            self.assertEqual(len(json.loads((p / "sft/clip_train.json").read_text())), 1)
            rows[1]["video_id"] = 0; file.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):
                export(file, p / "sft")

    def test_pipeline_checkpoint_and_unseen_recommendations(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            make_demo(p / "data", users=8, items=120)
            result = train_suite(p / "data", p / "out", epochs=2)
            self.assertEqual(len(result["experiments"]), 5)
            self.assertLess(result["experiments"]["distilled_student"]["parameters"], result["experiments"]["multimodal_ssl"]["parameters"])
            cards = recommend(p / "out/model.pt", 0, 5)["cards"]
            train, _, _ = temporal_split(load_dataset(p / "data")["interactions"])
            seen = set(train[train[:, 0] == 0, 1])
            self.assertEqual(len(cards), 5)
            self.assertTrue(all(c["item_id"] not in seen for c in cards))
            self.assertEqual(len({c["item_id"] for c in cards}), 5)
            with self.assertRaises(ValueError):
                recommend(p / "out/model.pt", 99)


if __name__ == "__main__":
    unittest.main()
