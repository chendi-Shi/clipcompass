import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import histories, load_dataset, temporal_split, write_json
from .evaluate import evaluate
from .model import Recommender, alignment_loss, pairwise_loss


def prepare(data):
    train, val, test = temporal_split(data["interactions"])
    combined = np.concatenate([data["text"], data["visual"]], axis=1)
    h = histories(train, combined, int(data["users"]))
    return train, val, test, torch.tensor(h), torch.tensor(data["text"]), torch.tensor(data["visual"])


@torch.no_grad()
def all_scores(model, h, text, visual):
    model.eval()
    users = model.user_vectors(torch.arange(len(h)), h)
    items = model.item_vectors(text, visual)
    retrieval = users @ items.T
    # Shared multitask ranker over the full candidate catalog for reproducible evaluation.
    rank = torch.stack([model.logits(u.expand_as(items), items)[:, 0] for u in users])
    return (retrieval + .2*rank).numpy()


def fit(data, epochs=80, dim=32, visual=True, ssl=.1, seed=42, teacher=None):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    tr, va, te, h, tx, vi = prepare(data)
    model = Recommender(tx.shape[1], vi.shape[1], len(h), dim, visual)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.008, weight_decay=.001)
    train_rows = torch.tensor(tr)
    validation = torch.tensor(va)
    available = torch.tensor(data["visual_available"])
    # Only items first observed in train participate in alignment training.
    observed = torch.zeros(len(tx), dtype=torch.bool)
    observed[train_rows[:, 1]] = True
    available &= observed
    best, best_state, curve = float("inf"), None, []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        item = model.item_vectors(tx, vi)
        user = model.user_vectors(torch.arange(len(h)), h)
        logits = model.logits(user[train_rows[:, 0]], item[train_rows[:, 1]])
        bce = F.binary_cross_entropy_with_logits(logits, train_rows[:, 3:5].float())
        bpr = pairwise_loss(user, item, train_rows, generator)
        align = alignment_loss(model, tx, vi, available)
        kd = torch.tensor(0.)
        if teacher is not None:
            # Teacher uses exactly the same training exposures; no test soft labels.
            kd = F.binary_cross_entropy_with_logits(logits / 2, torch.sigmoid(teacher / 2))*4
        loss = bce + .4*bpr + ssl*align + .3*kd
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            iu = model.user_vectors(validation[:, 0], h[validation[:, 0]])
            iv = model.item_vectors(tx[validation[:, 1]], vi[validation[:, 1]])
            val_loss = F.binary_cross_entropy_with_logits(model.logits(iu, iv), validation[:, 3:5].float()).item()
        curve.append({"epoch": epoch+1, "loss": float(loss.detach()), "validation_bce": val_loss,
                      "bpr": float(bpr.detach()), "alignment": float(align.detach())})
        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, curve


def train_suite(directory, output, epochs=80, seed=42):
    torch.set_num_threads(2)
    data = load_dataset(directory)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    tr, va, te, h, tx, vi = prepare(data)
    start = time.perf_counter()
    popularity = np.bincount(tr[tr[:, 3] == 1, 1], minlength=len(tx))
    experiments = {"popularity": evaluate(np.tile(popularity, (len(h), 1)).astype(float), tr, va, te, data["catalog"])}
    curves = {}
    full = None
    configs = [("text_only", False, 0.), ("multimodal_no_ssl", True, 0.), ("multimodal_ssl", True, .1)]
    for name, use_visual, ssl in configs:
        model, curve = fit(data, epochs=epochs, visual=use_visual, ssl=ssl, seed=seed)
        scores = all_scores(model, h, tx, vi)
        experiments[name] = evaluate(scores, tr, va, te, data["catalog"])
        experiments[name]["parameters"] = sum(p.numel() for p in model.parameters())
        experiments[name]["best_epoch"] = min(curve, key=lambda x: x["validation_bce"])["epoch"]
        curves[name] = curve
        if name == "multimodal_ssl":
            full = model
    full.eval()
    with torch.no_grad():
        iu = full.user_vectors(torch.tensor(tr[:, 0]), h[tr[:, 0]])
        iv = full.item_vectors(tx[tr[:, 1]], vi[tr[:, 1]])
        teacher = full.logits(iu, iv).detach()
    student, curve = fit(data, epochs=epochs, dim=12, ssl=.1, seed=seed, teacher=teacher)
    experiments["distilled_student"] = evaluate(all_scores(student, h, tx, vi), tr, va, te, data["catalog"])
    experiments["distilled_student"]["parameters"] = sum(p.numel() for p in student.parameters())
    curves["distilled_student"] = curve
    seen = [sorted(set(int(i) for uid, i, *_ in tr if uid == u)) for u in range(len(h))]
    torch.save({"state": full.state_dict(), "config": {"text_dim": tx.shape[1], "visual_dim": vi.shape[1],
               "users": len(h), "dim": 32, "use_visual": True}, "history": h,
               "text": tx, "visual": vi, "seen": seen, "catalog": data["catalog"],
               "provenance": data["provenance"]}, out / "model.pt")
    torch.save({"state": student.state_dict(), "config": {"text_dim": tx.shape[1], "visual_dim": vi.shape[1],
               "users": len(h), "dim": 12, "use_visual": True}}, out / "student.pt")
    report = {"source": data["provenance"], "seed": seed, "epochs": epochs,
              "torch": torch.__version__, "numpy": np.__version__, "seconds": time.perf_counter()-start,
              "protocol": "global temporal 70/15/15; train-only history; validation-BCE checkpoint; full catalog test ranking; earlier exposures masked; unknown publication times treated as closed catalog",
              "rows": {"train": len(tr), "validation": len(va), "test": len(te)}, "experiments": experiments}
    write_json(out / "metrics.json", report)
    write_json(out / "learning_curves.json", curves)
    lines = ["# 实测实验报告", "", f"数据来源：{data['provenance']['source']}。seed={seed}，epochs={epochs}。", "",
             "模拟数据仅验证方法和工程闭环，不代表 TikTok/KuaiRec 真实业务效果。", "",
             "| Model | Recall@10 | NDCG@10 | Coverage@10 |", "|---|---:|---:|---:|"]
    for name, m in experiments.items():
        lines.append(f"| {name} | {m.get('recall@10', 0):.4f} | {m.get('ndcg@10', 0):.4f} | {m['catalog_coverage@10']:.4f} |")
    lines += ["", "各语种分组、覆盖用户数、参数量见 metrics.json；所有变体均报告，不能假设增加模态或蒸馏一定提升。",
              "", "ranker 在评测时打分全目录；演示服务先召回再排序并做 MMR，二者效果需要单独评测。"]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report
