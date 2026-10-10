import csv
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np

TOPICS = {
    "food": ["十分钟做番茄意面", "Ten minute tomato pasta", "Pasta de tomate en diez minutos"],
    "fitness": ["居家深蹲训练技巧", "Squat workout at home", "Entrenamiento de sentadillas en casa"],
    "travel": ["城市夜景步行路线", "City night walking route", "Ruta nocturna por la ciudad"],
    "pets": ["给猫咪做益智玩具", "Make a puzzle toy for cats", "Un juguete de ingenio para gatos"],
    "tech": ["手机拍摄构图教程", "Phone photography composition tutorial", "Composicion fotografica con el movil"],
    "music": ["吉他和弦入门练习", "Beginner guitar chord practice", "Practica de acordes de guitarra"],
}
LANGUAGES = ["zh", "en", "es"]


def caption_records(path):
    raw = Path(path).read_text(encoding="utf-8-sig")
    repaired, count = re.subn(r"\n(?!\d+,)(?=.)", " ", raw)
    rows = list(csv.DictReader(io.StringIO(repaired, newline="")))
    if any(None in r or any(v is None for v in r.values()) or not r["video_id"].isdigit() for r in rows):
        raise ValueError("Caption CSV still malformed after line-break repair")
    if len({r["video_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate caption IDs")
    return rows, {"repaired_physical_line_breaks": count, "rows": len(rows), "rule": "non-ID-starting continuation line joins previous row with space"}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def hash_text(text, dim=128):
    """Stable signed character n-gram features; NOT a multilingual LM."""
    x = np.zeros(dim, dtype=np.float32)
    text = text.lower()
    for n in (1, 2, 3):
        for i in range(max(0, len(text) - n + 1)):
            h = hashlib.blake2b(text[i:i+n].encode(), digest_size=8).digest()
            x[int.from_bytes(h[:4], "little") % dim] += 1 if h[4] % 2 else -1
    return x / max(float(np.linalg.norm(x)), 1e-8)


def make_demo(output, seed=42, users=96, items=240):
    """Generate synthetic multilingual captions, RGB frames and exposure logs."""
    if users < 4 or items < 30:
        raise ValueError("demo requires >=4 users and >=30 items")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    keys = list(TOPICS)
    categories = np.arange(items) % len(keys)
    langs = rng.integers(0, 3, items)
    palette = rng.uniform(0.1, 0.9, (6, 4, 4, 3))
    frames = np.clip(palette[categories] + rng.normal(0, 0.13, (items, 4, 4, 3)), 0, 1)
    catalog = [{"item_id": i, "language": LANGUAGES[langs[i]], "topic": keys[categories[i]],
                "caption": TOPICS[keys[categories[i]]][langs[i]] + f" #{i}",
                "published_at": 0, "video_path": None, "synthetic": True} for i in range(items)]
    # Topic is supervision/metadata only: features hash caption, not topic or labels.
    text = np.stack([hash_text(c["caption"]) for c in catalog])
    visual = frames.reshape(items, -1).astype(np.float32)
    preferences = rng.dirichlet(np.ones(6) * 0.4, users)
    user_langs = rng.integers(0, 3, users)
    rows = []
    for u in range(users):
        sampled = rng.choice(items, min(items, 105), replace=False)
        for t, i in enumerate(sampled):
            p = float(np.clip(0.05 + 1.5 * preferences[u, categories[i]] + 0.12 * (langs[i] == user_langs[u]), .02, .95))
            engaged = int(rng.random() < p)
            complete = int(rng.random() < .15 + .68 * engaged)
            rows.append([u, int(i), t * users + u, engaged, complete])
    interactions = np.asarray(rows, dtype=np.int64)
    np.savez_compressed(out / "dataset.npz", text=text, visual=visual, interactions=interactions,
                        users=np.asarray(users), visual_available=np.ones(items, dtype=bool))
    write_json(out / "catalog.json", catalog)
    write_json(out / "provenance.json", {"source": "synthetic", "seed": seed,
        "labels": {"engaged": "synthetic Bernoulli preference", "complete": "synthetic completion"},
        "visual": "generated 4x4 RGB frames, NOT real videos", "languages": LANGUAGES})


def load_dataset(directory):
    p = Path(directory)
    with np.load(p / "dataset.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["catalog"] = json.loads((p / "catalog.json").read_text(encoding="utf-8"))
    data["provenance"] = json.loads((p / "provenance.json").read_text(encoding="utf-8"))
    events = data["interactions"]
    if len(events) < 10 or not np.isfinite(events).all():
        raise ValueError("insufficient or non-finite interactions")
    if not np.isfinite(data["text"]).all() or not np.isfinite(data["visual"]).all():
        raise ValueError("non-finite content features")
    if len(data["catalog"]) != len(data["text"]) or len(data["visual"]) != len(data["text"]):
        raise ValueError("catalog/feature row mismatch")
    if events[:, 0].min() < 0 or events[:, 0].max() >= int(data["users"]):
        raise ValueError("invalid user IDs")
    if events[:, 1].min() < 0 or events[:, 1].max() >= len(data["text"]):
        raise ValueError("invalid item IDs")
    if not np.isin(events[:, 3:5], [0, 1]).all():
        raise ValueError("labels must be binary")
    return data


def temporal_split(events):
    # Global timestamp boundaries keep simultaneous events together.
    times = np.unique(events[:, 2])
    if len(times) < 5:
        raise ValueError("need at least 5 distinct timestamps")
    a, b = times[int(len(times)*.7)], times[int(len(times)*.85)]
    return events[events[:, 2] < a], events[(events[:, 2] >= a) & (events[:, 2] < b)], events[events[:, 2] >= b]


def histories(events, features, users):
    h = np.zeros((users, features.shape[1]), dtype=np.float32)
    counts = np.zeros(users, dtype=np.float32)
    for u, i, _, engaged, _ in events:
        if engaged:
            h[u] += features[i]
            counts[u] += 1
    return h / np.maximum(counts[:, None], 1)


def import_kuairec(matrix, captions, output, max_users=96, max_items=600, max_rows=300000):
    """Bounded real CSV adapter. No fabricated vision or native like labels."""
    caption_map = {}
    caption_rows, caption_audit = caption_records(captions)
    for r in caption_rows:
        caption_map[int(r["video_id"])] = r.get("caption", "")
    umap, imap, events = {}, {}, []
    with open(matrix, encoding="utf-8-sig", newline="") as f:
        for row_number, r in enumerate(csv.DictReader(f)):
            if row_number >= max_rows:
                break
            uid, iid = int(r["user_id"]), int(r["video_id"])
            if iid not in caption_map or not caption_map[iid]:
                continue
            if (uid not in umap and len(umap) >= max_users) or (iid not in imap and len(imap) >= max_items):
                continue
            u = umap.setdefault(uid, len(umap))
            i = imap.setdefault(iid, len(imap))
            ratio = float(r["watch_ratio"])
            stamp = float(r["timestamp"])
            if not np.isfinite(ratio) or not np.isfinite(stamp):
                raise ValueError("non-finite KuaiRec label/timestamp")
            events.append([u, i, int(stamp*1000), int(ratio >= 2), int(ratio >= 1)])
    if len(events) < 10:
        raise ValueError("too few matching rows; check CSV schema and limits")
    catalog = [{"item_id": i, "source_video_id": original, "language": "unknown", "topic": "unknown",
                "caption": caption_map[original], "published_at": None, "video_path": None,
                "synthetic": False} for original, i in imap.items()]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "dataset.npz", text=np.stack([hash_text(c["caption"]) for c in catalog]),
                        visual=np.zeros((len(catalog), 48), dtype=np.float32),
                        visual_available=np.zeros(len(catalog), dtype=bool),
                        interactions=np.asarray(events, dtype=np.int64), users=np.asarray(len(umap)))
    write_json(out / "catalog.json", catalog)
    write_json(out / "provenance.json", {"source": "KuaiRec bounded CSV subset", "matrix": Path(matrix).name,
        "scan_limit": max_rows, "users": umap, "items": imap, "visual": "unavailable",
        "caption_parse_audit": caption_audit,
        "labels": {"engaged": "watch_ratio>=2 proxy, NOT a native like", "complete": "watch_ratio>=1"},
        "candidate_availability": "publication times unavailable; closed catalog offline evaluation"})
