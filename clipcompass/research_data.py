"""Full real-data benchmark; separate from the original synthetic smoke demo."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import caption_records, write_json


def load_captions(path):
    """Repair unquoted physical line breaks in the official caption CSV, audit count."""
    rows, audit = caption_records(path)
    frame = pd.DataFrame(rows).replace("UNKNOWN", "")
    frame["video_id"] = frame.video_id.astype(int)
    return frame, audit


def prepare(raw, output):
    raw, out = Path(raw), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(raw / "small_matrix.csv", usecols=["user_id", "video_id", "timestamp", "watch_ratio"])
    raw_count = len(frame)
    frame = frame.dropna().sort_values("timestamp", kind="stable").drop_duplicates(["user_id", "video_id"], keep="first")
    if not np.isfinite(frame[["timestamp", "watch_ratio"]].to_numpy()).all():
        raise ValueError("nonfinite raw values")
    uids, iids = np.sort(frame.user_id.unique()), np.sort(frame.video_id.unique())
    u = np.searchsorted(uids, frame.user_id.to_numpy()).astype(np.int32)
    i = np.searchsorted(iids, frame.video_id.to_numpy()).astype(np.int32)
    times = frame.timestamp.to_numpy(dtype=np.float64)
    ratio = frame.watch_ratio.to_numpy(dtype=np.float32)
    cut1, cut2 = np.quantile(times, [.7, .85])
    split = np.where(times < cut1, 0, np.where(times < cut2, 1, 2)).astype(np.int8)
    # Independent, fixed item holdout. Identical across model seeds.
    cold = np.array([int.from_bytes(hashlib.sha256(f"cold-v1:{int(original)}".encode()).digest()[:4], "little") % 10 == 0 for original in iids])
    np.savez_compressed(out / "events.npz", user=u, item=i, timestamp=times, watch_ratio=ratio,
                        split=split, cold=cold, original_users=uids, original_items=iids)
    captions, caption_audit = load_captions(raw / "kuairec_caption_category.csv")
    mapping = {int(r["video_id"]): r for r in captions.to_dict("records")}
    catalog = []
    for index, original in enumerate(iids):
        r = mapping.get(int(original), {})
        caption = str(r.get("caption", ""))
        cover = str(r.get("manual_cover_text", ""))
        catalog.append({"item_id": index, "video_id": int(original), "caption": caption, "cover_text": cover,
                        "text": (caption + " " + cover).strip() or "无文本内容",
                        "category": str(r.get("first_level_category_name", "unknown")) or "unknown",
                        "cold": bool(cold[index])})
    write_json(out / "catalog.json", catalog)
    train = (split == 0) & ~cold[i]
    info = {"source": "KuaiRec official small_matrix full file", "raw_rows": raw_count, "retained_rows": len(u),
            "users": len(uids), "items": len(iids), "cold_items": int(cold.sum()),
            "cutoffs_unix_seconds": [float(cut1), float(cut2)],
            "rows_by_time": {name: int((split == s).sum()) for s, name in enumerate(["train", "validation", "test"])},
            "training_rows_after_cold_holdout": int(train.sum()), "training_positive_rows": int((train & (ratio >= 2)).sum()),
            "missing_caption_items": sum(not c["caption"] for c in catalog),
            "caption_parse_audit": caption_audit,
            "deduplication": "first chronological exposure per user/item; no future-label aggregation",
            "positive_label": "watch_ratio >= 2, long-watch proxy", "train_negative_label": "watch_ratio < 1",
            "cold_split": "SHA256 cold-v1:original_video_id mod10==0; all its train/validation labels excluded from optimization/selection",
            "evaluation": "rank ALL observed held-out exposures for each user within warm/cold item groups; no sampled test negatives; retrospective candidate set",
            "raw_manifest": json.loads((raw / "download_manifest.json").read_text()),
            "limitations": ["mostly Chinese; not proof of cross-lingual transfer", "no raw videos or vision features",
                            "captions are release snapshots; original-time versions unknown", "logged exposures retain policy bias",
                            "retrospective exposure-ranking, not an online candidate-availability simulation"]}
    write_json(out / "protocol.json", info)
    print(json.dumps({k: info[k] for k in ["raw_rows", "retained_rows", "users", "items", "cold_items", "training_positive_rows"]}), flush=True)


def load(directory):
    path = Path(directory)
    with np.load(path / "events.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["catalog"] = json.loads((path / "catalog.json").read_text(encoding="utf-8"))
    data["protocol"] = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
    data["shape"] = (len(data["original_users"]), len(data["original_items"]))
    return data


def training_matrix(data):
    mask = (data["split"] == 0) & ~data["cold"][data["item"]] & (data["watch_ratio"] >= 2)
    return sp.csr_matrix((np.ones(int(mask.sum()), dtype=np.float32), (data["user"][mask], data["item"][mask])), shape=data["shape"])


def embed_texts(texts, model_name="intfloat/multilingual-e5-small", revision=None):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    encoder = AutoModel.from_pretrained(model_name, revision=revision, trust_remote_code=False)
    encoder.eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), 24):
            # E5 symmetric feature extraction uses the query prefix for both sides.
            batch = tokenizer(["query: " + text for text in texts[start:start+24]], padding=True, truncation=True, max_length=128, return_tensors="pt")
            hidden = encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).bool()
            pooled = hidden.masked_fill(~mask, 0).sum(1) / mask.sum(1).clamp(min=1)
            vectors.append(torch.nn.functional.normalize(pooled, dim=-1).numpy())
            if start % 240 == 0:
                print(f"E5 encoding {start}/{len(texts)}", flush=True)
    manifest = {"model": model_name, "resolved_revision": getattr(encoder.config, "_commit_hash", revision),
               "pooling": "attention-masked mean + L2", "prefix": "query: ", "max_length": 128,
               "trainable_encoder": False, "text_source": "caption + manual_cover_text; no category added",
               "tfidf_fit": "only warm training-observed items"}
    local_manifest = Path(model_name) / "encoder_download_manifest.json"
    if local_manifest.exists():
        manifest = json.loads(local_manifest.read_text(encoding="utf-8"))
        manifest = {"model": manifest["model"], "resolved_revision": manifest["revision"],
                   "pooling": "attention-masked mean + L2", "prefix": "query: ", "max_length": 128,
                   "trainable_encoder": False, "text_source": "caption + manual_cover_text; no category added",
                   "tfidf_fit": "only warm training-observed items", "weight_verification": manifest["verification"]}
    return np.concatenate(vectors).astype(np.float32), manifest


def encode(directory, model_name="intfloat/multilingual-e5-small", revision=None, cache=None):
    data = load(directory)
    out = Path(directory)
    from sklearn.feature_extraction.text import TfidfVectorizer
    train_items = np.unique(data["item"][(data["split"] == 0) & ~data["cold"][data["item"]]])
    texts = [c["text"] for c in data["catalog"]]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), max_features=12000, min_df=2, sublinear_tf=True)
    vectorizer.fit([texts[i] for i in train_items])
    sp.save_npz(out / "tfidf.npz", vectorizer.transform(texts).astype(np.float32))
    if cache:
        with np.load(cache, allow_pickle=False) as values:
            indices = {int(i): pos for pos, i in enumerate(values["video_ids"])}
            selected = [indices[c["video_id"]] for c in data["catalog"]]
            if not all(values["text_hashes"][pos] == hashlib.sha256(text.encode()).hexdigest() for pos, text in zip(selected, texts)):
                raise ValueError("Embedding cache/caption mismatch")
            features = values["features"][selected]
        manifest = json.loads(Path(str(cache)+".json").read_text(encoding="utf-8"))
    else:
        features, manifest = embed_texts(texts, model_name, revision)
    np.save(out / "e5.npy", features)
    write_json(out / "feature_manifest.json", manifest)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "encode"])
    p.add_argument("--raw", default="data/raw/kuairec")
    p.add_argument("--data", default="data/research/kuairec")
    p.add_argument("--revision")
    p.add_argument("--model", default="intfloat/multilingual-e5-small")
    p.add_argument("--cache")
    a = p.parse_args()
    prepare(a.raw, a.data) if a.command == "prepare" else encode(a.data, model_name=a.model, revision=a.revision, cache=a.cache)
