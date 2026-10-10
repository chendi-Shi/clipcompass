"""Attach reviewed VLM summaries and small RGB frame features to a dataset."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from clipcompass.data import hash_text, load_dataset, write_json
from clipcompass.understand import validate_analysis


def attach(directory, manifest, output):
    data = load_dataset(directory)
    rows = json.loads(Path(manifest).read_text(encoding="utf-8"))
    used = set()
    for r in rows:
        i = r["item_id"]
        if type(i) is not int or not 0 <= i < len(data["catalog"]) or i in used:
            raise ValueError("invalid or duplicate item ID")
        used.add(i)
        if r.get("reviewed") is not True:
            raise ValueError("human-reviewed analysis required")
        analysis = validate_analysis(r["analysis"], len(r["frames"]))
        arrays = []
        for file in r["frames"]:
            with Image.open(file) as image:
                arrays.append(np.asarray(image.convert("RGB").resize((4, 4)), dtype=np.float32).reshape(-1)/255)
        data["visual"][i] = np.mean(arrays, axis=0)
        data["visual_available"][i] = True
        data["text"][i] = hash_text(analysis["summary"], data["text"].shape[1])
        data["catalog"][i].update({"caption": analysis["summary"], "language": analysis["language"],
                                  "analysis": analysis, "content_source": "reviewed VLM + RGB frames"})
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "dataset.npz", **{k: data[k] for k in ["text", "visual", "visual_available", "interactions", "users"]})
    write_json(out / "catalog.json", data["catalog"])
    data["provenance"]["attached_content"] = {"items": len(used), "visual": "mean 4x4 RGB, elementary baseline; not CLIP/MLLM embeddings"}
    write_json(out / "provenance.json", data["provenance"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    attach(a.data, a.manifest, a.output)
