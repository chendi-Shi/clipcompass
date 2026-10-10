"""Strong classification baselines with the exact LoRA unique-label budget."""
import argparse
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from .data import write_json


@threadpool_limits.wrap(limits=2)
def run(directory, cache):
    out = Path(directory)
    report = json.loads((out / "lora_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
    splits = manifest["splits"]
    rng = np.random.default_rng(report["seed"])
    touched = set()
    for _ in range(report["steps"]):
        touched.update(int(i) for i in rng.choice(len(splits["train"]), report["batch_size"], replace=False))
    if len(touched) != report["unique_train_examples_seen"]:
        raise ValueError("Cannot reconstruct LoRA label budget")
    train = [splits["train"][i] for i in sorted(touched)]
    val, test = splits["validation"], splits["test"]
    classes = report["classes"]
    labels = [np.array([classes.index(r["category"]) for r in rows]) for rows in [train, val, test]]
    lexical = TfidfVectorizer(analyzer="char", ngram_range=(1,3), max_features=12000, min_df=2, sublinear_tf=True)
    matrices = {"tfidf_logistic": [lexical.fit_transform([r["text"] for r in train]), lexical.transform([r["text"] for r in val]), lexical.transform([r["text"] for r in test])]}
    with np.load(cache, allow_pickle=False) as values:
        mapping = {int(i): pos for pos, i in enumerate(values["video_ids"])}
        matrices["e5_logistic"] = [values["features"][[mapping[r["video_id"]] for r in rows]] for rows in [train, val, test]]
    results = {}
    for name, features in matrices.items():
        best, classifier, selected = -1., None, None
        for c in [.1, 1., 10.]:
            model = LogisticRegression(C=c, class_weight="balanced", max_iter=500, random_state=42).fit(features[0], labels[0])
            metric = f1_score(labels[1], model.predict(features[1]), labels=list(range(len(classes))), average="macro", zero_division=0)
            if metric > best:
                best, classifier, selected = metric, model, c
        predicted = classifier.predict(features[2])
        joblib.dump({"classifier": classifier, "classes": classes,
                     "vectorizer": lexical if name == "tfidf_logistic" else None,
                     "input": "released caption + manual cover text; frozen E5 query prefix + masked mean + L2 max128 tokens"}, out / f"{name}.joblib")
        results[name] = {"accuracy": float(accuracy_score(labels[2], predicted)),
                         "macro_f1": float(f1_score(labels[2], predicted, labels=list(range(len(classes))), average="macro", zero_division=0)),
                         "selected_C_by_validation": selected, "validation_macro_f1": float(best), "train_unique_examples": len(train)}
    write_json(out / "classification_baselines.json", {"matching": "same unique labeled training content actually seen by LoRA, reconstructed from fixed RNG; no extra labels",
               "input_scope": "linear baselines use full released caption/cover text (E5 max128 tokens); Qwen prompt uses first96 characters; matched labels, not identical input length",
               "train_count": len(train), "test_count": len(test), "results": results})
    lines = ["", "## 同标注预算的强基线", "", "| 方法 | Accuracy | Macro-F1 |", "|---|---:|---:|"]
    for name, values in results.items():
        lines.append(f"| {name} | {values['accuracy']:.4f} | {values['macro_f1']:.4f} |")
    lines += ["", "两种线性基线只使用 LoRA 实际见过的相同独立训练内容；C 由验证集选择。线性基线读取完整发布文本（E5最多128token），Qwen使用前96字符；标注预算匹配，但输入长度不同。LoRA 不保证优于预训练表征+线性分类器。"]
    path = out / "LORA_REPORT.md"
    content = path.read_text(encoding="utf-8")
    marker = "\n## 同标注预算的强基线"
    if marker in content:
        start = content.index(marker)
        end = content.find("\n## ", start+len(marker))
        content = content[:start]+(content[end:] if end >= 0 else "")
    path.write_text(content.rstrip()+"\n"+"\n".join(lines), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--directory", default="artifacts/research/qwen-lora")
    p.add_argument("--cache", default="data/raw/kuairec/preencoded.npz")
    a = p.parse_args()
    run(a.directory, a.cache)
