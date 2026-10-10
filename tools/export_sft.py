"""Export reviewed frame understanding labels to LlamaFactory ShareGPT format."""
import argparse
import json
from pathlib import Path

from clipcompass.data import write_json
from clipcompass.understand import PROMPT, validate_analysis


def export(input_path, output):
    rows = json.loads(Path(input_path).read_text(encoding="utf-8"))
    records, evaluation, groups = [], [], set()
    for r in rows:
        if r.get("reviewed") is not True:
            raise ValueError("every SFT label must have reviewed=true")
        group = r["video_id"]
        if group in groups:
            raise ValueError("one record per video; prevent duplicate video leakage")
        groups.add(group)
        if r.get("split") not in {"train", "eval"}:
            raise ValueError("explicit train/eval split required")
        paths = [str(Path(f).resolve()) for f in r["frames"]]
        if not paths or not all(Path(f).is_file() for f in paths):
            raise ValueError("frame files missing")
        answer = validate_analysis(r["answer"], len(paths))
        entry = {"messages": [{"role": "user", "content": "<image>"*len(paths) + PROMPT + "\nContext: " + r.get("context", "")},
                              {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}], "images": paths}
        (records if r["split"] == "train" else evaluation).append(entry)
    if not records or not evaluation:
        raise ValueError("both training and held-out videos required")
    out = Path(output)
    write_json(out / "clip_train.json", records)
    write_json(out / "clip_eval.json", evaluation)
    write_json(out / "dataset_info.json", {name: {"file_name": file, "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}
        for name, file in [("clip_train", "clip_train.json"), ("clip_eval", "clip_eval.json")]})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="data/sft")
    a = p.parse_args()
    export(a.input, a.output)
