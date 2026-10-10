"""Summarize already-completed predictions without reusing test labels for training."""
import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

from clipcompass.data import write_json

out = Path("artifacts/research/qwen-lora")
report = json.loads((out / "lora_results.json").read_text(encoding="utf-8"))
curve = json.loads((out / "curve.json").read_text(encoding="utf-8"))
selected = max(curve, key=lambda r:r["validation_macro_f1"])
classes = report["classes"]
metrics = {}
for name in ["base_test", "lora_test"]:
    rows = report[name]["predictions"]
    expected, predicted = [r["expected"] for r in rows], [r["predicted"] for r in rows]
    metrics[name] = {"class_report": classification_report(expected, predicted, labels=classes, output_dict=True, zero_division=0),
                     "confusion_matrix": confusion_matrix(expected, predicted, labels=classes).tolist()}
write_json(out / "error_analysis.json", {"classes":classes, "selected_validation_checkpoint":selected, "metrics":metrics,
    "errors": [r for r in report["lora_test"]["predictions"] if r["expected"] != r["predicted"]]})
lines = ["", "## 类别与错误分析", "", f"验证集选出的更新步：{selected['step']}；验证Macro-F1：{selected['validation_macro_f1']:.4f}。", "",
         "| 类别 | 测试样本 | Base F1 | LoRA F1 |", "|---|---:|---:|---:|"]
for category in classes:
    base, adapted = metrics["base_test"]["class_report"][category], metrics["lora_test"]["class_report"][category]
    lines.append(f"| {category} | {int(adapted['support'])} | {base['f1-score']:.4f} | {adapted['f1-score']:.4f} |")
lines += ["", "逐条错误与混淆矩阵保存在error_analysis.json。源类别有噪声，标题也可能不包含舞蹈/颜值等视觉证据；本实验只检验文本分类，不能由标签名称推断模型具备视频理解能力。"]
with (out / "LORA_REPORT.md").open("a", encoding="utf-8") as file:
    file.write("\n".join(lines))
print(json.dumps({"selected_step":selected['step'], "base_macro_f1":report['base_test']['macro_f1'], "lora_macro_f1":report['lora_test']['macro_f1']}))
