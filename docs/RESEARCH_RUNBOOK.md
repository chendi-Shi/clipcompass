# 研究版复现说明

本目录提供真实 KuaiRec 冷启动推荐与实际 Qwen LoRA 内容分类实验。旧的 `demo-data/train` 只用于快速 smoke test。

## 环境

Python3.10+，先创建并激活独立虚拟环境，安装 CPU 或 CUDA 版 PyTorch，再安装 `requirements-research.txt`。下文的 `python` 均指该环境的解释器。

```powershell
python -m pip install -r requirements-research.txt
$env:PYTHONPATH='.'
```

## 真实数据与表征

```powershell
python tools/download_kuairec.py --range-only
python tools/download_encoder.py --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3
python -m clipcompass.research_data prepare
python -m clipcompass.research_data encode --model data/models/multilingual-e5-small
```

官方服务器较慢时，可使用 `tools/download_kuairec_copy.py` 下载公开的字节一致副本。它必须同时通过副本的 SHA256 和官方 ZIP 中的文件大小/CRC32 校验，才能用于实验。来源与校验写入 `download_manifest.json`；不能宣称验证了整个官方 ZIP 的 MD5。

也可预先对全部10728条 caption 编码，再按真实候选目录选择特征，避免多轮重复编码：

```powershell
python tools/preencode_captions.py
python -m clipcompass.research_data encode --cache data/raw/kuairec/preencoded.npz
```

缓存按原始 video_id 关联，逐条检查实际文本 SHA256。E5 参数冻结，词表/编码器预训练；TF-IDF 词表只从 warm 训练期内容拟合。CSV 的未转义换行修复与缺失文本统计都会记录。

## 推荐实验

```powershell
python -m clipcompass.research_train --epochs 60 --seeds 42 2024 2025
python -m clipcompass.research_joint_eval
python tools/plot_research.py
python tools/audit_cold_ties.py
```

结果位于 `artifacts/research/kuairec/`：`REAL_REPORT.md`、`summary.json`、逐用户 `results.json`、曲线、12个模型 checkpoint。神经模型只按 warm 验证集选择，不根据测试表现修改结构。完整候选/标签/偏差边界见 `RESEARCH_PROTOCOL.md`。

## 实际 LoRA 实验

```powershell
python tools/download_encoder.py --model Qwen/Qwen2.5-0.5B-Instruct --output data/models/qwen2.5-0.5b --revision 7ae557604adf67be50417f59c2c2f167def9a775
python -m clipcompass.research_lora --steps 160
python -m clipcompass.research_lora_baselines
python tools/audit_lora_results.py
python tools/classify_caption.py --text "番茄炒蛋的做法和烹饪技巧" --output artifacts/research/qwen-lora/inference_example.json
python tools/classify_e5.py --text "番茄炒蛋的做法和烹饪技巧" --output artifacts/research/qwen-lora/e5_inference_example.json
```

结果位于 `artifacts/research/qwen-lora/`：`LORA_REPORT.md`、`lora_results.json`、训练/验证曲线、可加载的真实 `adapter/`。这是 **8个常见类别的文本内容理解适配**，类别从训练集频率确定，标签来自原数据一级 taxonomy。按重复文本分组隔离 train/validation/test；含任一 cold-v1 内容的文本组全部进测试。

训练 Qwen2.5-0.5B-Instruct 最后4层的 q/v LoRA（rank8）。监督是 assistant 回复的一个类别代码 token，使用完整词表 causal next-token CE。为降低 CPU 成本，冻结前20层和公共系统prompt KV可缓存；缓存路径与完整模型的 logits 数值一致性先检查，再训练。

报告 base prompt 与 LoRA 的 Accuracy/Macro-F1，以及不限制代码时首 token 合法率；类别预测指标使用固定代码集合的 constrained ranking，不能与开放式 JSON 生成合法率混淆。未做视频、7B/MLLM、大规模生成 SFT、多语种迁移实验。

`research_lora_baselines` 为 TF-IDF/E5 线性分类器复原 LoRA 实际见过的独立标注样本，只使用相同预算，并在验证集选 C。
它同时保存两种分类器的joblib；`classify_e5.py`直接加载已验证的E5分类器进行新标题推理，仅加载可信本项目文件。

## 服务与测量

从 `summary.json` 查看验证集选择的模型，例如 `content_gate`，再使用其 seed42 checkpoint：

```powershell
python -m clipcompass.research_serve --checkpoint artifacts/research/kuairec/content_gate_seed42.pt --benchmark
python -m clipcompass.research_serve --checkpoint artifacts/research/kuairec/content_gate_seed42.pt
```

访问 `http://127.0.0.1:8765/recommend?user=0&k=10`，`cold=true` 可限制 cold 内容。服务采用真实全目录精确向量检索、历史已曝光过滤，不需要在仅3327条内容的目录上虚构百万级 ANN。性能报告明确排除 HTTP、encoder 和加载时间。

## 验证

```powershell
python -m unittest discover -s tests -v
```

测试覆盖数值指标、图归一化、时间/内容隔离、负例采样、真实CSV修复、checkpoint推荐、内容桥接、冻结前缀缓存与 LoRA 梯度。保存 `pip freeze` 和运行命令后，可在同一数据/seed下复现。
