# ClipCompass · 内容语义增强的短视频冷启动推荐

**主研究版入口：[真实数据复现说明](RESEARCH_RUNBOOK.md) · [实验协议](RESEARCH_PROTOCOL.md)。**

研究版使用完整 KuaiRec small matrix（4676570原始交互，1411用户，3327内容），对比 EASE、BPR-MF、LightGCN、真实 multilingual E5 与语义门控图推荐，按时间与内容隔离评测 warm/cold 效果，运行3个随机种子。另提供实际 Qwen2.5-0.5B 的8类文本分类 LoRA 实验及冻结前缀/KV缓存优化。

真实实验输出位于 `artifacts/research/`；运行状态及数值以对应报告文件为准。视频 MLLM、开放式生成和创作者转化实验仍需另行执行，不能由这些文本实验替代。

下面保留最初的 CPU smoke demo 与 MLLM 数据接口说明，**其模拟结果不用于新版简历成果**。

面向 TikTok 内容生态算法实习 JD 的独立研究原型。目标是把短视频内容理解、行为建模和创作灵感推荐连成可复现闭环。核心算法由本项目实现，开源仓库作为数据来源、方法参考和大模型工具链。

**以下章节是旧的模拟演示路径。** 真正的研究版数据、训练、LoRA、服务及结果请使用上方入口；VLM 桥接依然只通过接口/数据测试，尚未运行真实视频理解。

## 与 JD 的对应

| JD 能力 | 项目实现 | 当前证据 |
|---|---|---|
| 推荐业务、用户体验 | 双塔 top50 召回 → engagement/completion 多任务排序 → MMR 多样性 | 推荐 JSON、离线 Recall/NDCG/coverage |
| 深度学习、Learning to rank | PyTorch 内容/用户编码器；BCE + 同用户曝光对 BPR | 训练曲线与消融 |
| 自监督、对比学习 | 同一内容的文本/RGB 视图对称 InfoNCE | 有无 SSL 对照；缺失模态掩码 |
| 模型压缩、蒸馏 | 32 维教师 → 12 维学生，温度为 2 的软标签蒸馏 | 参数量及学生效果；不是 LLM 蒸馏 |
| LLM/MLLM 应用 | 本地兼容接口接收关键帧、caption/ASR/OCR 上下文，输出带帧引用的 JSON | mock transport/schema 测试；待真实模型验证 |
| Prompt / SFT / Fine-tuning | 统一内容理解 prompt、人工复核标签导出、Qwen3-VL LoRA 配置 | 配置和导出测试；待 GPU 训练 |
| 多语种 | 中/英/西语模拟内容；按正例内容语言分组评测 | 机制验证；不是百语种能力证明 |

GNN、continuous pretraining、线上 A/B 实验不在当前实现范围。JD 的研究方向不是要求每种技术都堆进一个项目。

## 快速运行

Python 3.10+。在此目录执行：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .
.venv/Scripts/python.exe -m pip install -r requirements-content.txt
.venv/Scripts/python.exe -m clipcompass.cli demo-data
.venv/Scripts/python.exe -m clipcompass.cli train --epochs 80
.venv/Scripts/python.exe -m clipcompass.cli recommend --user 0 --output artifacts/recommendations.json
.venv/Scripts/python.exe -m unittest discover -s tests -p test_core.py -v
```

也可在已激活并安装依赖的 Python 3.10+ 环境中执行：

```powershell
$env:PYTHONPATH='.'
python -m clipcompass.cli demo-data
python -m clipcompass.cli train --epochs 80
python -m clipcompass.cli recommend --user 0 --output artifacts/recommendations.json
```

训练默认 CPU、固定 seed；80 epochs 完成 4 个模型训练和 popularity 基线。生成 `artifacts/metrics.json`、`REPORT.md`、`learning_curves.json`、教师/学生 checkpoint。模型文件只从可信来源加载。

Windows 也可直接运行 `./run_demo.ps1`；会依次检查项目 `.venv`、`.research-venv` 和当前 `python` 的 PyTorch 依赖。

## 数据和实验协议

模拟数据：96 用户、240 内容、3 语种、6 个主题；内容是预设 caption 和生成的 4×4 RGB 帧，行为由合成偏好采样。**没有真实短视频，RGB 和字符 hash 是轻量基线特征，不是 CLIP 或语义大模型 embedding。** Topic 不进入推荐特征，语言也不作为真值作弊特征。

按全局 timestamp 70/15/15 划分，严格保留同一 timestamp 的事件分组。用户历史只由训练正例构建。BPR 使用同用户训练曝光中的正/负对；SSL 只使用训练期已出现的内容；根据验证 BCE 选 checkpoint，测试仅报告效果。采用全目录排名，屏蔽训练和验证已曝光项以及测试开始后发布项，无合格正例的用户排除并报告。

Recall@10 衡量正例召回比例，NDCG@10 衡量正例排名质量，coverage@10 衡量目录覆盖。语言分组是固定全目录排序下、按正例语言切分，不是各语言内部排序。测试指标用于复现与诊断，不能根据这一次测试挑超参。

演示服务先 top50 再排序/MMR；离线报告是全目录 combined score，**尚未评测 top50 级联和 MMR 后的业务增益**。completion 和 engagement 由共同观看过程派生，不能宣称独立任务监督；输出概率也未做校准。

## 接入真实 KuaiRec

从 [KuaiRec 官方仓库](https://github.com/chongminggao/KuaiRec) 的下载区获得 `small_matrix.csv` 和 `kuairec_caption_category.csv`，遵守其 CC BY-SA 4.0 数据许可与引用要求。项目不再分发数据。

```powershell
python -m clipcompass.cli import-kuairec --matrix D:/datasets/KuaiRec/data/small_matrix.csv --captions D:/datasets/KuaiRec/data/kuairec_caption_category.csv --output data/kuairec
python -m clipcompass.cli train --data data/kuairec --output artifacts/kuairec --epochs 80
```

默认最多扫描 300000 行、96 用户、600 内容，方便 CPU 试跑；这是文件顺序限制的子集，不是无偏抽样。数据原始 video_id/user_id 映射保留在 provenance。`engaged=watch_ratio>=2` 是重观看代理，`complete=watch_ratio>=1`。KuaiRec 不提供原生点赞，也不在此流程提供视频文件；视觉缺失置零并从对比损失排除。此时多模态实验不能被解读为真实视觉增益。无出版时间的真实数据采用固定封闭目录，不能宣称满足线上 availability。

## 真正的 MLLM 路径

本地部署 Qwen3-VL 兼容服务后，传入自己的、有权使用的视频关键帧。可用 ffmpeg 抽帧：

```powershell
ffmpeg -i own_video.mp4 -vf "fps=1/5,scale=384:-1" -frames:v 16 data/frames/%03d.jpg
python tools/analyze_frames.py --frames data/frames/001.jpg data/frames/002.jpg --context "原始标题和已有 ASR/OCR 文本" --output data/analysis.json
```

依赖服务端支持多图 `image_url` 请求；默认 `http://127.0.0.1:8000/v1`，可设置 `--base-url`、`--model`，需要认证时从环境变量 `VLM_API_KEY` 读取。客户端最多16帧，每帧4MiB；拒绝不存在的证据帧索引，JSON 错误会明确失败。ASR/OCR 文本是外部输入，项目未实现音频转写。

模型返回的证据需要人工核对；帧索引有效不等于描述真实。生成灵感目前使用引用原始 caption 的固定模板，明确标记为模板输出，未声称 LLM 生成。

人工复核后，可把内容理解结果与真实 RGB 视图接入推荐。`content_manifest.json` 为数组，每行包含 `item_id`、`reviewed:true`、`frames:[绝对路径]`、`analysis:{language,summary,topics,evidence}`，其中 item_id 是导入后的内部 ID；只能使用推荐目标期之前可获得的内容信息：

```powershell
python -m pip install Pillow
python tools/attach_content.py --data data/kuairec --manifest data/content_manifest.json --output data/kuairec-content
python -m clipcompass.cli train --data data/kuairec-content --output artifacts/real-content
```

该桥接用复核摘要替换字符特征，用帧平均的4×4 RGB 替换零向量；仍是轻量基线，不是预训练视觉 encoder。

## SFT / LoRA 路径

参考 [LlamaFactory 官方 Qwen3-VL 配置](https://github.com/hiyouga/LlamaFactory/blob/main/examples/train_lora/qwen3vl_lora_sft.yaml)，使用 `Qwen/Qwen3-VL-4B-Instruct`。在 GPU Linux 环境按官方安装指引部署；此处不承诺特定显存下必然成功。

人工标注 JSON 数组格式：

```json
[
  {
    "video_id": "video_train_001", "split": "train", "reviewed": true,
    "frames": ["/absolute/path/train_001.jpg"], "context": "原始标题",
    "answer": {"language": "zh", "summary": "一只猫拨动玩具球。", "topics": ["pets"],
      "evidence": [{"frame_index": 0, "observation": "猫爪靠近玩具球"}]}
  },
  {
    "video_id": "video_eval_001", "split": "eval", "reviewed": true,
    "frames": ["/absolute/path/eval_001.jpg"], "context": "原始标题",
    "answer": {"language": "zh", "summary": "厨师在锅中翻动面条。", "topics": ["food"],
      "evidence": [{"frame_index": 0, "observation": "锅中可见面条"}]}
  }
]
```

```powershell
python tools/export_sft.py --input data/reviewed_labels.json --output data/sft
llamafactory-cli train configs/qwen3vl_lora_sft.yaml
```

train/eval 必须按视频独立分开；禁止重复 video_id，两组都必须存在。真实数据还应按作者/近重复视频分组避免泄漏。演示的两个样本仅说明格式，不能训练出可信模型。

建议至少构建几百条真实且人工复核的标签，留出独立内容理解测试集。比较 base prompt 与 LoRA 的 JSON 合法率、topic macro-F1、证据帧准确率、人工幻觉率和分语种结果，再比较真实推荐结果。**仅导出 SFT 数据不能称为已完成模型调优。**

## 开源依据与项目边界

| 开源仓库 | 选择原因 | 本项目使用方式 |
|---|---|---|
| [chongminggao/KuaiRec](https://github.com/chongminggao/KuaiRec) | 真实短视频观看行为及 caption/category | CSV schema/数据接入；尚未下载和训练完整数据 |
| [enoche/MMRec](https://github.com/enoche/MMRec) | 多模态推荐与自监督方法对照 | 研究参考；未复制框架源码 |
| [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) | 当前官方多模态模型与部署接口 | 本地兼容服务；尚未下载权重 |
| [hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory) | 多模态 LoRA 工具链 | ShareGPT 导出及训练配方 |

查阅日期：2026-10-10。旧 Qwen2.5-VL 仓库目前指向更新的官方 VL 项目，因此配方使用实际核验的 Qwen3-VL 示例。第三方模型/框架/数据各自许可单独适用；本项目未复制第三方实现，也未携带模型权重。

目录：`clipcompass/` 算法；`tools/` 内容理解和 SFT 数据工具；`configs/` GPU 配方；`tests/` 正确性验证；`artifacts/` 实测结果；`docs/INTERVIEW.md` 面试讲解与后续实验。
