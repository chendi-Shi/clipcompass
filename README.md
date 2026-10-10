# ClipCompass · 内容语义增强的短视频冷启动推荐

围绕一个可验证的问题展开：**将内容语义与协同图融合，能否改善新内容推荐，同时保持整体排序质量？** 项目面向 TikTok 内容生态算法 JD，使用真实公开日志、预训练文本表征、图推荐和实际 LoRA 训练。

## 实验入口

- [大白话入门](docs/BEGINNER_GUIDE.md)：用一个推荐做饭视频的例子，讲清模型目标、输入输出、训练方法和实际问题；适合第一次接触这些模型。
- [技术详解](docs/TECHNICAL_WALKTHROUGH.md)：从文字向量到图推荐、BPR/InfoNCE、Qwen LoRA、CPU缓存与服务，逐层解释公式、代码、问题处理和实验边界。
- [复现说明](docs/RESEARCH_RUNBOOK.md)：数据下载校验、编码、训练、评测和服务命令。
- [实验协议](docs/RESEARCH_PROTOCOL.md)：时间划分、冷启动隔离、标签、候选与统计口径。
- [推荐结果](artifacts/research/kuairec/REAL_REPORT.md)：9种方法、3个神经模型随机种子、分组与混合候选指标。
- [结果分析](docs/RESULTS_ANALYSIS.md)：负结果、冷内容平分偏差与选型边界。
- [Qwen LoRA 结果](artifacts/research/qwen-lora/LORA_REPORT.md)：base prompt、实际适配器和分类强基线，说明训练集合与输入长度差异。
- [简历表述](docs/RESUME.md) · [面试讲解](docs/INTERVIEW.md)。

## 实现

| 部分 | 方法与证据 |
|---|---|
| 数据工程 | 完整 KuaiRec small matrix：4,676,570 原始交互，首次曝光去重后4,494,578条；1,411用户、3,327内容。文件校验、原始ID映射、4处未转义换行修复均留存 |
| 推荐模型 | BPR-MF、两层 LightGCN、语义门控图推荐；真实 multilingual E5 编码 caption/封面文字，冷内容屏蔽ID协同分支 |
| 强基线和消融 | 热门、平滑长观看率、TF-IDF/E5历史画像、EASE；门控融合与图/语义 InfoNCE 消融 |
| 评测 | 全局时间70/15/15，314个内容完全隔离训练标签；验证选checkpoint；所有真实测试曝光候选排序，3seed、逐用户bootstrap与joint排序 |
| LLM适配 | Qwen2.5-0.5B-Instruct 最后4层q/v rank8 LoRA，8类文本内容分类；完整词表单回复token监督、冻结前缀/KV缓存及数值一致性检查 |
| 推理 | 固定向量全目录精确检索、历史曝光过滤、HTTP推荐接口、300查询CPU延迟测量；保存可加载模型和真实LoRA适配器 |

结果以已生成报告为准。推荐实验和LoRA实验分别验证，不声称LoRA输出已带来推荐收益。

## 已执行结果

本机CPU完成完整推荐对照、160步Qwen LoRA、分类强基线及实际推理。LoRA测试Macro-F1为0.1894（base 0.0367），E5线性分类器为0.5590，因此另提供可加载的E5分类器和新标题推理。线性基线使用整次160步运行触达的445条内容，最终选中的第120步adapter实际接触365条；两者输入截断长度也不同，不能视为严格相同训练集的架构对比。推荐融合未证明超过强基线，完整负结果与ID平分偏差见结果分析。

推荐向量服务在3327内容目录下300次查询的p95为1.738ms，排除HTTP/编码器/加载；实际HTTP合法请求200、无效用户400。20项测试全部通过。

## 快速复现

Python 3.10+，先安装合适的 PyTorch，再执行：

```powershell
python -m pip install -r requirements-research.txt
$env:PYTHONPATH='.'
python tools/download_kuairec_copy.py
python tools/download_encoder.py --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3
python -m clipcompass.research_data prepare
python -m clipcompass.research_data encode --model data/models/multilingual-e5-small
python -m clipcompass.research_train --epochs 60 --seeds 42 2024 2025
python -m clipcompass.research_joint_eval
python -m unittest discover -s tests -v
```

LoRA和推理复现见[完整说明](docs/RESEARCH_RUNBOOK.md)。机器只有CPU时，真实模型编码和LoRA训练需要较长时间。

## 范围与来源

本项目是公开日志上的离线研究，数据以中文为主。没有原始视频输入、真实MLLM训练、跨语种迁移验证、创作者转化标签或线上A/B结果。长观看由watch_ratio派生，不能写为CTR、点赞或业务转化。

数据来源：[KuaiRec官方仓库](https://github.com/chongminggao/KuaiRec)。仓库不包含完整原始数据文件；分类划分、误差分析和推荐样例含少量派生记录，其来源、修改与CC BY-SA 4.0归属见[数据与模型来源](docs/DATA_SOURCES.md)。方法参考：[LightGCN官方实现](https://github.com/gusye1234/LightGCN-PyTorch)、[EASE论文](https://arxiv.org/abs/1905.03375)、[multilingual E5](https://huggingface.co/intfloat/multilingual-e5-small)、[Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)、[PEFT](https://huggingface.co/docs/peft/en/package_reference/lora)。核心实验代码在本项目实现，不声称复现这些论文的原始榜单数值。

GitHub快照保留研究报告、分类器、LoRA适配器和验证集选出的`content_ssl_seed42.pt`。其余checkpoint按复现命令生成；基础模型和重复tokenizer文件通过固定commit下载。

最初的合成数据演示及尚未实跑的VLM接口保留于[旧版说明](docs/LEGACY_DEMO.md)，不用于本项目简历成果。
