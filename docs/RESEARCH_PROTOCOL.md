# 真实数据研究协议（v2）

主问题：**语义内容特征与协同图结合后，能否在保持 warm-item 推荐质量的同时改善 item cold start？**

实验在 KuaiRec 原始 `small_matrix.csv` 完整文件上执行，禁止旧演示的96用户截断采样。先读完整交互，按首次用户/内容曝光去重，再全局按时间70/15/15划分。约10%的预设 SHA256 item holdout 不进入训练或验证调参；实际约10%，数量以运行报告为准。

该冷启动是按内容 ID 的模拟冷启动：从训练中完全隐藏这批内容的行为，但推理时可以使用其发布的文本。它不是所有新发布视频的线上实验，不包含用户冷启动。

## 对照

- Popularity：训练长观看计数。
- Smoothed long-watch rate：以训练曝光为分母的平滑长观看率，避免只与热门计数这个较弱基线对比。
- TF-IDF：训练期 warm 内容拟合字符词表，用户历史内容向量聚合。
- Multilingual E5：真实预训练 encoder 提取 caption/封面文字 embedding，用户正例历史构建语义画像。模型权重 hash 和 commit 均记录，不是假特征。
- EASE：闭式协同推荐强基线，在验证集选择 lambda。
- BPR-MF：纯 ID 协同过滤。
- LightGCN：仅训练正例构图，两层归一化邻居聚合。
- Content Gate：LightGCN item embedding 与可学习语义投影融合，加入正例历史语义用户画像；cold item 强制使用语义分支。
- Content SSL：在相同结构上加入图/语义表征 InfoNCE，用于检查对齐是否有效。

神经模型使用相同 embedding dim=32、lr=0.02、60轮采样更新预算和3个 seed。代码参数名为 `epochs`，每轮每个合格用户抽128组正负对，执行一次全图前向、反向和Adam更新；不是完整遍历全部曝光60遍。此为固定实验预算，不代表所有基线经过充分超参搜索。各模型只通过 warm 验证 NDCG@20 选择 checkpoint，cold 验证标签不参与方法选择。

## 标签与候选

`watch_ratio>=2` 是长观看代理正例；训练负例为真实曝光且 `watch_ratio<1`。中间区间不参与 BPR 正负对，但在测试中仍是非正例候选。同用户采样，绝不把未曝光内容臆造为已知负反馈。

评测对每个用户、每个 warm/cold 分组，排名其测试期**全部实际曝光内容**。没有100个采样负例，没有测试图边回流，没有将训练已曝光项混入测试候选。只评估拥有可训练正负历史的用户，缺少测试正例的用户单独排除和报告。

这是一种对已记录曝光的回顾性排序评估。测试候选集合依赖日志曝光，不能作为真实线上候选策略，也不等于已解决推荐曝光偏差或因果推断。Caption 是数据发布快照，原始时刻版本无法核验，应公开这一限制。

报告 warm/cold Recall@20、NDCG@20、用户数量、候选规模，三seed均值与样本标准差；对 Content Gate 与 LightGCN 逐用户配对 bootstrap 2000次，先平均seed再重采样用户。bootstrap区间不代表线上统计显著性。

补充混合 warm/cold 曝光候选的 joint 排序指标，检查冷启动分支分数尺度是否伤害整体排序。这个补充测试不参与模型选择或超参修改。

研究依据：[KuaiRec 原论文](https://arxiv.org/abs/2202.10842)、[LightGCN 官方实现](https://github.com/gusye1234/LightGCN-PyTorch)、[EASE 原论文](https://arxiv.org/abs/1905.03375)、[Multilingual E5 模型卡](https://huggingface.co/intfloat/multilingual-e5-small)、[Qwen2.5 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)、[PEFT LoRA 文档](https://huggingface.co/docs/peft/en/package_reference/lora)。本项目实现与数据协议独立，不声称复现了原论文的原始基准数值。

## 与 JD 的关系

已执行后可证明的能力：真实推荐数据工程、内容表征应用、图神经网络、Learning to rank、对比学习、消融分析、评测防泄漏、推理部署与性能测量。

LLM微调证据以独立Qwen0.5B实验报告及实际adapter为准；E5本身保持冻结。视频理解、MLLM、多语种迁移和创作转化推荐仍无真实实验，旧VLM接口不计入已完成训练成果。

