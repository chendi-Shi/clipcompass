# 数据与模型来源

## KuaiRec

来源为Chongming Gao等作者公开的[KuaiRec项目](https://github.com/chongminggao/KuaiRec)及[论文 KuaiRec: A Fully-observed Dataset and Insights for Evaluating Recommender Systems](https://arxiv.org/abs/2202.10842)。数据按[Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/)提供。

本仓库不包含完整small_matrix.csv、完整caption表或视频文件。`artifacts/research/`中的分类划分、预测错误、推荐样例与模型catalog包含该数据的少量派生记录，保留上述来源与CC BY-SA 4.0归属；该归属不表示原作者背书本项目。

处理包括：修复4处未转义换行、去除UNKNOWN占位符、合并标题和封面文字、首次曝光去重、时间和内容划分、派生长观看标签及模型预测。源内容类别是公开弱标签，不是本项目新增人工标注。推荐指标与类别预测由本项目计算。

原始文件来源、镜像revision、SHA256及官方ZIP条目CRC32记录于实验的`summary.json`；下载脚本必须执行校验。完整官方ZIP的MD5未被本次流程验证。

## 基础模型

- [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)，实验commit `614241f622f53c4eeff9890bdc4f31cfecc418b3`。
- [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，实验commit `7ae557604adf67be50417f59c2c2f167def9a775`。

仓库只保存本次训练的LoRA增量参数和分类器，不重新分发基础模型权重；使用基础模型请遵循其模型卡和许可。LoRA的实际任务、结果和限制见[模型卡](../artifacts/research/qwen-lora/adapter/README.md)。
