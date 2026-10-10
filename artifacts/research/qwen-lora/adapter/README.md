---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
language:
  - zh
tags:
  - lora
  - text-classification
---

# ClipCompass Qwen content classifier

An executed, CPU-trained LoRA adapter for constrained classification of Chinese short-video captions and cover text into eight KuaiRec content categories. This adapter consumes text only; it has not been trained or evaluated on video frames.

## Training

- Base model: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), pinned revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- PEFT LoRA on `q_proj` and `v_proj` in the final four transformer layers (20–23): rank 8, alpha 16, dropout 0.05; **90,112 trainable parameters**.
- FP32 CPU training, seed 42, batch size 4, 160 optimization steps. The full run visited 445 unique training examples; validation Macro-F1 selected the saved checkpoint at **step 120**, which had seen 365 unique examples.
- Supervision: full-vocabulary causal cross-entropy at one assistant category-code token. Inference ranks the eight permitted codes; this is not open-ended generation.
- Dataset: KuaiRec caption/category metadata; duplicate normalized text groups are isolated across splits. Labels are released taxonomy labels and may be noisy. The eight classes are selected using training frequencies.
- Frozen prefix and shared prompt KV caching reduce CPU computation. Tests check forward equivalence and LoRA gradients.

## Evaluation and limits

On the fixed 160-example held-out test set, constrained Macro-F1 is **0.1894**, versus **0.0367** for the base prompt. The E5 plus linear-classifier baseline reaches **0.5590**, so this experiment does not establish that LoRA is the best classifier. The run uses one seed and a small labeling/training budget; results do not demonstrate video understanding, multilingual transfer, or a recommendation gain.

The linear baselines use the 445 unique labeled examples visited during the full 160-step LoRA run, which is 80 more than the 365 examples seen by the selected step-120 adapter. They therefore match the full run's label-access budget, not the selected adapter's exact training set. Inputs also differ: the linear baselines read the full released text (E5 is limited to 128 tokens), while Qwen receives the first 96 characters. The comparison does not isolate architecture effects under identical training sets, input lengths, or compute budgets.

See the [full experiment report](../LORA_REPORT.md), [recorded metrics](../lora_results.json), and [classification baselines](../classification_baselines.json) for exact protocols and results.

## Loading

Download the pinned base model using the [research runbook](../../../../docs/RESEARCH_RUNBOOK.md), then run from the repository root:

```powershell
python tools/classify_caption.py --text "番茄炒蛋的做法和烹饪技巧"
```

The inference script loads the base model explicitly and attaches this adapter with `PeftModel.from_pretrained`. Scores are uncalibrated. Base-model weights are downloaded separately.

## Sources

[KuaiRec](https://github.com/chongminggao/KuaiRec) supplies the content metadata. The base model and tokenizer come from Qwen; the adapter was trained using [PEFT](https://huggingface.co/docs/peft/en/package_reference/lora) 0.14.0. Dataset and base-model terms remain applicable; consult the repository's source and attribution documentation.
