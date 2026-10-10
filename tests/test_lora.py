import unittest

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from clipcompass.research_lora import PrefixCache, response_logits


class LoraTests(unittest.TestCase):
    def test_cached_prefix_parity_and_adapter_gradients(self):
        torch.manual_seed(42)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                             num_attention_heads=4, num_key_value_heads=2, attention_dropout=0., pad_token_id=0)
        model = Qwen2ForCausalLM(config).eval()
        inputs = {"input_ids": torch.tensor([[4, 5, 6, 7], [4, 9, 0, 0]]),
                  "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])}
        cache = PrefixCache(start=2, shared_ids=[4])
        with torch.no_grad():
            direct, cached = response_logits(model, inputs), response_logits(model, inputs, cache)
        self.assertTrue(torch.allclose(direct, cached, atol=1e-5, rtol=1e-4))
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                                              layers_to_transform=[2, 3], task_type="CAUSAL_LM"))
        model.train()
        loss = torch.nn.functional.cross_entropy(response_logits(model, inputs, cache), torch.tensor([10, 11]))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad))
        self.assertTrue(all("layers.2." in n or "layers.3." in n for n,p in model.named_parameters() if p.requires_grad))


if __name__ == "__main__":
    unittest.main()
