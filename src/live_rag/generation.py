"""LLM loading and generation (Hugging Face transformers)."""

import torch
from transformers import pipeline


class Generator:
    def __init__(self, model_id: str, max_new_tokens: int = 300):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No GPU detected. Run this on Google Colab (Runtime > Change runtime "
                "type > T4 GPU) or any CUDA machine."
            )
        self.max_new_tokens = max_new_tokens
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            dtype=torch.bfloat16,
            device_map="cuda",
        )

    @staticmethod
    def make_message(system_content: str, user_content: str) -> list[dict]:
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def generate(self, system_content: str, user_content: str) -> str:
        messages = self.make_message(system_content, user_content)
        out = self.pipe(messages, max_new_tokens=self.max_new_tokens, do_sample=False)
        return out[0]["generated_text"][-1]["content"]
