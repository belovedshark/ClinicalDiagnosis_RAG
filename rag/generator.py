# rag/generator.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .config import LLM_MODEL, DEVICE
import torch

class Generator:
    """Gemma-based text generation module."""

    def __init__(self):
        print(f"Loading Gemma model: {LLM_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map={"": "cpu"}
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(self, context: str, question: str) -> str:
        """Generate an answer grounded in retrieved context."""
        prompt = f"""
You are a clinical assistant. Answer the question based only on the context below.

Context:
{context}

Question: {question}
Answer:
"""
        output = self.pipe(
            prompt,
            max_new_tokens=128,
            temperature=0.3,
            top_p=0.9,
        )[0]["generated_text"]

        # Return the model's answer
        return output.split("Answer:")[-1].strip()