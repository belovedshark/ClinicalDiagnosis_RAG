# rag/generator.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .config import LLM_MODEL
import torch


class Generator:
    """Gemma-based text generation module.

    Loads the LLM onto GPU when available using `device_map='auto'` and
    mixed precision where appropriate. Falls back to CPU if GPU loading
    fails.
    """

    def __init__(self, device: str = "cpu"):
        print(f"Loading Gemma model: {LLM_MODEL} (preferred device={device})")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

        # Choose dtype based on device
        if device == "cuda":
            dtype = torch.float16
        elif device == "mps":
            # MPS has limited float16 support; use bfloat16 or float32
            dtype = torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float32
        else:
            dtype = torch.float32

        # Try to load model with an automatic device mapping which helps place
        # large models across GPU/CPU and supports offloading when used with
        # accelerate. Fall back to loading onto the single device if that fails.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                torch_dtype=dtype,
                device_map='auto'
            )
        except Exception as e:
            print(f"Warning: failed to load with device_map='auto' ({e}), falling back to simpler load")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=dtype)
                if device in {"cuda", "mps"} and torch.device(device).type != 'cpu':
                    try:
                        self.model.to(device)
                    except Exception:
                        pass
            except Exception as e2:
                print(f"Failed to load LLM model: {e2}")
                raise

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