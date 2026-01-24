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

    def generate(self, context: str, question: str, custom_prompt: str = None) -> str:
        """Generate an answer grounded in retrieved context.
        
        Args:
            context: Retrieved context text
            question: The question/prompt to answer
            custom_prompt: Optional custom prompt template. If provided, should contain
                          {context} and {question} placeholders.
        """
        if custom_prompt:
            prompt = custom_prompt.format(context=context, question=question)
        else:
            prompt = f"""
You are a clinical assistant. Answer the question based only on the context below.

Context:
{context}

Question: {question}
Answer:
"""
        output = self.pipe(
            prompt,
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.9,
            return_full_text=False,  # Only return the generated text, not the prompt
        )[0]["generated_text"]

        # Clean up the generated output
        answer = output.strip()
        
        # If the output still contains "Answer:", extract after it
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        # Remove markdown formatting
        answer = answer.replace('**', '').replace('*', '')
        
        # Split into lines and filter out separators/empty lines
        lines = []
        for line in answer.split('\n'):
            line = line.strip()
            # Skip empty lines, separator lines (dashes, underscores, equals)
            if not line:
                continue
            if set(line) <= {'-', '_', '=', '*', '#'}:
                continue
            # Skip lines that are too short (likely formatting artifacts)
            if len(line) < 3:
                continue
            lines.append(line)
        
        if lines:
            answer = lines[0]
        else:
            answer = "Unable to determine diagnosis"
        
        # Remove any trailing punctuation or extra text after the diagnosis
        # Common patterns: "Diagnosis.", "Diagnosis - explanation", etc.
        if ' - ' in answer:
            answer = answer.split(' - ')[0].strip()
        if answer.endswith('.'):
            answer = answer[:-1]
        if answer.endswith(':'):
            answer = answer[:-1]
            
        return answer