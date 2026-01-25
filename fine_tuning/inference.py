"""Inference module for fine-tuned Gemma model."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from .config import BASE_MODEL, OUTPUT_DIR


def get_device():
    """Detect available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_fine_tuned_model(adapter_path: str = None):
    """Load base model with fine-tuned LoRA adapter.

    Args:
        adapter_path: Path to LoRA adapter. Defaults to OUTPUT_DIR.

    Returns:
        model, tokenizer tuple
    """
    adapter_path = adapter_path or str(OUTPUT_DIR)
    device = get_device()

    print(f"Loading base model: {BASE_MODEL}")
    print(f"Loading adapter from: {adapter_path}")
    print(f"Using device: {device}")

    # Choose dtype based on device
    if device == "cuda":
        dtype = torch.float16
    elif device == "mps":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    print("Model loaded successfully!")
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    context: str,
    question: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> str:
    """Generate an answer using the fine-tuned model.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        context: Research context/abstract
        question: The clinical question
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated answer string
    """
    # Format as Gemma chat
    prompt = f"""<bos><start_of_turn>user
Context: {context}

Question: {question}<end_of_turn>
<start_of_turn>model
"""

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)

    # Clean up
    answer = answer.strip()
    if "<end_of_turn>" in answer:
        answer = answer.split("<end_of_turn>")[0].strip()

    return answer


def main():
    """CLI for testing inference."""
    parser = argparse.ArgumentParser(description="Test fine-tuned Gemma model")
    parser.add_argument("--question", "-q", required=True, help="Clinical question")
    parser.add_argument("--context", "-c", default="", help="Optional context")
    parser.add_argument("--adapter", "-a", default=None, help="Path to LoRA adapter")
    args = parser.parse_args()

    model, tokenizer = load_fine_tuned_model(args.adapter)

    answer = generate_answer(
        model,
        tokenizer,
        context=args.context or "No specific context provided.",
        question=args.question,
    )

    print("\n" + "="*50)
    print("Question:", args.question)
    print("="*50)
    print("Answer:", answer)
    print("="*50)


if __name__ == "__main__":
    main()
