import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "google/gemma-2b-it"
# Use absolute path relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LORA_PATH = os.path.join(PROJECT_ROOT, "lora_instruction_response")

# --------------------
# Tokenizer
# --------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# --------------------
# Base model (MPS for Apple Silicon)
# --------------------
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="mps"
)

# --------------------
# Load LoRA
# --------------------
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# --------------------
# Ask a question
# --------------------
instruction = "Explain how malaria is transmitted."

prompt = (
    "### Instruction:\n"
    f"{instruction}\n\n"
    "### Response:\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

print("\n=== GENERATED ANSWER ===\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
