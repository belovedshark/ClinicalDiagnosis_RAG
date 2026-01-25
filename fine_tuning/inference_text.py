from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# CONFIG
BASE_MODEL = "google/gemma-2b-it"   # FP16 base model
LORA_ADAPTER = "./gemma2b_qlora"   # path to your trained LoRA
MAX_LENGTH = 256                    # max tokens to generate

# LOAD BASE MODEL (FP16)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
)

# ATTACH LORA ADAPTER
model = PeftModel.from_pretrained(model, LORA_ADAPTER)
model.to("cuda")
model.eval()  # inference mode

# LOAD TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# PROMPT / INPUT
prompt = "Explain how photosynthesis works in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# GENERATE
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=MAX_LENGTH,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

# DECODE OUTPUT
text = tokenizer.decode(output[0], skip_special_tokens=True)
print("=== GENERATED ===")
print(text)
