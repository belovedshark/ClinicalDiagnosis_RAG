import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# =====================
# CONFIG
# =====================
BASE_MODEL = "google/gemma-3-4b-it"
LORA_PATH = "./lora_gemma3_4b_medical"

MAX_NEW_TOKENS = 128

# =====================
# TOKENIZER
# =====================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =====================
# MODEL (4-bit)
# =====================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# =====================
# INFERENCE FUNCTION
# =====================
def generate_response(instruction: str) -> str:
    prompt = (
        "### Instruction:\n"
        f"{instruction.strip()}\n\n"
        "### Response:\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,          # 🔒 IMPORTANT
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    if "### Response:" in decoded:
        return decoded.split("### Response:")[-1].strip()
    else:
        return decoded.strip()

# =====================
# INTERACTIVE LOOP
# =====================
if __name__ == "__main__":
    print("🧠 Medical LoRA Inference (Ctrl+C to exit)\n")

    while True:
        try:
            q = input("🩺 Medical Question:\n> ").strip()
            if not q:
                continue
            print("\n🤖 Answer:\n")
            print(generate_response(q))
            print("\n" + "-" * 60 + "\n")
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break
