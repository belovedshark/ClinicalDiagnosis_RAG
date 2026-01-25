import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

BASE_MODEL = "google/gemma-2b-it"
DATA_PATH = "./dataset_instruction_response.jsonl"
OUTPUT_DIR = "./lora_instruction_response"

# --------------------
# Tokenizer
# --------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# --------------------
# Quantized model
# --------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

# --------------------
# LoRA
# --------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --------------------
# Dataset
# --------------------
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def tokenize(example):
    prompt = (
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Response:\n"
    )

    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        add_special_tokens=False
    )["input_ids"]

    response_ids = tokenizer(
        example["response"] + tokenizer.eos_token,
        truncation=True,
        max_length=512,
        add_special_tokens=False
    )["input_ids"]

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids)
    }

tokenized = dataset.map(
    tokenize,
    remove_columns=dataset.column_names
)

# --------------------
# Training
# --------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=True,
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training finished")
