import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# =====================
# CONFIG
# =====================
BASE_MODEL = "google/gemma-3-4b-it"
DATA_PATH = "medical_reasoning_tropical_sft_clean.jsonl"
OUTPUT_DIR = "./lora_gemma3_4b_medical_2"

MAX_LENGTH = 256
EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 2e-4

# =====================
# TOKENIZER
# =====================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =====================
# MODEL (4-bit QLoRA)
# =====================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

# =====================
# LORA CONFIG
# =====================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.config.use_cache = False   # 🔥 IMPORTANT
model.print_trainable_parameters()

# =====================
# LOAD DATASET
# =====================
dataset = load_dataset("json", data_files=DATA_PATH)["train"]

# =====================
# FORMAT FOR SFT
# =====================
def format_example(example):
    prompt = (
        "### Instruction:\n"
        f"{example['instruction'].strip()}\n\n"
        "### Response:\n"
    )

    instr = tokenizer(prompt, add_special_tokens=False)
    resp = tokenizer(example["response"].strip(), add_special_tokens=False)

    input_ids = instr["input_ids"] + resp["input_ids"] + [tokenizer.eos_token_id]
    labels = [-100] * len(instr["input_ids"]) + resp["input_ids"] + [tokenizer.eos_token_id]

    input_ids = input_ids[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }

dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# =====================
# TRAINING
# =====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ TRAINING COMPLETE")
