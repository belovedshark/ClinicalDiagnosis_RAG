import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# WINDOWS + TORCH STABILITY
torch._dynamo.disable()

# PATHS
BASE_MODEL = "google/gemma-2b-it"
DATA_PATH = "./openmed_train.jsonl"   # expects {"text": "..."}
OUTPUT_DIR = "./gemma2b_qlora"

# CHECK CUDA
print("CUDA available:", torch.cuda.is_available())

# QLORA CONFIG (4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# LOAD QUANTIZED BASE MODEL
print("Loading Gemma-2B in 4-bit...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# REQUIRED FOR QLORA
model = prepare_model_for_kbit_training(model)

# LORA CONFIG (Gemma optimized)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ATTACH LORA (TRAINABLE)
model = get_peft_model(model, lora_config)
model.config.use_cache = False
model.train()

print("Trainable parameters:")
model.print_trainable_parameters()

# LOAD DATASET
print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train",
)

# TOKENIZATION (FAST)
def tokenize_fn(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding=False,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(
    tokenize_fn,
    remove_columns=dataset.column_names,
)

# DATA COLLATOR
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# TRAINING ARGS (WINDOWS SAFE)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

# TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)


# TRAIN
print("🚀 Starting QLoRA training...")
trainer.train()

# SAVE LORA ADAPTER
print("Saving LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ QLoRA training complete")
