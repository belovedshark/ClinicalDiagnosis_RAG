"""Main training script for Gemma 3 4B QLoRA fine-tuning."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from .config import (
    BASE_MODEL,
    OUTPUT_DIR,
    QLORA_CONFIG,
    BNB_CONFIG,
    TRAINING_CONFIG,
)
from .data_prep import load_and_prepare_data


def get_device():
    """Detect available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer():
    """Load quantized model and tokenizer."""
    print(f"Loading model: {BASE_MODEL}")
    device = get_device()
    print(f"Using device: {device}")

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=BNB_CONFIG["load_in_4bit"],
        bnb_4bit_quant_type=BNB_CONFIG["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, BNB_CONFIG["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=BNB_CONFIG["bnb_4bit_use_double_quant"],
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with quantization
    # Note: bitsandbytes 4-bit requires CUDA. For MPS, we use float16 without quantization
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # MPS/CPU: Load in float16/bfloat16 without 4-bit quantization
        print("Note: 4-bit quantization requires CUDA. Using bfloat16 for MPS/CPU.")
        dtype = torch.bfloat16 if device == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            device_map="auto" if device == "mps" else None,
            trust_remote_code=True,
        )
        model.gradient_checkpointing_enable()

    return model, tokenizer


def setup_lora(model):
    """Configure and apply LoRA to model."""
    lora_config = LoraConfig(
        r=QLORA_CONFIG["r"],
        lora_alpha=QLORA_CONFIG["lora_alpha"],
        lora_dropout=QLORA_CONFIG["lora_dropout"],
        target_modules=QLORA_CONFIG["target_modules"],
        bias=QLORA_CONFIG["bias"],
        task_type=QLORA_CONFIG["task_type"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def train():
    """Run the training loop."""
    # Load data
    dataset = load_and_prepare_data()

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Apply LoRA
    model = setup_lora(model)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        eval_strategy="steps",
        eval_steps=TRAINING_CONFIG["eval_steps"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        load_best_model_at_end=TRAINING_CONFIG["load_best_model_at_end"],
        metric_for_best_model=TRAINING_CONFIG["metric_for_best_model"],
        greater_is_better=TRAINING_CONFIG["greater_is_better"],
        fp16=torch.cuda.is_available(),
        bf16=torch.backends.mps.is_available(),
        optim="adamw_torch",
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        max_seq_length=TRAINING_CONFIG["max_seq_length"],
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving model to {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nTraining complete!")


if __name__ == "__main__":
    train()
