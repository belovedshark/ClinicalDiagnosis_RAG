"""Configuration for Gemma 3 4B QLoRA fine-tuning."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "pubmedqa_tropical_filtered.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "gemma-3-4b-tropical-fine-tuned"

# Model
BASE_MODEL = "google/gemma-3-4b-it"

# QLoRA configuration
QLORA_CONFIG = {
    "r": 16,                    # LoRA rank
    "lora_alpha": 32,           # LoRA alpha (typically 2x rank)
    "lora_dropout": 0.05,       # Dropout for regularization
    "target_modules": [         # Which layers to apply LoRA
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# BitsAndBytes quantization config
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16",  # float16 is safer for Blackwell GPUs
    "bnb_4bit_use_double_quant": True,
}

# Training hyperparameters
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,      # Reduced for 8GB VRAM
    "per_device_eval_batch_size": 1,       # Reduced for 8GB VRAM
    "gradient_accumulation_steps": 16,     # Increased to maintain effective batch size
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "max_seq_length": 768,                 # Balanced for 8GB VRAM
    "logging_steps": 50,
    "save_steps": 500,
    "eval_steps": 500,
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
}

# Data split
TRAIN_TEST_SPLIT = 0.1  # 10% for validation
RANDOM_SEED = 42

# Data sampling (reduce dataset size for faster training)
DATA_SAMPLE_PERCENT = 30  # Use 25% of data for faster training (set to 100 for full dataset)
