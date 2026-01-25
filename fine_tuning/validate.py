"""Quick validation script to catch errors before long training runs.

Run this BEFORE training to verify:
1. CUDA/GPU compatibility
2. Model loads correctly
3. Forward pass works
4. Backward pass works (gradient computation)
5. Memory fits

Usage: python -m fine_tuning.validate
"""

import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .config import BASE_MODEL, BNB_CONFIG, QLORA_CONFIG, TRAINING_CONFIG


def validate_cuda():
    """Check CUDA availability and GPU info."""
    print("=" * 50)
    print("1. CUDA Validation")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available!")
        return False
    
    print(f"[OK] CUDA available")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA version: {torch.version.cuda}")
    return True


def validate_model_loading():
    """Test model and tokenizer loading."""
    print("\n" + "=" * 50)
    print("2. Model Loading Validation")
    print("=" * 50)
    
    try:
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=BNB_CONFIG.get("load_in_8bit", False),
            load_in_4bit=BNB_CONFIG.get("load_in_4bit", False),
            bnb_4bit_quant_type=BNB_CONFIG.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, BNB_CONFIG.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=BNB_CONFIG.get("bnb_4bit_use_double_quant", False),
        )
        
        print(f"   Loading {BASE_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        
        print(f"[OK] Model loaded successfully")
        print(f"   Memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        return model, tokenizer
    
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        return None, None


def validate_lora(model):
    """Test LoRA setup."""
    print("\n" + "=" * 50)
    print("3. LoRA Validation")
    print("=" * 50)
    
    try:
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
        print(f"[OK] LoRA applied successfully")
        return model
    
    except Exception as e:
        print(f"[FAIL] LoRA setup failed: {e}")
        return None


def validate_forward_backward(model, tokenizer):
    """Test forward and backward pass with dummy data."""
    print("\n" + "=" * 50)
    print("4. Forward/Backward Pass Validation")
    print("=" * 50)
    
    try:
        # Create dummy input
        test_text = "Hello, this is a test input for validation."
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        
        # Forward pass
        print("   Testing forward pass...")
        model.train()
        outputs = model(**inputs)
        loss = outputs.loss
        print(f"   Forward pass OK, loss: {loss.item():.4f}")
        
        # Backward pass
        print("   Testing backward pass...")
        loss.backward()
        print(f"[OK] Forward and backward pass successful")
        print(f"   Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        
        # Clear gradients and cache
        model.zero_grad()
        torch.cuda.empty_cache()
        return True
    
    except Exception as e:
        print(f"[FAIL] Forward/backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_training_step(model, tokenizer):
    """Test a realistic training step with actual sequence length."""
    print("\n" + "=" * 50)
    print("5. Realistic Training Step Validation")
    print("=" * 50)
    
    try:
        max_len = TRAINING_CONFIG["max_seq_length"]
        batch_size = TRAINING_CONFIG["per_device_train_batch_size"]
        
        print(f"   Testing with max_seq_length={max_len}, batch_size={batch_size}")
        
        # Create realistic-length dummy input
        dummy_ids = torch.randint(100, 1000, (batch_size, max_len)).to(model.device)
        attention_mask = torch.ones_like(dummy_ids)
        
        model.train()
        outputs = model(input_ids=dummy_ids, attention_mask=attention_mask, labels=dummy_ids)
        loss = outputs.loss
        
        print(f"   Forward pass OK, loss: {loss.item():.4f}")
        loss.backward()
        print(f"[OK] Realistic training step successful")
        print(f"   Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"   Available VRAM: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.max_memory_allocated()) / 1e9:.2f} GB")
        
        model.zero_grad()
        torch.cuda.empty_cache()
        return True
    
    except torch.cuda.OutOfMemoryError:
        print(f"[FAIL] Out of memory! Reduce batch_size or max_seq_length in config.py")
        return False
    except Exception as e:
        print(f"[FAIL] Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validations."""
    print("\n[VALIDATION] Catching bugs before the 2-hour mark!\n")
    
    # 1. CUDA
    if not validate_cuda():
        sys.exit(1)
    
    # 2. Model loading
    model, tokenizer = validate_model_loading()
    if model is None:
        sys.exit(1)
    
    # 3. LoRA
    model = validate_lora(model)
    if model is None:
        sys.exit(1)
    
    # 4. Simple forward/backward
    if not validate_forward_backward(model, tokenizer):
        sys.exit(1)
    
    # 5. Realistic training step
    if not validate_training_step(model, tokenizer):
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("ALL VALIDATIONS PASSED!")
    print("=" * 50)
    print("\nYou can now safely run: python -m fine_tuning.train")
    print("Expected training time based on config will be shown at start.\n")


if __name__ == "__main__":
    main()
