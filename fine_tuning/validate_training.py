"""Validation script that catches training bugs by simulating real training.

This runs multiple gradient accumulation steps to catch issues like:
- cuBLAS BFloat16 errors on Blackwell GPUs
- OOM during gradient accumulation
- Checkpoint recomputation errors

Usage: python -m fine_tuning.validate_training
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Config - CHANGE THESE TO TEST DIFFERENT SETTINGS
TEST_CONFIG = {
    "model": "google/gemma-3-4b-it",  # Larger model where error occurred
    "compute_dtype": "bfloat16",  # TEST 1: This should cause cuBLAS error on Blackwell
    "use_fp16_training": False,
    "use_bf16_training": True,
    "batch_size": 1,
    "seq_length": 512,
    "gradient_accumulation_steps": 4,  # Simulate accumulation
    "num_iterations": 5,  # Run multiple iterations to catch intermittent errors
}


def validate():
    print("\n" + "=" * 60)
    print("TRAINING VALIDATION (catches bfloat16/cuBLAS bugs)")
    print("=" * 60)
    
    # 1. Check CUDA
    print("\n[1] CUDA Check")
    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available!")
        return False
    print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
    
    # 2. Load model with quantization
    print(f"\n[2] Loading model: {TEST_CONFIG['model']}")
    print(f"    Compute dtype: {TEST_CONFIG['compute_dtype']}")
    
    compute_dtype = getattr(torch, TEST_CONFIG['compute_dtype'])
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype
    )
    
    tokenizer = AutoTokenizer.from_pretrained(TEST_CONFIG['model'])
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        TEST_CONFIG['model'],
        quantization_config=bnb_config,
        device_map="auto"
    )
    print(f"[OK] Model loaded, memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # 3. Prepare model for k-bit training (enables gradients)
    print("\n[3] Preparing model for training")
    model = prepare_model_for_kbit_training(model)
    print("[OK] Model prepared for k-bit training")
    
    # 4. Apply LoRA
    print("\n[4] Applying LoRA")
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
    print("[OK] LoRA applied")
    
    # 5. Simulate training with gradient accumulation
    print(f"\n[5] Simulating training")
    print(f"    Batch size: {TEST_CONFIG['batch_size']}")
    print(f"    Seq length: {TEST_CONFIG['seq_length']}")
    print(f"    Gradient accumulation: {TEST_CONFIG['gradient_accumulation_steps']}")
    print(f"    Iterations: {TEST_CONFIG['num_iterations']}")
    print(f"    FP16: {TEST_CONFIG['use_fp16_training']}, BF16: {TEST_CONFIG['use_bf16_training']}")
    
    # Set up mixed precision
    scaler = None
    if TEST_CONFIG['use_fp16_training']:
        scaler = torch.amp.GradScaler('cuda')
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    try:
        for iteration in range(TEST_CONFIG['num_iterations']):
            optimizer.zero_grad()
            accumulated_loss = 0
            
            for accum_step in range(TEST_CONFIG['gradient_accumulation_steps']):
                # Create dummy batch
                input_ids = torch.randint(
                    100, 10000, 
                    (TEST_CONFIG['batch_size'], TEST_CONFIG['seq_length'])
                ).to(model.device)
                attention_mask = torch.ones_like(input_ids)
                labels = input_ids.clone()
                
                # Forward pass with autocast
                if TEST_CONFIG['use_fp16_training']:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss / TEST_CONFIG['gradient_accumulation_steps']
                elif TEST_CONFIG['use_bf16_training']:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss / TEST_CONFIG['gradient_accumulation_steps']
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / TEST_CONFIG['gradient_accumulation_steps']
                
                # Backward pass (this is where cuBLAS error occurs!)
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulated_loss += loss.item()
            
            # Optimizer step
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            print(f"    Iteration {iteration + 1}/{TEST_CONFIG['num_iterations']}: loss = {accumulated_loss:.4f}")
        
        print(f"\n[OK] Training simulation successful!")
        print(f"    Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = validate()
    
    print("\n" + "=" * 60)
    if success:
        print("ALL VALIDATIONS PASSED!")
        print("Your training config should work.")
    else:
        print("VALIDATION FAILED!")
        print("Fix the issue before running full training.")
    print("=" * 60 + "\n")
