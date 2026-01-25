# rag/generator.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .config import LLM_MODEL, LORA_BASE_MODEL, LORA_ADAPTER_PATH
import torch
import os


def _check_bitsandbytes_available():
    """Check if bitsandbytes is available and functional."""
    try:
        import bitsandbytes
        # Also check if CUDA is available (bitsandbytes requires CUDA)
        return torch.cuda.is_available()
    except ImportError:
        return False


class Generator:
    """Gemma-based text generation module.

    Loads the LLM onto GPU when available using `device_map='auto'` and
    mixed precision where appropriate. Falls back to CPU if GPU loading
    fails.
    
    Supports loading fine-tuned LoRA adapters when use_lora=True.
    """

    def __init__(self, device: str = "cpu", use_lora: bool = False, lora_path: str = None):
        """
        Initialize the generator.
        
        Args:
            device: Device to run model on ("auto", "cuda", "mps", "cpu")
            use_lora: Whether to load a LoRA fine-tuned adapter
            lora_path: Path to LoRA adapter (defaults to LORA_ADAPTER_PATH from config)
        """
        self.use_lora = use_lora
        
        if use_lora:
            self._init_lora_model(device, lora_path)
        else:
            self._init_base_model(device)
    
    def _init_lora_model(self, device: str, lora_path: str = None):
        """Initialize model with LoRA adapter.
        
        Uses 4-bit quantization on CUDA if bitsandbytes is available,
        otherwise falls back to standard precision loading.
        """
        from peft import PeftModel
        
        lora_path = lora_path or LORA_ADAPTER_PATH
        
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA adapter not found at: {lora_path}")
        
        print(f"Loading fine-tuned model with LoRA adapter")
        print(f"  Base model: {LORA_BASE_MODEL}")
        print(f"  LoRA adapter: {lora_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(LORA_BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check if we can use bitsandbytes quantization (requires CUDA)
        use_quantization = _check_bitsandbytes_available()
        
        if use_quantization:
            print("  Using 4-bit quantization (bitsandbytes + CUDA)")
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                LORA_BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            # Fallback for non-CUDA environments (macOS MPS, CPU)
            print("  Using standard precision (no bitsandbytes/CUDA)")
            
            # Determine dtype based on device
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            
            if device == "cuda":
                dtype = torch.float16
            elif device == "mps":
                dtype = torch.float32  # MPS works best with float32 for this model
            else:
                dtype = torch.float32
            
            print(f"  Device: {device}, dtype: {dtype}")
            
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    LORA_BASE_MODEL,
                    torch_dtype=dtype,
                    device_map="auto"
                )
            except Exception as e:
                print(f"  Warning: device_map='auto' failed ({e}), trying direct load")
                base_model = AutoModelForCausalLM.from_pretrained(
                    LORA_BASE_MODEL,
                    torch_dtype=dtype
                )
                if device in {"cuda", "mps"}:
                    base_model = base_model.to(device)
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        self.model.eval()
        
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.model_name = f"{LORA_BASE_MODEL} + LoRA"
        print("✅ Fine-tuned LoRA model loaded successfully")
    
    def _init_base_model(self, device: str):
        """Initialize base model without LoRA."""
        print(f"Loading Gemma model: {LLM_MODEL} (preferred device={device})")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

        # Choose dtype based on device
        if device == "cuda":
            dtype = torch.float16
        elif device == "mps":
            # MPS has limited float16 support; use bfloat16 or float32
            dtype = torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float32
        else:
            dtype = torch.float32

        # Try to load model with an automatic device mapping which helps place
        # large models across GPU/CPU and supports offloading when used with
        # accelerate. Fall back to loading onto the single device if that fails.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                torch_dtype=dtype,
                device_map='auto'
            )
        except Exception as e:
            print(f"Warning: failed to load with device_map='auto' ({e}), falling back to simpler load")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=dtype)
                if device in {"cuda", "mps"} and torch.device(device).type != 'cpu':
                    try:
                        self.model.to(device)
                    except Exception:
                        pass
            except Exception as e2:
                print(f"Failed to load LLM model: {e2}")
                raise

        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.model_name = LLM_MODEL

    def generate(self, context: str, question: str, custom_prompt: str = None) -> str:
        """Generate an answer grounded in retrieved context.
        
        Args:
            context: Retrieved context text
            question: The question/prompt to answer
            custom_prompt: Optional custom prompt template. If provided, should contain
                          {context} and {question} placeholders.
        """
        if custom_prompt:
            prompt = custom_prompt.format(context=context, question=question)
        else:
            prompt = f"""
You are a clinical assistant. Answer the question based only on the context below.

Context:
{context}

Question: {question}
Answer:
"""
        output = self.pipe(
            prompt,
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.9,
            return_full_text=False,  # Only return the generated text, not the prompt
        )[0]["generated_text"]

        # Clean up the generated output
        answer = output.strip()
        
        # If the output still contains "Answer:", extract after it
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        # Remove markdown formatting
        answer = answer.replace('**', '').replace('*', '')
        
        # Split into lines and filter out separators/empty lines
        lines = []
        for line in answer.split('\n'):
            line = line.strip()
            # Skip empty lines, separator lines (dashes, underscores, equals)
            if not line:
                continue
            if set(line) <= {'-', '_', '=', '*', '#'}:
                continue
            # Skip lines that are too short (likely formatting artifacts)
            if len(line) < 3:
                continue
            lines.append(line)
        
        if lines:
            answer = lines[0]
        else:
            answer = "Unable to determine diagnosis"
        
        # Remove any trailing punctuation or extra text after the diagnosis
        # Common patterns: "Diagnosis.", "Diagnosis - explanation", etc.
        if ' - ' in answer:
            answer = answer.split(' - ')[0].strip()
        if answer.endswith('.'):
            answer = answer[:-1]
        if answer.endswith(':'):
            answer = answer[:-1]
            
        return answer