# rag/pipeline.py

# from .embedder import Embedder
from .retriever import Retriever
from .generator import Generator
from .config import DEVICE as CONFIG_DEVICE
import torch


class RAGPipeline:
    """Full retrieval-augmented generation pipeline.

    This pipeline will detect the best device to use (GPU/CPU/mps) based on
    the configuration in `rag.config` and on runtime availability, then pass
    that device to the Retriever and Generator so models are loaded onto GPU
    when available.
    
    Supports both base model and fine-tuned LoRA model via the use_lora parameter.
    """

    def __init__(self, use_lora: bool = False, lora_path: str = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            use_lora: Whether to load a fine-tuned LoRA adapter for the generator.
                      Defaults to False (uses base model).
            lora_path: Path to LoRA adapter. If None, uses LORA_ADAPTER_PATH from config.
        """
        # Decide runtime device
        device_pref = (CONFIG_DEVICE or "auto").lower()
        if device_pref == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                # MPS for newer Mac hardware
                try:
                    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"
                except Exception:
                    device = "cpu"
        else:
            device = device_pref

        # self.embedder = Embedder()
        self.retriever = Retriever(device=device)
        self.generator = Generator(device=device, use_lora=use_lora, lora_path=lora_path)
        self.use_lora = use_lora

    def query(self, question: str, image_path: str = None, k: int = 5) -> str:
        """Ask a question and/or provide an image → retrieve relevant → generate an answer."""
        if image_path is not None:
            # Retrieve by image path directly (retriever will embed the image)
            contexts = self.retriever.retrieve(query_image=image_path, top_k=k)
        elif question is not None:
            contexts = self.retriever.retrieve(query_text=question, top_k=k)
        else:
            raise ValueError("Provide question or image_path (or both).")
        if not contexts:
            return "No relevant documents found in the database."
        
        print(f"🧠 Querying RAG with: {question}")
        # contexts = self.retriever.retrieve(query_text=question, top_k=k)
        context_text = "\n\n".join(contexts)
        pre_prompt = """
        You are a clinical reasoning assistant specializing in tropical and infectious diseases.

        The following case describes a patient in a tropical, resource-limited region. 
        Based on the patient's presentation, provide:
        1. The **most likely diagnosis** with reasoning.
        2. The **key differential diagnoses** with brief distinguishing features.
        3. A **recommended management approach** appropriate for a low-resource setting (diagnostic steps + initial treatment priorities).

        Patient case:
        """
        q = pre_prompt + (question or "Describe the image.")

        return self.generator.generate(context_text, q)