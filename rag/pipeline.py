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
    
    Supports enhanced retrieval with:
    - Hybrid search (semantic + BM25)
    - Cross-encoder reranking
    """

    def __init__(self, 
                 use_hybrid_search: bool = True,
                 use_reranking: bool = True,
                 semantic_weight: float = 0.7,
                 bm25_weight: float = 0.3):
        """
        Initialize the RAG pipeline.
        
        Args:
            use_hybrid_search: Enable BM25 + semantic hybrid search
            use_reranking: Enable cross-encoder reranking
            semantic_weight: Weight for semantic similarity (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
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
        self.retriever = Retriever(
            device=device,
            use_hybrid_search=use_hybrid_search,
            use_reranking=use_reranking,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight
        )
        self.generator = Generator(device=device)

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