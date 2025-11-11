# rag/pipeline.py

# from .embedder import Embedder
from .retriever import Retriever
from .generator import Generator

class RAGPipeline:
    """Full retrieval-augmented generation pipeline."""

    def __init__(self):
        # self.embedder = Embedder()
        self.retriever = Retriever()
        self.generator = Generator()

    def query(self, question: str, image_path: str = None, k: int = 5) -> str:
        """Ask a question and/or provide an image → retrieve relevant → generate an answer."""
        if image_path is not None:
            vector = self.retriever.embed_query(query_image=image_path)
            contexts = self.retriever.retrieve(query_or_vector=vector, top_k=k)
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