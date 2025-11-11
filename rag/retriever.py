# rag/retriever.py
import os
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
import numpy as np


class Retriever:
    def __init__(self,
                 collection="documents",
                 model_name="openai/clip-vit-base-patch32",
                 device="cpu"):
        """Retriever that queries Qdrant Cloud and uses CLIP for query encoding.

        Qdrant URL and API key are read from environment variables if present:
        - QDRANT_URL
        - QDRANT_API_KEY
        """
        self.collection = collection
        self.device = device

        # Read connection info from environment with sensible defaults
        from .config import QDRANT_URL as DEFAULT_QDRANT_URL
        qdrant_url = os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL)
        qdrant_api_key = os.environ.get("QDRANT_API_KEY")

        if not qdrant_api_key:
            print("Warning: QDRANT_API_KEY not set in environment. Qdrant may reject the connection if the collection is protected.")

        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        print(f"🔗 Connected to Qdrant at {qdrant_url}")

        # Load CLIP model
        print(f"🧠 Loading CLIP model: {model_name} (device={device})")
        # Normalize device string for torch
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(device)
        except Exception:
            # Fall back to CPU if an invalid device string was passed
            print(f"Warning: failed to load CLIP on device '{device}', falling back to cpu")
            self.model = CLIPModel.from_pretrained(model_name).to("cpu")
            self.device = "cpu"

        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_query(self, query_text=None, query_image=None):
        """Generate CLIP embeddings from text or image query."""
        if query_text:
            inputs = self.processor(text=query_text, return_tensors='pt', truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self.model.get_text_features(**inputs).cpu().numpy()[0]
            return emb
        elif query_image:
            img = Image.open(query_image).convert('RGB')
            inputs = self.processor(images=img, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs).cpu().numpy()[0]
            return emb
        else:
            raise ValueError("Provide either query_text or query_image.")

    def retrieve(self, query_text=None, query_image=None, top_k=5):
        """Retrieve similar chunks from Qdrant."""
        query_emb = self.embed_query(query_text, query_image)
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_emb.tolist(),
            limit=top_k,
        )

        if not results or not results.points:
            print("⚠️ No results found.")
            return []

        contexts = []
        for point in results.points:
            payload = point.payload or {}
            text = payload.get("text", "")
            contexts.append(text)

        return contexts