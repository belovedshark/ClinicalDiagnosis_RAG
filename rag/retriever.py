# rag/retriever.py
import os
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
import numpy as np

from .utils import is_table_query, parse_query_table
from .config import TABLE_QUERY_BOOST


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
        # Build a query embedding. If the user provided a table block, parse it and
        # embed a compact flat_text representation (headers + first row) so that
        # retrieval is biased towards table chunks.
        table_query = False
        parsed_table = None
        if query_text and is_table_query(query_text):
            table_query = True
            parsed_table = parse_query_table(query_text)
            emb_text = parsed_table.get('flat_text') or query_text
            query_emb = self.embed_query(query_text=emb_text)
        else:
            query_emb = self.embed_query(query_text, query_image)

        # Use `search` if available (returns scored points), otherwise fall back
        # to `query_points`. Request a larger pool so we can boost table candidates
        # and then take the top_k after re-scoring.
        pool_k = max(top_k * 10, top_k + 10)
        if hasattr(self.client, 'search'):
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_emb.tolist(),
                limit=pool_k,
                with_payload=True,
            )
        else:
            # Older client variants
            results = self.client.query_points(
                collection_name=self.collection,
                query_vector=query_emb.tolist(),
                limit=pool_k,
                with_payload=True,
            )

        # Normalize results to an iterable of points
        if hasattr(results, 'points'):
            points = results.points
        else:
            points = results

        if not points:
            print("⚠️ No results found.")
            return []

        scored = []
        for p in points:
            # Support both dict-like and object-like points
            if isinstance(p, dict):
                payload = p.get('payload', {}) or {}
                score = p.get('score')
            else:
                payload = getattr(p, 'payload', {}) or {}
                score = getattr(p, 'score', None)

            if score is None:
                # If no score is present, skip re-scoring; assign a tiny value
                score = 0.0

            boosted = float(score)

            # If this is a table-query, preferentially boost points whose payload
            # indicate they are tables and whose headers/text contain any of the
            # parsed headers from the query.
            if table_query and payload.get('type') == 'table':
                headers = parsed_table.get('headers', []) if parsed_table else []
                hay = " ".join([
                    (payload.get('headers') or ''),
                    (payload.get('text') or ''),
                ]).lower()
                # lexical header match
                if any(h.strip().lower() and h.strip().lower() in hay for h in headers):
                    boosted *= TABLE_QUERY_BOOST

            scored.append((boosted, payload))

        # Sort by boosted score desc and take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        # For table queries, put table chunks first in the returned contexts
        contexts = []
        table_texts = []
        other_texts = []
        for sc, payload in top:
            txt = payload.get('text', '')
            if table_query and payload.get('type') == 'table':
                table_texts.append(txt)
            else:
                other_texts.append(txt)

        contexts.extend(table_texts)
        contexts.extend(other_texts)

        return contexts