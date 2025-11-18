# rag/retriever.py
import os
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModel
from PIL import Image
import torch
import numpy as np

from .utils import is_table_query, parse_query_table
from .config import TABLE_QUERY_BOOST


class Retriever:
    def __init__(self,
                 collection="documents",
                 text_model_name="thenlper/gte-small",
                 image_model_name="openai/clip-vit-base-patch32",
                 device="cpu"):
        """Retriever that queries Qdrant and uses appropriate models for encoding.

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

        # Load text embedding model (for text queries)
        print(f"🧠 Loading text model: {text_model_name} (device={device})")
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            self.text_model = AutoModel.from_pretrained(text_model_name).to(device)
        except Exception as e:
            print(f"Warning: failed to load text model on device '{device}', falling back to cpu: {e}")
            self.text_model = AutoModel.from_pretrained(text_model_name).to("cpu")
            self.device = "cpu"

        # Load CLIP model (for image queries)
        print(f"🧠 Loading CLIP model: {image_model_name} (device={device})")
        try:
            self.clip_model = CLIPModel.from_pretrained(image_model_name).to(device)
        except Exception:
            print(f"Warning: failed to load CLIP on device '{device}', falling back to cpu")
            self.clip_model = CLIPModel.from_pretrained(image_model_name).to("cpu")

        self.clip_processor = CLIPProcessor.from_pretrained(image_model_name)

    def embed_query(self, query_text=None, query_image=None):
        """Generate embeddings from text or image query using appropriate models."""
        if query_text:
            # Use GTE text model for text queries (384-dim)
            inputs = self.text_tokenizer(query_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                # Use mean pooling for GTE model
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    emb = outputs.pooler_output.cpu().numpy()[0]
                else:
                    # Mean pooling
                    token_embeddings = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    emb = (sum_embeddings / sum_mask).cpu().numpy()[0]
            return emb
        elif query_image:
            # Use CLIP for image queries (512-dim)
            img = Image.open(query_image).convert('RGB')
            inputs = self.clip_processor(images=img, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self.clip_model.get_image_features(**inputs).cpu().numpy()[0]
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
        is_image_query = query_image is not None
        
        if query_text and is_table_query(query_text):
            table_query = True
            parsed_table = parse_query_table(query_text)
            emb_text = parsed_table.get('flat_text') or query_text
            query_emb = self.embed_query(query_text=emb_text)
        else:
            query_emb = self.embed_query(query_text, query_image)

        # Determine which named vector to search (text or image)
        vector_name = 'image' if is_image_query else 'text'

        # Use query_points with named vectors. Request a larger pool so we can boost 
        # table candidates and then take the top_k after re-scoring.
        pool_k = max(top_k * 10, top_k + 10)
        
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_emb.tolist(),
            using=vector_name,
            limit=pool_k,
            with_payload=True,
        )
        
        # Extract points from results
        points = results.points if hasattr(results, 'points') else results

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