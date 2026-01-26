# rag/retriever.py
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModel
from PIL import Image
import torch
import numpy as np

from .utils import is_table_query, parse_query_table
from .config import TABLE_QUERY_BOOST


class BM25:
    """Simple BM25 implementation for keyword-based retrieval."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = {}  # term -> number of docs containing term
        self.idf = {}  # term -> IDF score
        self.doc_term_freqs = []  # list of dicts, term -> freq in each doc
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens
    
    def fit(self, documents: List[str]):
        """Build BM25 index from documents."""
        self.corpus = documents
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.doc_freqs = {}
        
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            
            # Count term frequencies in this document
            term_freq = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            self.doc_term_freqs.append(term_freq)
            
            # Update document frequencies
            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Calculate IDF for each term
        n_docs = len(documents)
        for term, df in self.doc_freqs.items():
            # IDF with smoothing
            self.idf[term] = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score for a query against a specific document."""
        query_tokens = self._tokenize(query)
        doc_term_freq = self.doc_term_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        score = 0.0
        for token in query_tokens:
            if token not in doc_term_freq:
                continue
            
            tf = doc_term_freq[token]
            idf = self.idf.get(token, 0)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            score += idf * numerator / denominator
        
        return score
    
    def get_scores(self, query: str) -> np.ndarray:
        """Get BM25 scores for all documents."""
        scores = np.array([self.score(query, i) for i in range(len(self.corpus))])
        return scores


class CrossEncoderReranker:
    """Cross-encoder model for reranking retrieved documents."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from transformers import AutoModelForSequenceClassification
                print(f"🔄 Loading cross-encoder: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self._model.to(self.device)
                self._model.eval()
                print(f"✅ Cross-encoder loaded on {self.device}")
            except Exception as e:
                print(f"Warning: Failed to load cross-encoder: {e}")
                return False
        return True
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: The query string
            documents: List of document texts
            top_k: Number of top documents to return
            
        Returns:
            List of (original_index, score, text) tuples, sorted by score descending
        """
        if not self._load_model() or not documents:
            # Fallback: return documents in original order
            return [(i, 0.0, doc) for i, doc in enumerate(documents[:top_k])]
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc] for doc in documents]
        
        # Tokenize
        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get scores
        with torch.no_grad():
            outputs = self._model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().numpy()
        
        # Handle single document case
        if scores.ndim == 0:
            scores = np.array([scores.item()])
        
        # Create scored results
        scored_docs = [(i, float(scores[i]), documents[i]) for i in range(len(documents))]
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]


class Retriever:
    """Enhanced retriever with hybrid search (semantic + BM25) and cross-encoder reranking."""
    
    def __init__(self,
                 collection="documents",
                 text_model_name="thenlper/gte-small",
                 image_model_name="openai/clip-vit-base-patch32",
                 device="cpu",
                 use_hybrid_search: bool = True,
                 use_reranking: bool = True,
                 semantic_weight: float = 0.7,
                 bm25_weight: float = 0.3,
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Enhanced Retriever with hybrid search and reranking.

        Args:
            collection: Qdrant collection name
            text_model_name: Text embedding model name
            image_model_name: CLIP model name for image queries
            device: Device for models
            use_hybrid_search: Whether to use BM25 + semantic hybrid search
            use_reranking: Whether to use cross-encoder reranking
            semantic_weight: Weight for semantic similarity scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
            reranker_model: Cross-encoder model name for reranking
        """
        self.collection = collection
        self.device = device
        self.use_hybrid_search = use_hybrid_search
        self.use_reranking = use_reranking
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

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
        
        # Initialize BM25 (will be populated on first use)
        self.bm25 = None
        self._bm25_doc_map = {}  # Maps BM25 index to document text
        
        # Initialize cross-encoder reranker
        if use_reranking:
            self.reranker = CrossEncoderReranker(model_name=reranker_model, device=device)
        else:
            self.reranker = None

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

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to 0-1 range."""
        if len(scores) == 0:
            return scores
        min_score = scores.min()
        max_score = scores.max()
        if max_score - min_score == 0:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def retrieve(self, query_text=None, query_image=None, top_k=5):
        """
        Retrieve similar chunks from Qdrant with optional hybrid search and reranking.
        
        Args:
            query_text: Text query
            query_image: Image query path
            top_k: Number of results to return
            
        Returns:
            List of context strings
        """
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

        # Retrieve more candidates for hybrid search and reranking
        pool_k = max(top_k * 10, 50) if (self.use_hybrid_search or self.use_reranking) else max(top_k * 10, top_k + 10)
        
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

        # Extract documents and scores
        documents = []
        semantic_scores = []
        payloads = []
        
        for p in points:
            if isinstance(p, dict):
                payload = p.get('payload', {}) or {}
                score = p.get('score', 0.0)
            else:
                payload = getattr(p, 'payload', {}) or {}
                score = getattr(p, 'score', 0.0)
            
            text = payload.get('text', '')
            if text:
                documents.append(text)
                semantic_scores.append(float(score))
                payloads.append(payload)

        if not documents:
            return []

        semantic_scores = np.array(semantic_scores)
        
        # Apply hybrid search if enabled
        if self.use_hybrid_search and query_text and not is_image_query:
            # Build BM25 index for retrieved documents
            bm25 = BM25()
            bm25.fit(documents)
            bm25_scores = bm25.get_scores(query_text)
            
            # Normalize both scores
            norm_semantic = self._normalize_scores(semantic_scores)
            norm_bm25 = self._normalize_scores(bm25_scores)
            
            # Combine scores
            combined_scores = (self.semantic_weight * norm_semantic + 
                             self.bm25_weight * norm_bm25)
        else:
            combined_scores = semantic_scores

        # Apply table query boosting
        if table_query:
            for i, payload in enumerate(payloads):
                if payload.get('type') == 'table':
                    headers = parsed_table.get('headers', []) if parsed_table else []
                    hay = " ".join([
                        (payload.get('headers') or ''),
                        (payload.get('text') or ''),
                    ]).lower()
                    if any(h.strip().lower() and h.strip().lower() in hay for h in headers):
                        combined_scores[i] *= TABLE_QUERY_BOOST

        # Sort by combined scores
        sorted_indices = np.argsort(combined_scores)[::-1]
        
        # Get top candidates for reranking
        rerank_k = min(top_k * 3, len(documents)) if self.use_reranking else top_k
        top_indices = sorted_indices[:rerank_k]
        top_docs = [documents[i] for i in top_indices]
        top_payloads = [payloads[i] for i in top_indices]

        # Apply cross-encoder reranking if enabled
        if self.use_reranking and self.reranker and query_text:
            reranked = self.reranker.rerank(query_text, top_docs, top_k=top_k)
            
            # Build final context list
            contexts = []
            table_texts = []
            other_texts = []
            
            for orig_idx, score, text in reranked:
                payload = top_payloads[orig_idx]
                if table_query and payload.get('type') == 'table':
                    table_texts.append(text)
                else:
                    other_texts.append(text)
            
            contexts.extend(table_texts)
            contexts.extend(other_texts)
            return contexts[:top_k]
        else:
            # No reranking, just take top-k
            top = [(combined_scores[i], payloads[i], documents[i]) for i in top_indices[:top_k]]
            
            contexts = []
            table_texts = []
            other_texts = []
            
            for sc, payload, txt in top:
                if table_query and payload.get('type') == 'table':
                    table_texts.append(txt)
                else:
                    other_texts.append(txt)
            
            contexts.extend(table_texts)
            contexts.extend(other_texts)
            return contexts

    def retrieve_with_scores(self, query_text=None, query_image=None, top_k=5) -> List[Dict[str, Any]]:
        """
        Retrieve similar chunks with detailed scoring information.
        
        Returns:
            List of dicts with 'text', 'score', 'source', and other metadata
        """
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

        vector_name = 'image' if is_image_query else 'text'
        pool_k = max(top_k * 10, 50)
        
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_emb.tolist(),
            using=vector_name,
            limit=pool_k,
            with_payload=True,
        )
        
        points = results.points if hasattr(results, 'points') else results

        if not points:
            return []

        documents = []
        semantic_scores = []
        payloads = []
        
        for p in points:
            if isinstance(p, dict):
                payload = p.get('payload', {}) or {}
                score = p.get('score', 0.0)
            else:
                payload = getattr(p, 'payload', {}) or {}
                score = getattr(p, 'score', 0.0)
            
            text = payload.get('text', '')
            if text:
                documents.append(text)
                semantic_scores.append(float(score))
                payloads.append(payload)

        if not documents:
            return []

        semantic_scores = np.array(semantic_scores)
        
        # Hybrid search
        if self.use_hybrid_search and query_text and not is_image_query:
            bm25 = BM25()
            bm25.fit(documents)
            bm25_scores = bm25.get_scores(query_text)
            
            norm_semantic = self._normalize_scores(semantic_scores)
            norm_bm25 = self._normalize_scores(bm25_scores)
            
            combined_scores = (self.semantic_weight * norm_semantic + 
                             self.bm25_weight * norm_bm25)
        else:
            combined_scores = semantic_scores
            bm25_scores = np.zeros_like(semantic_scores)

        sorted_indices = np.argsort(combined_scores)[::-1]
        rerank_k = min(top_k * 3, len(documents)) if self.use_reranking else top_k
        top_indices = sorted_indices[:rerank_k]
        top_docs = [documents[i] for i in top_indices]
        top_payloads = [payloads[i] for i in top_indices]

        # Reranking
        if self.use_reranking and self.reranker and query_text:
            reranked = self.reranker.rerank(query_text, top_docs, top_k=top_k)
            
            results = []
            for orig_idx, rerank_score, text in reranked:
                global_idx = top_indices[orig_idx]
                payload = top_payloads[orig_idx]
                results.append({
                    'text': text,
                    'semantic_score': float(semantic_scores[global_idx]),
                    'bm25_score': float(bm25_scores[global_idx]) if self.use_hybrid_search else 0.0,
                    'combined_score': float(combined_scores[global_idx]),
                    'rerank_score': float(rerank_score),
                    'source': payload.get('source', 'unknown'),
                    'type': payload.get('type', 'text')
                })
            return results
        else:
            results = []
            for i in top_indices[:top_k]:
                payload = payloads[i]
                results.append({
                    'text': documents[i],
                    'semantic_score': float(semantic_scores[i]),
                    'bm25_score': float(bm25_scores[i]) if self.use_hybrid_search else 0.0,
                    'combined_score': float(combined_scores[i]),
                    'rerank_score': 0.0,
                    'source': payload.get('source', 'unknown'),
                    'type': payload.get('type', 'text')
                })
            return results
