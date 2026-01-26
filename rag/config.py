# rag/config.py

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"

# Use the same embedding model as in your ingestion script
EMBED_MODEL = "openai/clip-vit-base-patch32"

# Gemma model for generation
LLM_MODEL = "google/gemma-3-4b-it"   # or "google/gemma-7b-it" if you have GPU VRAM
# LLM_MODEL = "google/gemma-1.1-2b-it"

DEVICE = "auto"  # "cuda" | "cpu" | "mps"

# How much to boost table-matching points when the user query contains a table
TABLE_QUERY_BOOST = 1.5

# =============================================================================
# Hybrid Search and Reranking Configuration
# =============================================================================

# Enable hybrid search (combines semantic similarity with BM25 keyword matching)
USE_HYBRID_SEARCH = True

# Enable cross-encoder reranking for better relevance
USE_RERANKING = True

# Weights for hybrid search (should sum to 1.0)
SEMANTIC_WEIGHT = 0.7  # Weight for semantic/embedding similarity
BM25_WEIGHT = 0.3      # Weight for BM25 keyword matching

# Cross-encoder model for reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
