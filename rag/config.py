# rag/config.py

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"

# Use the same embedding model as in your ingestion script
EMBED_MODEL = "openai/clip-vit-base-patch32"

# Gemma model for generation
LLM_MODEL = "google/gemma-2b-it"   # or "google/gemma-7b-it" if you have GPU VRAM
# LLM_MODEL = "google/gemma-1.1-2b-it"

DEVICE = "cuda"  # "cuda" | "cpu" | "mps"
# How much to boost table-matching points when the user query contains a table
TABLE_QUERY_BOOST = 1.5