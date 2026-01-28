# rag/config.py
import os

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"

# Use the same embedding model as in your ingestion script
EMBED_MODEL = "openai/clip-vit-base-patch32"

# Gemma model for generation
LLM_MODEL = "google/gemma-3-4b-it"   # or "google/gemma-7b-it" if you have GPU VRAM
# LLM_MODEL = "google/gemma-1.1-2b-it"

# Fine-tuned LoRA configuration
# The LoRA adapter was trained on gemma-3-4b-it for medical diagnosis
LORA_BASE_MODEL = "google/gemma-3-4b-it"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LORA_ADAPTER_PATH = os.path.join(PROJECT_ROOT, "lora_gemma3_4b_medical")

DEVICE = "auto"  # "cuda" | "cpu" | "mps"
# How much to boost table-matching points when the user query contains a table
TABLE_QUERY_BOOST = 1.5