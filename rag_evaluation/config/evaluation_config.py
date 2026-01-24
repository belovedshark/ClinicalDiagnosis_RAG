"""
Configuration for RAG evaluation pipeline.
"""
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALUATION_DATA_PATH = os.path.join(PROJECT_ROOT, "evaluate_RAG.jsonl")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "rag_evaluation", "results")

# RAG System Configuration
TOP_K_RETRIEVAL = 5  # Number of context chunks to retrieve
RAG_DEVICE = "auto"  # Device for RAG models: "auto", "cuda", "mps", "cpu"

# Evaluation Settings
BATCH_SIZE = 1  # Process one case at a time to avoid memory issues
SAVE_INTERVAL = 5  # Save intermediate results every N cases

# Output files
RAG_INFERENCE_RESULTS_FILE = os.path.join(RESULTS_DIR, "rag_inference_results.json")
RAG_INFERENCE_CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "rag_inference_checkpoint.json")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)
