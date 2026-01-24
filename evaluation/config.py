"""
Configuration for the evaluation framework.

Shared configuration used by all evaluators.
"""
import os

# Base paths
EVALUATION_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVALUATION_ROOT)

# Input data
TEST_CASES_PATH = os.path.join(EVALUATION_ROOT, "test_cases.jsonl")

# Model configuration
TOP_K_RETRIEVAL = 5  # Number of context chunks to retrieve (for RAG)
DEVICE = "auto"  # Device for models: "auto", "cuda", "mps", "cpu"

# Evaluation settings
BATCH_SIZE = 1  # Process one case at a time
SAVE_INTERVAL = 5  # Save checkpoint every N cases

# Output paths for each evaluator type
RAG_RESULTS_DIR = os.path.join(EVALUATION_ROOT, "rag", "results")
RAG_RESULTS_FILE = os.path.join(RAG_RESULTS_DIR, "inference_results.json")
RAG_CHECKPOINT_FILE = os.path.join(RAG_RESULTS_DIR, "checkpoint.json")

BASE_MODEL_RESULTS_DIR = os.path.join(EVALUATION_ROOT, "base_model", "results")
BASE_MODEL_RESULTS_FILE = os.path.join(BASE_MODEL_RESULTS_DIR, "inference_results.json")
BASE_MODEL_CHECKPOINT_FILE = os.path.join(BASE_MODEL_RESULTS_DIR, "checkpoint.json")

FINETUNED_RESULTS_DIR = os.path.join(EVALUATION_ROOT, "finetuned", "results")
FINETUNED_RESULTS_FILE = os.path.join(FINETUNED_RESULTS_DIR, "inference_results.json")
FINETUNED_CHECKPOINT_FILE = os.path.join(FINETUNED_RESULTS_DIR, "checkpoint.json")

FINETUNED_RAG_RESULTS_DIR = os.path.join(EVALUATION_ROOT, "finetuned_rag", "results")
FINETUNED_RAG_RESULTS_FILE = os.path.join(FINETUNED_RAG_RESULTS_DIR, "inference_results.json")
FINETUNED_RAG_CHECKPOINT_FILE = os.path.join(FINETUNED_RAG_RESULTS_DIR, "checkpoint.json")

# Ensure all results directories exist
for results_dir in [RAG_RESULTS_DIR, BASE_MODEL_RESULTS_DIR, 
                    FINETUNED_RESULTS_DIR, FINETUNED_RAG_RESULTS_DIR]:
    os.makedirs(results_dir, exist_ok=True)
