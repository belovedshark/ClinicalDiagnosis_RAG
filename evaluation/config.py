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

# Matching settings for evaluation
USE_SEMANTIC_MATCHING = True  # Enable semantic similarity fallback
SEMANTIC_THRESHOLD = 0.85  # Similarity threshold (0.0-1.0) for semantic matching
EMBEDDING_MODEL = "thenlper/gte-small"  # Model for semantic embeddings

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

# =============================================================================
# FRAMEWORK EVALUATION SETTINGS (RAGAS & DeepEval)
# =============================================================================

# RAGAS settings
RAGAS_ENABLED = True
RAGAS_LLM_MODEL = "gpt-4o-mini"  # OpenAI model for RAGAS metrics
RAGAS_EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
RAGAS_METRICS = [
    "faithfulness",       # Is answer grounded in context?
    "answer_relevancy",   # Is answer relevant to question?
    "context_precision",  # How much context is relevant?
    "context_recall"      # Does context have needed info?
]

# DeepEval settings
DEEPEVAL_ENABLED = True
DEEPEVAL_MODEL = "gpt-4o-mini"  # Cost-effective option; also supports gpt-4, claude-3-opus
DEEPEVAL_THRESHOLD = 0.5  # Minimum threshold for passing (0.0-1.0)
DEEPEVAL_INCLUDE_HALLUCINATION = True  # Check for fabricated facts
DEEPEVAL_INCLUDE_RELEVANCY = True  # Check answer relevancy

# Clinical reasoning evaluation criteria (used by DeepEval G-Eval)
CLINICAL_REASONING_CRITERIA = """
Evaluate the clinical diagnostic reasoning based on:
1. Symptom Recognition: Are key symptoms correctly identified?
2. Differential Diagnosis: Are alternative diagnoses considered?
3. Evidence-Based: Is the conclusion supported by clinical evidence?
4. Logical Flow: Does the reasoning follow a logical diagnostic process?
"""

# Framework results directory
FRAMEWORKS_RESULTS_DIR = os.path.join(EVALUATION_ROOT, "frameworks", "results")

# Ensure all results directories exist
for results_dir in [RAG_RESULTS_DIR, BASE_MODEL_RESULTS_DIR, 
                    FINETUNED_RESULTS_DIR, FINETUNED_RAG_RESULTS_DIR,
                    FRAMEWORKS_RESULTS_DIR]:
    os.makedirs(results_dir, exist_ok=True)
