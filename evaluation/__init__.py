"""
Unified Evaluation Framework for Clinical Diagnosis Models.

This framework provides a consistent interface for evaluating different
model configurations:
- RAG (Retrieval-Augmented Generation)
- Base Model (LLM without retrieval)
- Fine-tuned Model
- Fine-tuned + RAG

All evaluators implement the BaseEvaluator interface for consistent
input/output formats.
"""

from .base_interface import BaseEvaluator
from .config import (
    TEST_CASES_PATH,
    RAG_RESULTS_FILE,
    BASE_MODEL_RESULTS_FILE,
    FINETUNED_RESULTS_FILE,
    FINETUNED_RAG_RESULTS_FILE,
)

__all__ = [
    'BaseEvaluator',
    'TEST_CASES_PATH',
    'RAG_RESULTS_FILE',
    'BASE_MODEL_RESULTS_FILE',
    'FINETUNED_RESULTS_FILE',
    'FINETUNED_RAG_RESULTS_FILE',
]
