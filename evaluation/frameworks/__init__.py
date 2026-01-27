"""
Evaluation frameworks for clinical diagnosis RAG system.

This module provides integrations with external evaluation frameworks:
- RAGAS: RAG-specific metrics (faithfulness, context quality, etc.)
- DeepEval: Reasoning evaluation with G-Eval metrics
- Batched: Fast single-prompt evaluator (all metrics in one LLM call)
"""

from .ragas_evaluator import RagasEvaluator
from .deepeval_evaluator import DeepEvalReasoningEvaluator
from .batched_evaluator import BatchedLLMEvaluator

__all__ = ["RagasEvaluator", "DeepEvalReasoningEvaluator", "BatchedLLMEvaluator"]
