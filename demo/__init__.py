"""
Demo module for comparing clinical diagnosis models.

This module provides tools to run and compare outputs from all 4 model variants:
- Base model (LLM only)
- RAG (LLM + retrieval)
- Fine-tuned (LoRA adapter)
- Fine-tuned + RAG (LoRA + retrieval)
"""

from .sample_questions import SAMPLE_QUESTIONS, get_sample_question

__all__ = ["SAMPLE_QUESTIONS", "get_sample_question"]
