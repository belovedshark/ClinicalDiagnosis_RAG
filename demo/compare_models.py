#!/usr/bin/env python3
"""
Clinical Diagnosis Model Comparison Demo

This script runs all 4 model variants on the same clinical question
and displays a side-by-side comparison of their diagnoses.

Model variants:
1. Base Model - LLM only (google/gemma-3-4b-it)
2. RAG - Base LLM with retrieved context
3. Fine-tuned - LoRA adapter on gemma-2b-it
4. Fine-tuned + RAG - LoRA adapter with retrieved context

Usage:
    # Default sample question
    python -m demo.compare_models
    
    # Custom question
    python -m demo.compare_models --question "A 24-year-old man presents with..."
    
    # From sample questions
    python -m demo.compare_models --sample-id 5
    
    # List available samples
    python -m demo.compare_models --list-samples
    
    # Select specific models to run
    python -m demo.compare_models --models base rag
"""

import sys
import os
import argparse
import time
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lazy imports for heavy modules - only import when needed
Generator = None
Retriever = None
LLM_MODEL = None
LORA_BASE_MODEL = None
LORA_ADAPTER_PATH = None

def _load_rag_modules():
    """Lazy load RAG modules to avoid slow startup for simple operations."""
    global Generator, Retriever, LLM_MODEL, LORA_BASE_MODEL, LORA_ADAPTER_PATH
    if Generator is None:
        from rag.generator import Generator as _Generator
        from rag.retriever import Retriever as _Retriever
        from rag.config import LLM_MODEL as _LLM, LORA_BASE_MODEL as _LORA_BASE, LORA_ADAPTER_PATH as _LORA_PATH
        Generator = _Generator
        Retriever = _Retriever
        LLM_MODEL = _LLM
        LORA_BASE_MODEL = _LORA_BASE
        LORA_ADAPTER_PATH = _LORA_PATH

from demo.sample_questions import SAMPLE_QUESTIONS, get_sample_question, list_sample_questions


# ==============================================================================
# ANSWER CLEANING
# ==============================================================================

def clean_diagnosis_answer(answer: str) -> str:
    """
    Clean up model output to extract just the disease name.
    
    Handles common output patterns:
    - "The most likely diagnosis is X"
    - "Diagnosis: X"
    - Extra explanation after the disease name
    """
    if not answer:
        return "Unable to determine diagnosis"
    
    answer = answer.strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        "the most likely diagnosis is ",
        "most likely diagnosis is ",
        "the diagnosis is ",
        "diagnosis is ",
        "diagnosis: ",
        "answer: ",
        "the answer is ",
    ]
    answer_lower = answer.lower()
    for prefix in prefixes_to_remove:
        if answer_lower.startswith(prefix):
            answer = answer[len(prefix):].strip()
            break
    
    # Take first line if multiline
    if '\n' in answer:
        answer = answer.split('\n')[0].strip()
    
    # Remove trailing explanation (after period or comma followed by explanation)
    # But preserve disease names with periods like "P. falciparum"
    if '. ' in answer and not answer.startswith('P.'):
        parts = answer.split('. ')
        # Take first sentence if it's reasonably short
        if len(parts[0]) < 80:
            answer = parts[0]
    
    # Remove parenthetical explanations if answer is too long
    if len(answer) > 60 and '(' in answer:
        # Keep short parentheticals like "(visceral)" but remove long ones
        paren_start = answer.find('(')
        paren_end = answer.find(')')
        if paren_end > paren_start and paren_end - paren_start > 30:
            answer = answer[:paren_start].strip()
    
    # Truncate very long answers
    if len(answer) > 80:
        # Try to find a natural break point
        for sep in [', which', ', a ', ' - ', '. ']:
            if sep in answer:
                answer = answer.split(sep)[0].strip()
                break
        # Final truncation if still too long
        if len(answer) > 80:
            answer = answer[:77] + "..."
    
    # Clean up trailing punctuation
    answer = answer.rstrip('.,;:')
    
    # Capitalize first letter
    if answer and answer[0].islower():
        answer = answer[0].upper() + answer[1:]
    
    return answer if answer else "Unable to determine diagnosis"


# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================

BASE_MODEL_PROMPT = """You are an expert clinical diagnostician. Analyze the patient case and provide a diagnosis.

PATIENT CASE:
{question}

Based on the symptoms, travel history, and clinical presentation described above, provide the most likely diagnosis.

RESPONSE FORMAT:
You must respond with ONLY the disease name. No explanations, no formatting, no punctuation.

DIAGNOSIS:"""

RAG_PROMPT = """You are an expert clinical diagnostician. Based on the reference literature provided, analyze the patient case and provide the most likely diagnosis.

REFERENCE LITERATURE:
{context}

PATIENT CASE:
{question}

Based on the symptoms described and the reference literature above, what is the most likely diagnosis?

Respond with ONLY the disease name. No explanations.

DIAGNOSIS:"""

FINETUNED_PROMPT = """### Instruction:
You are an expert clinical diagnostician. Analyze the following patient case and provide the most likely diagnosis.

PATIENT CASE:
{question}

Based on the symptoms, travel history, and clinical presentation described above, provide the most likely diagnosis. Respond with ONLY the disease name.

### Response:
"""

FINETUNED_RAG_PROMPT = """### Instruction:
You are an expert clinical diagnostician. Based on the reference literature provided, analyze the patient case and provide the most likely diagnosis.

REFERENCE LITERATURE:
{context}

PATIENT CASE:
{question}

Based on the symptoms described and the reference literature above, what is the most likely diagnosis? Respond with ONLY the disease name.

### Response:
"""


# ==============================================================================
# MODEL CLASSES
# ==============================================================================

def _cleanup_memory():
    """Force garbage collection and clear GPU/MPS memory."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    except Exception:
        pass


class ModelRunner:
    """Base class for running inference on a model."""
    
    name: str = "Unknown"
    description: str = ""
    
    def __init__(self):
        self.loaded = False
    
    def load(self):
        """Load the model. Override in subclass."""
        pass
    
    def run(self, question: str) -> Dict[str, Any]:
        """Run inference. Override in subclass."""
        raise NotImplementedError
    
    def cleanup(self):
        """Clean up resources and free memory."""
        self.loaded = False
        _cleanup_memory()


class BaseModelRunner(ModelRunner):
    """Runner for base LLM model without RAG."""
    
    name = "BASE MODEL"
    
    @property
    def description(self):
        _load_rag_modules()
        return f"LLM only ({LLM_MODEL})"
    
    def __init__(self, device: str = "auto"):
        super().__init__()
        self.device = device
        self.generator = None
    
    def load(self):
        if not self.loaded:
            _load_rag_modules()
            print(f"  Loading {self.name}...")
            self.generator = Generator(device=self.device, use_lora=False)
            self.loaded = True
    
    def run(self, question: str) -> Dict[str, Any]:
        self.load()
        
        start_time = time.time()
        answer = self.generator.generate(
            context="",
            question=question,
            custom_prompt=BASE_MODEL_PROMPT
        )
        elapsed = time.time() - start_time
        
        return {
            "answer": self._clean_answer(answer),
            "contexts": [],
            "elapsed_time": elapsed,
            "model_name": LLM_MODEL
        }
    
    def _clean_answer(self, answer: str) -> str:
        """Clean up model output."""
        return clean_diagnosis_answer(answer)
    
    def cleanup(self):
        """Clean up model resources."""
        if self.generator is not None:
            del self.generator
            self.generator = None
        super().cleanup()


class RAGModelRunner(ModelRunner):
    """Runner for base LLM with RAG retrieval."""
    
    name = "RAG"
    
    @property
    def description(self):
        _load_rag_modules()
        return f"Base LLM + Retrieved Context ({LLM_MODEL})"
    
    def __init__(self, device: str = "auto", top_k: int = 5):
        super().__init__()
        self.device = device
        self.top_k = top_k
        self.generator = None
        self.retriever = None
    
    def load(self):
        if not self.loaded:
            _load_rag_modules()
            print(f"  Loading {self.name}...")
            self.retriever = Retriever(device=self.device)
            self.generator = Generator(device=self.device, use_lora=False)
            self.loaded = True
    
    def run(self, question: str) -> Dict[str, Any]:
        self.load()
        
        start_time = time.time()
        
        # Retrieve contexts
        query_emb = self.retriever.embed_query(query_text=question)
        results = self.retriever.client.query_points(
            collection_name=self.retriever.collection,
            query=query_emb.tolist(),
            using='text',
            limit=self.top_k,
            with_payload=True,
        )
        
        points = results.points if hasattr(results, 'points') else results
        
        contexts = []
        sources = []
        for p in points:
            if isinstance(p, dict):
                payload = p.get('payload', {}) or {}
            else:
                payload = getattr(p, 'payload', {}) or {}
            
            text = payload.get('text', '')
            source = payload.get('source', 'unknown')
            contexts.append(text)
            sources.append(source)
        
        context_text = "\n\n".join(contexts)
        
        # Generate answer
        answer = self.generator.generate(
            context=context_text,
            question=question,
            custom_prompt=RAG_PROMPT
        )
        elapsed = time.time() - start_time
        
        return {
            "answer": self._clean_answer(answer),
            "contexts": contexts,
            "sources": sources,
            "elapsed_time": elapsed,
            "model_name": LLM_MODEL
        }
    
    def _clean_answer(self, answer: str) -> str:
        return clean_diagnosis_answer(answer)
    
    def cleanup(self):
        """Clean up model resources."""
        if self.generator is not None:
            del self.generator
            self.generator = None
        if self.retriever is not None:
            del self.retriever
            self.retriever = None
        super().cleanup()


class FinetunedModelRunner(ModelRunner):
    """Runner for fine-tuned LoRA model without RAG."""
    
    name = "FINE-TUNED"
    
    @property
    def description(self):
        _load_rag_modules()
        return f"LoRA adapter ({LORA_BASE_MODEL})"
    
    def __init__(self, device: str = "auto", lora_path: str = None):
        super().__init__()
        self.device = device
        self._lora_path = lora_path  # Store initially, resolve later
        self.generator = None
    
    @property
    def lora_path(self):
        if self._lora_path is None:
            _load_rag_modules()
            return LORA_ADAPTER_PATH
        return self._lora_path
    
    def load(self):
        if not self.loaded:
            _load_rag_modules()
            print(f"  Loading {self.name}...")
            self.generator = Generator(device=self.device, use_lora=True, lora_path=self.lora_path)
            self.loaded = True
    
    def run(self, question: str) -> Dict[str, Any]:
        self.load()
        
        start_time = time.time()
        answer = self.generator.generate(
            context="",
            question=question,
            custom_prompt=FINETUNED_PROMPT
        )
        elapsed = time.time() - start_time
        
        return {
            "answer": self._clean_answer(answer),
            "contexts": [],
            "elapsed_time": elapsed,
            "model_name": f"{LORA_BASE_MODEL} + LoRA"
        }
    
    def _clean_answer(self, answer: str) -> str:
        return clean_diagnosis_answer(answer)
    
    def cleanup(self):
        """Clean up model resources."""
        if self.generator is not None:
            del self.generator
            self.generator = None
        super().cleanup()


class FinetunedRAGModelRunner(ModelRunner):
    """Runner for fine-tuned LoRA model with RAG retrieval."""
    
    name = "FINE-TUNED + RAG"
    
    @property
    def description(self):
        _load_rag_modules()
        return f"LoRA adapter + Retrieved Context ({LORA_BASE_MODEL})"
    
    def __init__(self, device: str = "auto", top_k: int = 5, lora_path: str = None):
        super().__init__()
        self.device = device
        self.top_k = top_k
        self._lora_path = lora_path  # Store initially, resolve later
        self.generator = None
        self.retriever = None
    
    @property
    def lora_path(self):
        if self._lora_path is None:
            _load_rag_modules()
            return LORA_ADAPTER_PATH
        return self._lora_path
    
    def load(self):
        if not self.loaded:
            _load_rag_modules()
            print(f"  Loading {self.name}...")
            self.retriever = Retriever(device=self.device)
            self.generator = Generator(device=self.device, use_lora=True, lora_path=self.lora_path)
            self.loaded = True
    
    def run(self, question: str) -> Dict[str, Any]:
        self.load()
        
        start_time = time.time()
        
        # Retrieve contexts
        query_emb = self.retriever.embed_query(query_text=question)
        results = self.retriever.client.query_points(
            collection_name=self.retriever.collection,
            query=query_emb.tolist(),
            using='text',
            limit=self.top_k,
            with_payload=True,
        )
        
        points = results.points if hasattr(results, 'points') else results
        
        contexts = []
        sources = []
        for p in points:
            if isinstance(p, dict):
                payload = p.get('payload', {}) or {}
            else:
                payload = getattr(p, 'payload', {}) or {}
            
            text = payload.get('text', '')
            source = payload.get('source', 'unknown')
            contexts.append(text)
            sources.append(source)
        
        context_text = "\n\n".join(contexts)
        
        # Generate answer
        answer = self.generator.generate(
            context=context_text,
            question=question,
            custom_prompt=FINETUNED_RAG_PROMPT
        )
        elapsed = time.time() - start_time
        
        return {
            "answer": self._clean_answer(answer),
            "contexts": contexts,
            "sources": sources,
            "elapsed_time": elapsed,
            "model_name": f"{LORA_BASE_MODEL} + LoRA"
        }
    
    def _clean_answer(self, answer: str) -> str:
        return clean_diagnosis_answer(answer)
    
    def cleanup(self):
        """Clean up model resources."""
        if self.generator is not None:
            del self.generator
            self.generator = None
        if self.retriever is not None:
            del self.retriever
            self.retriever = None
        super().cleanup()


# ==============================================================================
# DISPLAY FUNCTIONS
# ==============================================================================

def print_header(question: str, ground_truth: Optional[str] = None):
    """Print the demo header with question and ground truth."""
    print("\n" + "=" * 70)
    print("CLINICAL DIAGNOSIS COMPARISON DEMO")
    print("=" * 70)
    print(f"\nQUESTION:\n{question}")
    if ground_truth:
        print(f"\nGROUND TRUTH: {ground_truth}")
    print()


def print_model_result(index: int, runner: ModelRunner, result: Dict[str, Any], show_context: bool = False):
    """Print a single model's result."""
    print("-" * 70)
    print(f"{index}. {runner.name} ({runner.description})")
    print("-" * 70)
    
    # Show retrieved context info if available
    if result.get("contexts"):
        sources = result.get("sources", [])
        unique_sources = list(set(sources))[:3]
        source_str = ", ".join(os.path.basename(s) for s in unique_sources)
        print(f"   Retrieved {len(result['contexts'])} contexts from: {source_str}...")
        
        if show_context:
            print("\n   Context snippets:")
            for i, ctx in enumerate(result['contexts'][:2]):  # Show first 2
                snippet = ctx[:150] + "..." if len(ctx) > 150 else ctx
                print(f"   [{i+1}] {snippet}")
            print()
    
    print(f"   Diagnosis: {result['answer']}")
    print(f"   Time: {result['elapsed_time']:.2f}s")
    print()


def print_summary(results: Dict[str, Dict[str, Any]], ground_truth: Optional[str] = None):
    """Print a summary comparison of all results."""
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Table header
    print(f"{'Model':<25} {'Diagnosis':<30} {'Time':>8}")
    print("-" * 70)
    
    for name, result in results.items():
        diagnosis = result['answer'][:28] + ".." if len(result['answer']) > 30 else result['answer']
        time_str = f"{result['elapsed_time']:.2f}s"
        
        # Check if matches ground truth
        match_indicator = ""
        if ground_truth:
            if result['answer'].lower().strip() == ground_truth.lower().strip():
                match_indicator = " [MATCH]"
        
        print(f"{name:<25} {diagnosis:<30} {time_str:>8}{match_indicator}")
    
    print("=" * 70)


# ==============================================================================
# MAIN COMPARISON FUNCTION
# ==============================================================================

def run_comparison(
    question: str,
    ground_truth: Optional[str] = None,
    models: List[str] = None,
    device: str = "auto",
    top_k: int = 5,
    show_context: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Run comparison across all specified models.
    
    Models are loaded one at a time and cleaned up after inference
    to avoid memory issues when running multiple large models.
    
    Args:
        question: The clinical case question
        ground_truth: Expected diagnosis (for comparison)
        models: List of model names to run ("base", "rag", "finetuned", "finetuned_rag")
                If None, runs all models
        device: Device to run on ("auto", "cuda", "mps", "cpu")
        top_k: Number of contexts to retrieve for RAG models
        show_context: Whether to show retrieved context snippets
        
    Returns:
        Dictionary of model name -> result
    """
    # Default to all models
    if models is None:
        models = ["base", "rag", "finetuned", "finetuned_rag"]
    
    # Define model configurations (created on-demand to save memory)
    model_configs = {
        "base": lambda: BaseModelRunner(device=device),
        "rag": lambda: RAGModelRunner(device=device, top_k=top_k),
        "finetuned": lambda: FinetunedModelRunner(device=device),
        "finetuned_rag": lambda: FinetunedRAGModelRunner(device=device, top_k=top_k),
    }
    
    # Print header
    print_header(question, ground_truth)
    
    # Run inference one model at a time to conserve memory
    results = {}
    idx = 0
    for model_key in models:
        if model_key not in model_configs:
            print(f"  Warning: Unknown model '{model_key}', skipping")
            continue
        
        idx += 1
        runner = None
        try:
            # Create and load model
            print(f"Loading model {idx}/{len(models)}...")
            runner = model_configs[model_key]()
            runner.load()
            
            # Run inference
            result = runner.run(question)
            results[runner.name] = result
            print_model_result(idx, runner, result, show_context)
            
        except Exception as e:
            model_name = runner.name if runner else model_key.upper()
            print(f"  Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {
                "answer": f"ERROR: {str(e)}",
                "contexts": [],
                "elapsed_time": 0.0
            }
        finally:
            # Clean up to free memory before next model
            if runner is not None:
                print(f"  Cleaning up {runner.name} to free memory...")
                runner.cleanup()
    
    # Print summary
    print_summary(results, ground_truth)
    
    return results


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare clinical diagnosis across 4 model variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m demo.compare_models                           # Default sample
  python -m demo.compare_models --sample-id 5             # Use sample question #5
  python -m demo.compare_models --question "A patient..." # Custom question
  python -m demo.compare_models --list-samples            # List available samples
  python -m demo.compare_models --models base rag         # Run specific models
        """
    )
    
    # Question input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--question", "-q",
        type=str,
        help="Custom clinical question to evaluate"
    )
    input_group.add_argument(
        "--sample-id", "-s",
        type=int,
        help="ID of sample question to use (1-10)"
    )
    input_group.add_argument(
        "--list-samples", "-l",
        action="store_true",
        help="List all available sample questions"
    )
    
    # Model selection
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        choices=["base", "rag", "finetuned", "finetuned_rag"],
        help="Which models to run (default: all)"
    )
    
    # Options
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to run models on (default: auto)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of contexts to retrieve for RAG (default: 5)"
    )
    parser.add_argument(
        "--show-context", "-c",
        action="store_true",
        help="Show retrieved context snippets"
    )
    parser.add_argument(
        "--ground-truth", "-g",
        type=str,
        help="Ground truth diagnosis for comparison"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the demo."""
    args = parse_args()
    
    # List samples if requested
    if args.list_samples:
        list_sample_questions()
        return
    
    # Get question
    if args.question:
        question = args.question
        ground_truth = args.ground_truth
    elif args.sample_id:
        sample = get_sample_question(question_id=args.sample_id)
        question = sample["question"]
        ground_truth = sample["ground_truth"]
        print(f"\nUsing sample question #{sample['id']}: {sample['name']}")
    else:
        # Default: use first sample
        sample = get_sample_question(question_id=1)
        question = sample["question"]
        ground_truth = sample["ground_truth"]
        print(f"\nUsing default sample: {sample['name']}")
    
    # Run comparison
    run_comparison(
        question=question,
        ground_truth=ground_truth,
        models=args.models,
        device=args.device,
        top_k=args.top_k,
        show_context=args.show_context
    )


if __name__ == "__main__":
    main()
