#!/usr/bin/env python3
"""
Main entry point for running evaluations.

Usage:
    # Run inference evaluators
    python -m evaluation.run_evaluation rag
    python -m evaluation.run_evaluation base_model
    python -m evaluation.run_evaluation finetuned
    python -m evaluation.run_evaluation finetuned_rag
    python -m evaluation.run_evaluation all
    
    # Run framework evaluation (fast batched - all metrics in single LLM call)
    python -m evaluation.run_evaluation frameworks --results PATH
    python -m evaluation.run_evaluation frameworks --results PATH --output custom_name.json
    python -m evaluation.run_evaluation frameworks --results PATH --name rag
"""
import argparse
import sys
import os
import json
from datetime import datetime


def run_rag():
    """Run RAG evaluation."""
    from evaluation.rag.evaluator import main
    main()


def run_base_model():
    """Run base model evaluation."""
    from evaluation.base_model.evaluator import main
    main()


def run_finetuned():
    """Run fine-tuned model evaluation (no RAG)."""
    from evaluation.finetuned.evaluator import main
    main()


def run_finetuned_rag():
    """Run fine-tuned model + RAG evaluation."""
    from evaluation.finetuned_rag.evaluator import main
    main()


def _detect_model_name(results_path: str) -> str:
    """
    Auto-detect model name from results path.
    
    Examples:
        evaluation/rag/results/inference_results.json → rag
        evaluation/base_model/results/inference_results.json → base_model
        /path/to/finetuned_results.json → finetuned_results
    """
    # Try to extract from path structure (evaluation/{model}/results/)
    path_parts = results_path.replace("\\", "/").split("/")
    
    # Look for known model names in path
    known_models = ["rag", "base_model", "finetuned", "finetuned_rag"]
    for part in path_parts:
        if part in known_models:
            return part
    
    # Fall back to filename without extension
    basename = os.path.splitext(os.path.basename(results_path))[0]
    return basename


def run_frameworks(results_path: str, output_path: str = None, name: str = None):
    """
    Run fast batched LLM evaluation (all metrics in single call per case).
    
    This is the default and recommended approach - much faster than 
    running RAGAS/DeepEval separately.
    
    Args:
        results_path: Path to inference results JSON
        output_path: Full output path (optional)
        name: Model name for output file (optional, auto-detected if not provided)
    """
    from evaluation.frameworks.batched_evaluator import BatchedLLMEvaluator
    from evaluation.config import FRAMEWORKS_RESULTS_DIR, DEEPEVAL_MODEL
    
    # Determine output path
    if output_path is None:
        # Use provided name or auto-detect from path
        model_name = name if name else _detect_model_name(results_path)
        output_path = os.path.join(FRAMEWORKS_RESULTS_DIR, f"{model_name}_evaluation.json")
    elif not output_path.endswith('.json'):
        # If output is just a name without .json, add it
        output_path = os.path.join(FRAMEWORKS_RESULTS_DIR, f"{output_path}.json")
    
    print(f"📁 Output will be saved to: {output_path}")
    
    evaluator = BatchedLLMEvaluator(model=DEEPEVAL_MODEL)
    
    return evaluator.evaluate_from_file(
        results_path=results_path,
        output_path=output_path
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run clinical diagnosis model evaluations"
    )
    parser.add_argument(
        "evaluator",
        choices=[
            "rag", "base_model", "finetuned", "finetuned_rag", "all",
            "frameworks"
        ],
        help="Which evaluator to run"
    )
    parser.add_argument(
        "--results", "-r",
        help="Path to inference_results.json (required for frameworks)",
        default=None
    )
    parser.add_argument(
        "--output", "-o",
        help="Output path for framework evaluation results",
        default=None
    )
    parser.add_argument(
        "--name", "-n",
        help="Model name for output file (e.g., 'rag', 'base_model'). Auto-detected if not provided.",
        default=None
    )
    
    args = parser.parse_args()
    
    # Available inference evaluators
    inference_evaluators = {
        "rag": run_rag,
        "base_model": run_base_model,
        "finetuned": run_finetuned,
        "finetuned_rag": run_finetuned_rag,
    }
    
    if args.evaluator == "frameworks":
        # Framework evaluation requires results path
        if not args.results:
            # Try to find default results file
            from evaluation.config import RAG_RESULTS_FILE
            if os.path.exists(RAG_RESULTS_FILE):
                args.results = RAG_RESULTS_FILE
                print(f"Using default results file: {args.results}")
            else:
                print("Error: --results PATH is required for framework evaluation")
                print("Example: python -m evaluation.run_evaluation frameworks --results evaluation/rag/results/inference_results.json")
                sys.exit(1)
        
        if not os.path.exists(args.results):
            print(f"Error: Results file not found: {args.results}")
            sys.exit(1)
        
        run_frameworks(args.results, args.output, args.name)
    
    elif args.evaluator == "all":
        print("Running all inference evaluators...")
        for name, func in inference_evaluators.items():
            print(f"\n{'='*80}")
            print(f"Running {name} evaluator...")
            print('='*80)
            try:
                func()
            except Exception as e:
                print(f"❌ Error in {name}: {e}")
    else:
        inference_evaluators[args.evaluator]()


if __name__ == "__main__":
    main()
