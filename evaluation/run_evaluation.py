#!/usr/bin/env python3
"""
Main entry point for running evaluations.

Usage:
    python -m evaluation.run_evaluation rag
    python -m evaluation.run_evaluation base_model
    python -m evaluation.run_evaluation all

RAG Evaluation Features:
    - Hybrid search (semantic + BM25)
    - Cross-encoder reranking
    
Future (not implemented yet):
    python -m evaluation.run_evaluation finetuned
    python -m evaluation.run_evaluation finetuned_rag
"""
import argparse
import sys


def run_rag():
    """Run RAG evaluation with hybrid search and reranking."""
    from evaluation.rag.evaluator import main
    main()


def run_base_model():
    """Run base model evaluation."""
    from evaluation.base_model.evaluator import main
    main()


def main():
    parser = argparse.ArgumentParser(
        description="Run clinical diagnosis model evaluations"
    )
    parser.add_argument(
        "evaluator",
        choices=["rag", "base_model", "finetuned", "finetuned_rag", "all"],
        help="Which evaluator to run"
    )
    
    args = parser.parse_args()
    
    # Available evaluators
    available = {
        "rag": run_rag,
        "base_model": run_base_model,
    }
    
    # Placeholder evaluators (not implemented yet)
    placeholders = ["finetuned", "finetuned_rag"]
    
    if args.evaluator in placeholders:
        print(f"❌ {args.evaluator} evaluator is not implemented yet.")
        print("   Available evaluators: rag, base_model")
        sys.exit(1)
    
    if args.evaluator == "all":
        print("Running all available evaluators...")
        for name, func in available.items():
            print(f"\n{'='*80}")
            print(f"Running {name} evaluator...")
            print('='*80)
            try:
                func()
            except Exception as e:
                print(f"❌ Error in {name}: {e}")
    else:
        available[args.evaluator]()


if __name__ == "__main__":
    main()
