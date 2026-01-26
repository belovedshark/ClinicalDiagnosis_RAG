#!/usr/bin/env python3
"""
Main entry point for running evaluations.

Usage:
    python -m evaluation.run_evaluation rag
    python -m evaluation.run_evaluation base_model
    python -m evaluation.run_evaluation finetuned
    python -m evaluation.run_evaluation finetuned_rag
    python -m evaluation.run_evaluation all
"""
import argparse
import sys


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
        "finetuned": run_finetuned,
        "finetuned_rag": run_finetuned_rag,
    }
    
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
