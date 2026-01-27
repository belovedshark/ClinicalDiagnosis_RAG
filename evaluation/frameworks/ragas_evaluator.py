"""
RAGAS Evaluator - RAG-specific evaluation metrics.

This module provides RAGAS (Retrieval Augmented Generation Assessment) metrics
for evaluating the quality of RAG pipeline outputs.

Metrics included:
- Faithfulness: Is the answer factually grounded in retrieved contexts?
- Answer Relevancy: How relevant is the answer to the question?
- Context Precision: How much of the retrieved context is actually relevant?
- Context Recall: Does the context contain info needed to answer?
"""
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("Warning: RAGAS not installed. Install with: pip install ragas datasets")


class RagasEvaluator:
    """
    Evaluator using RAGAS metrics for RAG quality assessment.
    
    RAGAS provides metrics specifically designed for evaluating
    Retrieval-Augmented Generation systems.
    """
    
    AVAILABLE_METRICS = {
        "faithfulness": "Measures if the answer is factually grounded in the context",
        "answer_relevancy": "Measures how relevant the answer is to the question",
        "context_precision": "Measures how much of the context is relevant",
        "context_recall": "Measures if context contains info needed to answer"
    }
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the RAGAS evaluator.
        
        Args:
            metrics: List of metric names to use. If None, uses all available.
            llm_model: OpenAI model for LLM-based metrics (e.g., faithfulness)
            embedding_model: OpenAI model for embedding-based metrics
        """
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAS is not installed. Install with: pip install ragas datasets"
            )
        
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        # Select metrics
        if metrics is None:
            metrics = list(self.AVAILABLE_METRICS.keys())
        
        self.metric_names = metrics
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Set up the RAGAS metrics with configured LLM."""
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            
            # Configure LLM for RAGAS
            llm = LangchainLLMWrapper(ChatOpenAI(model=self.llm_model))
            embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(model=self.embedding_model)
            )
            
            # Build metric list
            self.metrics = []
            metric_map = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall
            }
            
            for name in self.metric_names:
                if name in metric_map:
                    metric = metric_map[name]
                    # Configure metric with LLM
                    metric.llm = llm
                    if hasattr(metric, 'embeddings'):
                        metric.embeddings = embeddings
                    self.metrics.append(metric)
            
            self._llm_configured = True
            
        except ImportError:
            print("Warning: langchain-openai not installed. Using default RAGAS config.")
            print("Install with: pip install langchain-openai")
            
            # Fall back to default metrics without custom LLM
            metric_map = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall
            }
            self.metrics = [metric_map[name] for name in self.metric_names if name in metric_map]
            self._llm_configured = False
    
    def _prepare_dataset(self, results: List[Dict[str, Any]]) -> Dataset:
        """
        Convert inference results to RAGAS Dataset format.
        
        Args:
            results: List of inference result dictionaries
            
        Returns:
            HuggingFace Dataset formatted for RAGAS
        """
        # Filter out error cases and cases without contexts
        valid_results = [
            r for r in results 
            if not r.get("answer", "").startswith("ERROR")
            and r.get("contexts")  # Must have contexts for RAGAS
        ]
        
        if not valid_results:
            raise ValueError("No valid results with contexts found for RAGAS evaluation")
        
        # RAGAS expects specific column names
        data = {
            "question": [r["question"] for r in valid_results],
            "answer": [r["answer"] for r in valid_results],
            "contexts": [r["contexts"] for r in valid_results],
            "ground_truth": [r["ground_truth"] for r in valid_results]
        }
        
        return Dataset.from_dict(data)
    
    def evaluate_results(
        self,
        results: List[Dict[str, Any]],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate inference results using RAGAS metrics.
        
        Args:
            results: List of inference result dictionaries from evaluators
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing:
            - scores: Per-metric average scores
            - per_case_scores: Per-case breakdown
            - metadata: Evaluation metadata
        """
        if verbose:
            print("\n" + "="*60)
            print("RAGAS EVALUATION")
            print("="*60)
            print(f"Metrics: {', '.join(self.metric_names)}")
            print(f"LLM Model: {self.llm_model}")
        
        # Prepare dataset
        dataset = self._prepare_dataset(results)
        
        if verbose:
            print(f"Evaluating {len(dataset)} cases...")
        
        # Run RAGAS evaluation
        try:
            ragas_result = evaluate(
                dataset=dataset,
                metrics=self.metrics
            )
        except Exception as e:
            print(f"Error during RAGAS evaluation: {e}")
            raise
        
        # Extract scores
        scores = {}
        for metric_name in self.metric_names:
            if metric_name in ragas_result:
                scores[metric_name] = float(ragas_result[metric_name])
        
        # Get per-case scores from the result dataframe
        per_case_scores = []
        if hasattr(ragas_result, 'to_pandas'):
            df = ragas_result.to_pandas()
            for idx, row in df.iterrows():
                case_scores = {
                    "question": row.get("question", "")[:100],
                    "scores": {}
                }
                for metric_name in self.metric_names:
                    if metric_name in row:
                        case_scores["scores"][metric_name] = float(row[metric_name])
                per_case_scores.append(case_scores)
        
        # Print summary
        if verbose:
            print("\n📊 RAGAS Scores:")
            print("-" * 40)
            for metric, score in scores.items():
                print(f"  {metric}: {score:.4f}")
            print("-" * 40)
        
        return {
            "scores": scores,
            "per_case_scores": per_case_scores,
            "metadata": {
                "num_cases": len(dataset),
                "metrics_used": self.metric_names,
                "llm_model": self.llm_model,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def evaluate_from_file(
        self,
        results_path: str,
        output_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate results from a JSON file.
        
        Args:
            results_path: Path to inference_results.json
            output_path: Optional path to save RAGAS results
            verbose: Whether to print progress
            
        Returns:
            RAGAS evaluation results
        """
        # Load results
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if verbose:
            print(f"📂 Loaded {len(results)} results from {results_path}")
        
        # Run evaluation
        ragas_results = self.evaluate_results(results, verbose=verbose)
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(ragas_results, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"💾 RAGAS results saved to {output_path}")
        
        return ragas_results


def main():
    """Run RAGAS evaluation on RAG results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on inference results")
    parser.add_argument(
        "results_path",
        help="Path to inference_results.json file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to save RAGAS results",
        default=None
    )
    parser.add_argument(
        "--metrics", "-m",
        nargs="+",
        help="Metrics to use (default: all)",
        default=None
    )
    parser.add_argument(
        "--model",
        help="OpenAI model for evaluation",
        default="gpt-4o-mini"
    )
    
    args = parser.parse_args()
    
    evaluator = RagasEvaluator(
        metrics=args.metrics,
        llm_model=args.model
    )
    
    evaluator.evaluate_from_file(
        results_path=args.results_path,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
