"""
RAGAS Evaluation for Clinical Diagnosis RAG System.

Evaluates RAG inference results using RAGAS metrics:
- Faithfulness: Is the answer grounded in the retrieved contexts?
- Answer Relevancy: Does the answer address the question?
- Context Precision: Are the retrieved contexts relevant to the question?

Supports both OpenAI and Google Gemini as LLM judges (strategy pattern).
"""
import argparse
import json
import os
import time
import warnings
from datetime import datetime
from typing import Optional
from abc import ABC, abstractmethod

import numpy as np
from dotenv import load_dotenv
from datasets import Dataset

# Suppress deprecation warnings for older RAGAS API (still works with evaluate)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

from rag_evaluation.config.evaluation_config import (
    RAG_INFERENCE_RESULTS_FILE,
    RAGAS_LLM_PROVIDER,
    RAGAS_OPENAI_MODEL,
    RAGAS_GEMINI_MODEL,
    RAGAS_RESULTS_FILE,
    RAGAS_RETRY_ATTEMPTS,
    RAGAS_RETRY_DELAY,
)

load_dotenv()


# ============== Strategy Pattern for LLM Providers ==============

class LLMProviderStrategy(ABC):
    """Abstract base class for LLM provider strategies."""
    
    @abstractmethod
    def get_llm(self) -> LangchainLLMWrapper:
        """Return a LangchainLLMWrapper for the provider."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name being used."""
        pass


class OpenAIStrategy(LLMProviderStrategy):
    """OpenAI LLM provider strategy."""
    
    def __init__(self, model_name: str = RAGAS_OPENAI_MODEL):
        self.model_name = model_name
        
    def get_llm(self) -> LangchainLLMWrapper:
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Get an API key at https://platform.openai.com/api-keys"
            )
        
        openai_llm = ChatOpenAI(
            model=self.model_name,
            api_key=api_key,
            temperature=0,
        )
        return LangchainLLMWrapper(openai_llm)
    
    def get_model_name(self) -> str:
        return f"openai/{self.model_name}"


class GeminiStrategy(LLMProviderStrategy):
    """Google Gemini LLM provider strategy."""
    
    def __init__(self, model_name: str = RAGAS_GEMINI_MODEL):
        self.model_name = model_name
        
    def get_llm(self) -> LangchainLLMWrapper:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Get a free API key at https://makersuite.google.com/app/apikey"
            )
        
        gemini_llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=api_key,
            temperature=0,
        )
        return LangchainLLMWrapper(gemini_llm)
    
    def get_model_name(self) -> str:
        return f"gemini/{self.model_name}"


def get_llm_strategy(provider: str = RAGAS_LLM_PROVIDER) -> LLMProviderStrategy:
    """Factory function to get the appropriate LLM strategy."""
    strategies = {
        "openai": OpenAIStrategy,
        "gemini": GeminiStrategy,
    }
    
    provider = provider.lower()
    if provider not in strategies:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(strategies.keys())}")
    
    return strategies[provider]()


def load_inference_results(filepath: str = RAG_INFERENCE_RESULTS_FILE) -> list[dict]:
    """Load RAG inference results from JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Inference results not found at {filepath}. "
            "Run 'python run_rag_inference.py' first."
        )

    with open(filepath, "r") as f:
        data = json.load(f)

    # Handle both list format and dict with 'results' key
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data


def transform_to_ragas_dataset(inference_results: list[dict]) -> Dataset:
    """Transform inference results to RAGAS Dataset format."""
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for result in inference_results:
        data["question"].append(result["question"])
        data["answer"].append(result["answer"])
        data["contexts"].append(result["contexts"])
        data["ground_truth"].append(result.get("ground_truth", ""))

    return Dataset.from_dict(data)


def setup_evaluator(provider: str = RAGAS_LLM_PROVIDER) -> tuple[LangchainLLMWrapper, str]:
    """Set up LLM as RAGAS evaluator using strategy pattern.
    
    Returns:
        Tuple of (LangchainLLMWrapper, model_name_string)
    """
    strategy = get_llm_strategy(provider)
    return strategy.get_llm(), strategy.get_model_name()


def setup_embeddings() -> LangchainEmbeddingsWrapper:
    """Set up embeddings model for metrics that require it."""
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return LangchainEmbeddingsWrapper(hf_embeddings)


def run_evaluation(
    dataset: Dataset,
    llm: LangchainLLMWrapper,
    embeddings: LangchainEmbeddingsWrapper,
    retry_attempts: int = RAGAS_RETRY_ATTEMPTS,
    retry_delay: int = RAGAS_RETRY_DELAY,
) -> dict:
    """
    Run RAGAS evaluation with retry logic for rate limits.

    Returns dict with 'scores' (per-sample) and 'aggregate' (summary stats).
    """
    metrics = [faithfulness, answer_relevancy, context_precision]

    for attempt in range(retry_attempts):
        try:
            print(f"\nRunning RAGAS evaluation (attempt {attempt + 1}/{retry_attempts})...")
            print(f"Evaluating {len(dataset)} samples with {len(metrics)} metrics...")

            results = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
            )

            # Extract per-sample scores
            scores_df = results.to_pandas()
            per_case_scores = scores_df.to_dict(orient="records")

            # Calculate aggregate statistics
            aggregate = {}
            for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
                if metric in scores_df.columns:
                    values = scores_df[metric].dropna()
                    if len(values) > 0:
                        aggregate[metric] = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "min": float(np.min(values)),
                            "max": float(np.max(values)),
                        }

            return {
                "scores": per_case_scores,
                "aggregate": aggregate,
            }

        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg or "quota" in error_msg or "429" in error_msg:
                if attempt < retry_attempts - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Rate limit hit. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
            raise

    raise RuntimeError(f"Evaluation failed after {retry_attempts} attempts")


def print_summary(aggregate: dict, num_cases: int, model: str) -> None:
    """Print formatted evaluation summary to console."""
    print("\n" + "=" * 55)
    print("           RAGAS Evaluation Results")
    print("=" * 55)
    print(f"Cases evaluated: {num_cases}")
    print(f"Evaluator model: {model}")
    print()
    print(f"{'METRIC':<20} {'MEAN':>8} {'STD':>8} {'MIN':>8} {'MAX':>8}")
    print("-" * 55)

    metric_display_names = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision",
    }

    for metric, stats in aggregate.items():
        display_name = metric_display_names.get(metric, metric)
        print(
            f"{display_name:<20} "
            f"{stats['mean']:>8.2f} "
            f"{stats['std']:>8.2f} "
            f"{stats['min']:>8.2f} "
            f"{stats['max']:>8.2f}"
        )

    print("=" * 55)


def save_results(
    evaluation_results: dict,
    inference_results: list[dict],
    output_path: str = RAGAS_RESULTS_FILE,
) -> None:
    """Save evaluation results to JSON file."""
    # Combine case IDs with scores
    per_case_scores = []
    for i, scores in enumerate(evaluation_results["scores"]):
        case_data = {
            "case_id": inference_results[i].get("case_id", f"case_{i+1}"),
            **{k: v for k, v in scores.items() if k not in ["question", "answer", "contexts", "ground_truth"]},
        }
        per_case_scores.append(case_data)

    output = {
        "evaluation_metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": evaluation_results.get("model_name", "unknown"),
            "num_cases": len(inference_results),
            "metrics": ["faithfulness", "answer_relevancy", "context_precision"],
        },
        "aggregate_scores": evaluation_results["aggregate"],
        "per_case_scores": per_case_scores,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    provider: Optional[str] = None,
) -> None:
    """Run RAGAS evaluation pipeline."""
    input_path = input_path or RAG_INFERENCE_RESULTS_FILE
    output_path = output_path or RAGAS_RESULTS_FILE
    provider = provider or RAGAS_LLM_PROVIDER

    print("=" * 55)
    print("     Clinical Diagnosis RAG - RAGAS Evaluation")
    print("=" * 55)

    # Load inference results
    print(f"\nLoading inference results from: {input_path}")
    inference_results = load_inference_results(input_path)
    print(f"Loaded {len(inference_results)} cases")

    # Transform to RAGAS dataset format
    dataset = transform_to_ragas_dataset(inference_results)

    # Setup evaluator using strategy pattern
    print(f"\nSetting up evaluator with provider: {provider}")
    evaluator_llm, model_name = setup_evaluator(provider)
    print(f"Using model: {model_name}")

    # Setup embeddings
    print("Setting up embeddings model...")
    embeddings = setup_embeddings()

    # Run evaluation
    evaluation_results = run_evaluation(dataset, evaluator_llm, embeddings)
    evaluation_results["model_name"] = model_name  # Add model name for saving

    # Print summary
    print_summary(
        evaluation_results["aggregate"],
        len(inference_results),
        model_name,
    )

    # Save results
    save_results(evaluation_results, inference_results, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on RAG inference results"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to inference results JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "gemini"],
        help="LLM provider to use: 'openai' or 'gemini' (default from config)",
    )

    args = parser.parse_args()
    main(input_path=args.input, output_path=args.output, provider=args.provider)
