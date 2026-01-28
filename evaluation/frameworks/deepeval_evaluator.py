"""
DeepEval Reasoning Evaluator - Clinical reasoning evaluation metrics.

This module provides DeepEval-based metrics for evaluating the quality
of clinical diagnostic reasoning in model outputs.

Metrics included:
- G-Eval Clinical Coherence: Is the diagnostic reasoning logically structured?
- G-Eval Correctness: Does reasoning align with reference reasoning?
- Hallucination: Does the model fabricate clinical facts?
- Answer Relevancy: Is the answer relevant to the clinical question?
"""
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm

try:
    from deepeval.metrics import GEval, HallucinationMetric, AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval import evaluate as deepeval_evaluate
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    print("Warning: DeepEval not installed. Install with: pip install deepeval")


# Clinical reasoning evaluation criteria
CLINICAL_COHERENCE_CRITERIA = """
Evaluate whether the diagnostic reasoning follows a logical clinical thought process.
Consider:
1. Are the key symptoms correctly identified from the patient presentation?
2. Is there appropriate consideration of differential diagnoses?
3. Does the conclusion logically follow from the clinical evidence presented?
4. Is the reasoning appropriate for the clinical context?
"""

CLINICAL_COHERENCE_STEPS = [
    "Identify the key symptoms and clinical findings mentioned in the reasoning",
    "Check if the reasoning considers relevant differential diagnoses",
    "Evaluate whether the conclusion follows logically from the evidence",
    "Assess if the reasoning demonstrates appropriate clinical knowledge"
]

CLINICAL_CORRECTNESS_CRITERIA = """
Compare the model's diagnostic reasoning against the reference clinical reasoning.
Evaluate:
1. Does the model identify the same key clinical features?
2. Does the model reach the same diagnostic conclusion?
3. Is the reasoning pathway similar to the reference?
4. Are there any significant omissions or errors?
"""

CLINICAL_CORRECTNESS_STEPS = [
    "Compare the key symptoms identified by the model vs reference",
    "Check if the diagnostic conclusion matches the reference",
    "Evaluate the similarity of the reasoning pathway",
    "Identify any critical omissions or factual errors"
]


class DeepEvalReasoningEvaluator:
    """
    Evaluator using DeepEval G-Eval for clinical reasoning assessment.
    
    Uses LLM-as-judge approach with customized clinical criteria
    to evaluate diagnostic reasoning quality.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        include_hallucination: bool = True,
        include_relevancy: bool = True,
        threshold: float = 0.5
    ):
        """
        Initialize the DeepEval reasoning evaluator.
        
        Args:
            model: OpenAI model for G-Eval (e.g., gpt-4o-mini, gpt-4)
            include_hallucination: Whether to include hallucination metric
            include_relevancy: Whether to include answer relevancy metric
            threshold: Minimum threshold for passing (0.0-1.0)
        """
        if not DEEPEVAL_AVAILABLE:
            raise ImportError(
                "DeepEval is not installed. Install with: pip install deepeval"
            )
        
        self.model = model
        self.threshold = threshold
        self.include_hallucination = include_hallucination
        self.include_relevancy = include_relevancy
        
        # Set model via environment variable (DeepEval's preferred method)
        os.environ["OPENAI_MODEL_NAME"] = model
        
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Initialize the G-Eval metrics with clinical criteria."""
        # Clinical Reasoning Coherence metric
        self.coherence_metric = GEval(
            name="Clinical Reasoning Coherence",
            criteria=CLINICAL_COHERENCE_CRITERIA,
            evaluation_steps=CLINICAL_COHERENCE_STEPS,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT
            ],
            model=self.model,
            threshold=self.threshold
        )
        
        # Clinical Reasoning Correctness metric (compares to reference)
        self.correctness_metric = GEval(
            name="Clinical Reasoning Correctness",
            criteria=CLINICAL_CORRECTNESS_CRITERIA,
            evaluation_steps=CLINICAL_CORRECTNESS_STEPS,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT
            ],
            model=self.model,
            threshold=self.threshold
        )
        
        # Optional: Hallucination metric
        if self.include_hallucination:
            self.hallucination_metric = HallucinationMetric(
                threshold=self.threshold,
                model=self.model
            )
        else:
            self.hallucination_metric = None
        
        # Optional: Answer Relevancy metric
        if self.include_relevancy:
            self.relevancy_metric = AnswerRelevancyMetric(
                threshold=self.threshold,
                model=self.model
            )
        else:
            self.relevancy_metric = None
    
    def _create_test_case(self, result: Dict[str, Any]) -> LLMTestCase:
        """
        Create a DeepEval test case from an inference result.
        
        Args:
            result: Inference result dictionary
            
        Returns:
            LLMTestCase for DeepEval
        """
        # Build expected output from ground truth and reference reasoning
        expected_output = f"Diagnosis: {result.get('ground_truth', '')}"
        if result.get('diagnostic_reasoning'):
            expected_output += f"\nReasoning: {result['diagnostic_reasoning']}"
        
        # Get contexts for hallucination check
        contexts = result.get("contexts", [])
        if not contexts:
            contexts = None
        
        return LLMTestCase(
            input=result["question"],
            actual_output=result["answer"],
            expected_output=expected_output,
            context=contexts,
            retrieval_context=contexts
        )
    
    def evaluate_case(
        self,
        result: Dict[str, Any],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a single case using all configured metrics.
        
        Args:
            result: Inference result dictionary
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary with metric scores for this case
        """
        test_case = self._create_test_case(result)
        
        scores = {}
        reasons = {}
        
        # Evaluate coherence
        try:
            self.coherence_metric.measure(test_case)
            scores["coherence"] = self.coherence_metric.score
            reasons["coherence"] = self.coherence_metric.reason
        except Exception as e:
            if verbose:
                print(f"  Warning: Coherence metric failed: {e}")
            scores["coherence"] = None
            reasons["coherence"] = str(e)
        
        # Evaluate correctness
        try:
            self.correctness_metric.measure(test_case)
            scores["correctness"] = self.correctness_metric.score
            reasons["correctness"] = self.correctness_metric.reason
        except Exception as e:
            if verbose:
                print(f"  Warning: Correctness metric failed: {e}")
            scores["correctness"] = None
            reasons["correctness"] = str(e)
        
        # Evaluate hallucination (if enabled and has contexts)
        if self.hallucination_metric and test_case.context:
            try:
                self.hallucination_metric.measure(test_case)
                # Hallucination score: higher = more hallucination, so invert for quality
                scores["factual_accuracy"] = 1.0 - self.hallucination_metric.score
                reasons["factual_accuracy"] = self.hallucination_metric.reason
            except Exception as e:
                if verbose:
                    print(f"  Warning: Hallucination metric failed: {e}")
                scores["factual_accuracy"] = None
                reasons["factual_accuracy"] = str(e)
        
        # Evaluate relevancy (if enabled)
        if self.relevancy_metric:
            try:
                self.relevancy_metric.measure(test_case)
                scores["relevancy"] = self.relevancy_metric.score
                reasons["relevancy"] = self.relevancy_metric.reason
            except Exception as e:
                if verbose:
                    print(f"  Warning: Relevancy metric failed: {e}")
                scores["relevancy"] = None
                reasons["relevancy"] = str(e)
        
        return {
            "case_id": result.get("case_id", "unknown"),
            "scores": scores,
            "reasons": reasons
        }
    
    def evaluate_results(
        self,
        results: List[Dict[str, Any]],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate all inference results using DeepEval metrics.
        
        Args:
            results: List of inference result dictionaries
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing:
            - average_scores: Per-metric average scores
            - per_case_scores: Per-case breakdown
            - metadata: Evaluation metadata
        """
        if verbose:
            print("\n" + "="*60)
            print("DEEPEVAL REASONING EVALUATION")
            print("="*60)
            print(f"Model: {self.model}")
            print(f"Metrics: coherence, correctness", end="")
            if self.include_hallucination:
                print(", factual_accuracy", end="")
            if self.include_relevancy:
                print(", relevancy", end="")
            print()
        
        # Filter out error cases
        valid_results = [
            r for r in results
            if not r.get("answer", "").startswith("ERROR")
        ]
        
        if verbose:
            print(f"Evaluating {len(valid_results)} cases...")
        
        # Evaluate each case
        per_case_scores = []
        for result in tqdm(valid_results, desc="DeepEval evaluation", disable=not verbose):
            case_result = self.evaluate_case(result, verbose=False)
            per_case_scores.append(case_result)
        
        # Compute average scores
        average_scores = {}
        metric_names = ["coherence", "correctness"]
        if self.include_hallucination:
            metric_names.append("factual_accuracy")
        if self.include_relevancy:
            metric_names.append("relevancy")
        
        for metric in metric_names:
            valid_scores = [
                case["scores"].get(metric)
                for case in per_case_scores
                if case["scores"].get(metric) is not None
            ]
            if valid_scores:
                average_scores[metric] = sum(valid_scores) / len(valid_scores)
            else:
                average_scores[metric] = None
        
        # Print summary
        if verbose:
            print("\n📊 DeepEval Reasoning Scores:")
            print("-" * 40)
            for metric, score in average_scores.items():
                if score is not None:
                    print(f"  {metric}: {score:.4f}")
                else:
                    print(f"  {metric}: N/A")
            print("-" * 40)
        
        return {
            "average_scores": average_scores,
            "per_case_scores": per_case_scores,
            "metadata": {
                "num_cases": len(valid_results),
                "model": self.model,
                "threshold": self.threshold,
                "metrics_used": metric_names,
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
            output_path: Optional path to save DeepEval results
            verbose: Whether to print progress
            
        Returns:
            DeepEval evaluation results
        """
        # Load results
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if verbose:
            print(f"📂 Loaded {len(results)} results from {results_path}")
        
        # Run evaluation
        deepeval_results = self.evaluate_results(results, verbose=verbose)
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(deepeval_results, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"💾 DeepEval results saved to {output_path}")
        
        return deepeval_results


def main():
    """Run DeepEval reasoning evaluation on inference results."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run DeepEval reasoning evaluation on inference results"
    )
    parser.add_argument(
        "results_path",
        help="Path to inference_results.json file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to save DeepEval results",
        default=None
    )
    parser.add_argument(
        "--model",
        help="OpenAI model for G-Eval",
        default="gpt-4o-mini"
    )
    parser.add_argument(
        "--no-hallucination",
        action="store_true",
        help="Disable hallucination metric"
    )
    parser.add_argument(
        "--no-relevancy",
        action="store_true",
        help="Disable relevancy metric"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum threshold for passing (0.0-1.0)"
    )
    
    args = parser.parse_args()
    
    evaluator = DeepEvalReasoningEvaluator(
        model=args.model,
        include_hallucination=not args.no_hallucination,
        include_relevancy=not args.no_relevancy,
        threshold=args.threshold
    )
    
    evaluator.evaluate_from_file(
        results_path=args.results_path,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
