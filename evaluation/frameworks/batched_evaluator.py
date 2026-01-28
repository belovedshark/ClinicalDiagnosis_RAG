"""
Batched LLM Evaluator - Optimized 2-step evaluation.

Step 1: Extract claims from answer (for RAGAS-style faithfulness)
Step 2: Verify claims + compute all other metrics in single call

This maintains RAGAS faithfulness methodology while being fast.
Total: 2 API calls per case.
"""
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not installed. Install with: pip install openai")


# Step 1: Extract claims (RAGAS faithfulness methodology)
EXTRACT_CLAIMS_PROMPT = """Given a question and an answer, extract all factual claims/statements from the answer.

**Question:** {question}

**Answer:** {answer}

Extract each distinct factual claim as a separate item. Include diagnostic claims, symptom mentions, and reasoning statements.

Respond with ONLY a JSON array of strings:
```json
["claim 1", "claim 2", "claim 3"]
```
"""


# Step 2: Verify claims + all other metrics
EVALUATION_PROMPT = """You are an expert medical evaluator. Evaluate this clinical diagnosis case.

## Case Information

**Question (Patient Case):**
{question}

**Retrieved Contexts:**
{contexts}

**Model's Answer:**
{answer}

**Ground Truth Diagnosis:**
{ground_truth}

**Reference Diagnostic Reasoning:**
{diagnostic_reasoning}

**Claims extracted from the answer:**
{claims}

## Evaluation Tasks

### 1. FAITHFULNESS (RAGAS methodology)
For each claim listed above, determine if it is supported by the retrieved contexts.
Count how many claims are supported vs total claims.
Score = supported_claims / total_claims (0.0 to 1.0)

### 2. ANSWER RELEVANCY (RAGAS methodology)  
Does the answer directly and appropriately address the clinical question?
- 1.0 = Perfectly relevant diagnosis for the symptoms
- 0.0 = Completely irrelevant to the question

### 3. CONTEXT PRECISION (RAGAS methodology)
How much of the retrieved context is actually relevant to answering this question?
- 1.0 = All context is highly relevant
- 0.0 = Context is mostly irrelevant

### 4. CONTEXT RECALL (RAGAS methodology)
Does the context contain the information needed to reach the correct diagnosis?
- 1.0 = Context fully supports the ground truth diagnosis
- 0.0 = Context missing critical information

### 5. REASONING COHERENCE (DeepEval G-Eval methodology)
Is the diagnostic reasoning logical and clinically sound?
- 1.0 = Clear, logical clinical reasoning
- 0.0 = Illogical or inappropriate reasoning

### 6. CORRECTNESS (DeepEval methodology)
Does the answer match the ground truth diagnosis?
- 1.0 = Exact or clinically equivalent match
- 0.5 = Partially correct (related condition)
- 0.0 = Completely wrong diagnosis

## Response Format

Respond with ONLY valid JSON:
```json
{{
  "faithfulness": <score>,
  "faithfulness_detail": "<X of Y claims supported>",
  "answer_relevancy": <score>,
  "context_precision": <score>,
  "context_recall": <score>,
  "reasoning_coherence": <score>,
  "correctness": <score>,
  "explanation": "<brief overall assessment>"
}}
```
"""


class BatchedLLMEvaluator:
    """
    Optimized 2-step evaluator combining RAGAS + DeepEval methodologies.
    
    Step 1: Extract claims (RAGAS faithfulness)
    Step 2: Verify claims + compute all metrics
    
    Total: 2 API calls per case (much faster than running frameworks separately)
    """
    
    # Metrics grouped by framework
    RAGAS_METRICS = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ]
    
    DEEPEVAL_METRICS = [
        "reasoning_coherence",
        "correctness"
    ]
    
    METRICS = RAGAS_METRICS + DEEPEVAL_METRICS
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 2
    ):
        """
        Initialize the batched evaluator.
        
        Args:
            model: OpenAI model to use
            temperature: Temperature for evaluation (0 for consistency)
            max_retries: Number of retries on parse failure
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not installed. Install with: pip install openai")
        
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = OpenAI()
    
    def _format_contexts(self, contexts: List[str], max_length: int = 3000) -> str:
        """Format contexts for the prompt, truncating if too long."""
        if not contexts:
            return "(No contexts retrieved)"
        
        formatted = []
        total_length = 0
        for i, ctx in enumerate(contexts, 1):
            ctx_text = f"[Context {i}]: {ctx[:600]}..."
            if total_length + len(ctx_text) > max_length:
                formatted.append(f"... ({len(contexts) - i + 1} more contexts truncated)")
                break
            formatted.append(ctx_text)
            total_length += len(ctx_text)
        
        return "\n\n".join(formatted)
    
    def _extract_claims(self, question: str, answer: str) -> List[str]:
        """Step 1: Extract claims from the answer (RAGAS methodology)."""
        prompt = EXTRACT_CLAIMS_PROMPT.format(
            question=question,
            answer=answer
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON array
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0]
            else:
                json_str = response_text
            
            claims = json.loads(json_str.strip())
            return claims if isinstance(claims, list) else [str(claims)]
            
        except Exception as e:
            # Fallback: treat the whole answer as one claim
            return [answer]
    
    def _evaluate_with_claims(
        self,
        result: Dict[str, Any],
        claims: List[str]
    ) -> Dict[str, Any]:
        """Step 2: Verify claims and compute all metrics."""
        prompt = EVALUATION_PROMPT.format(
            question=result.get("question", ""),
            contexts=self._format_contexts(result.get("contexts", [])),
            answer=result.get("answer", ""),
            ground_truth=result.get("ground_truth", ""),
            diagnostic_reasoning=result.get("diagnostic_reasoning", ""),
            claims=json.dumps(claims, indent=2)
        )
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=600
                )
                
                response_text = response.choices[0].message.content
                
                # Parse JSON
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0]
                else:
                    json_str = response_text
                
                scores = json.loads(json_str.strip())
                
                # Validate and clamp scores
                for metric in self.METRICS:
                    if metric in scores:
                        scores[metric] = max(0.0, min(1.0, float(scores[metric])))
                
                return scores
                
            except Exception as e:
                if attempt == self.max_retries:
                    return {"error": str(e)}
        
        return {"error": "Failed to parse response"}
    
    def evaluate_case(
        self,
        result: Dict[str, Any],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a single case using 2-step approach.
        
        Args:
            result: Inference result dictionary
            verbose: Whether to print details
            
        Returns:
            Dictionary with all metric scores
        """
        # Step 1: Extract claims
        claims = self._extract_claims(
            result.get("question", ""),
            result.get("answer", "")
        )
        
        if verbose:
            print(f"  Extracted {len(claims)} claims")
        
        # Step 2: Verify claims + compute all metrics
        scores = self._evaluate_with_claims(result, claims)
        
        if "error" in scores:
            return {
                "case_id": result.get("case_id", "unknown"),
                "ragas": {k: None for k in self.RAGAS_METRICS},
                "deepeval": {k: None for k in self.DEEPEVAL_METRICS},
                "claims_extracted": len(claims),
                "error": scores["error"]
            }
        
        return {
            "case_id": result.get("case_id", "unknown"),
            "ragas": {
                "faithfulness": scores.get("faithfulness"),
                "faithfulness_detail": scores.get("faithfulness_detail", ""),
                "answer_relevancy": scores.get("answer_relevancy"),
                "context_precision": scores.get("context_precision"),
                "context_recall": scores.get("context_recall")
            },
            "deepeval": {
                "reasoning_coherence": scores.get("reasoning_coherence"),
                "correctness": scores.get("correctness")
            },
            "claims_extracted": len(claims),
            "explanation": scores.get("explanation", "")
        }
    
    def evaluate_results(
        self,
        results: List[Dict[str, Any]],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate all inference results.
        
        Args:
            results: List of inference result dictionaries
            verbose: Whether to print progress
            
        Returns:
            Dictionary with average scores and per-case breakdown
        """
        if verbose:
            print("\n" + "="*60)
            print("BATCHED LLM EVALUATION (RAGAS + DeepEval)")
            print("="*60)
            print(f"Model: {self.model}")
            print(f"Method: 2-step (extract claims → verify + evaluate)")
            print(f"Metrics: {', '.join(self.METRICS)}")
        
        # Filter out error cases
        valid_results = [
            r for r in results
            if not r.get("answer", "").startswith("ERROR")
        ]
        
        if verbose:
            print(f"Evaluating {len(valid_results)} cases (2 API calls per case)...")
        
        # Evaluate each case
        per_case_scores = []
        for result in tqdm(valid_results, desc="Evaluating", disable=not verbose):
            case_result = self.evaluate_case(result, verbose=False)
            per_case_scores.append(case_result)
        
        # Compute averages grouped by framework
        ragas_averages = {}
        for metric in self.RAGAS_METRICS:
            valid_scores = [
                case["ragas"].get(metric)
                for case in per_case_scores
                if case["ragas"].get(metric) is not None
            ]
            if valid_scores:
                ragas_averages[metric] = sum(valid_scores) / len(valid_scores)
            else:
                ragas_averages[metric] = None
        
        deepeval_averages = {}
        for metric in self.DEEPEVAL_METRICS:
            valid_scores = [
                case["deepeval"].get(metric)
                for case in per_case_scores
                if case["deepeval"].get(metric) is not None
            ]
            if valid_scores:
                deepeval_averages[metric] = sum(valid_scores) / len(valid_scores)
            else:
                deepeval_averages[metric] = None
        
        # Print summary
        if verbose:
            print("\n" + "="*60)
            print("📊 EVALUATION RESULTS")
            print("="*60)
            
            print("\n┌─ RAGAS Framework ─────────────────────────────────────┐")
            for metric in self.RAGAS_METRICS:
                score = ragas_averages.get(metric)
                if score is not None:
                    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                    print(f"│  {metric:20s}: {score:.3f} [{bar}] │")
            print("└────────────────────────────────────────────────────────┘")
            
            print("\n┌─ DeepEval Framework ──────────────────────────────────┐")
            for metric in self.DEEPEVAL_METRICS:
                score = deepeval_averages.get(metric)
                if score is not None:
                    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                    print(f"│  {metric:20s}: {score:.3f} [{bar}] │")
            print("└────────────────────────────────────────────────────────┘")
            print("="*60)
        
        return {
            "ragas": {
                "average_scores": ragas_averages,
                "metrics": self.RAGAS_METRICS,
                "description": "Retrieval-Augmented Generation Assessment"
            },
            "deepeval": {
                "average_scores": deepeval_averages,
                "metrics": self.DEEPEVAL_METRICS,
                "description": "LLM Evaluation with G-Eval methodology"
            },
            "per_case_scores": per_case_scores,
            "metadata": {
                "num_cases": len(valid_results),
                "model": self.model,
                "method": "2-step: extract_claims + verify_and_evaluate",
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
            output_path: Optional path to save results
            verbose: Whether to print progress
            
        Returns:
            Evaluation results
        """
        # Load results
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if verbose:
            print(f"📂 Loaded {len(results)} results from {results_path}")
        
        # Run evaluation
        eval_results = self.evaluate_results(results, verbose=verbose)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"💾 Results saved to {output_path}")
        
        return eval_results


def main():
    """Run batched LLM evaluation on inference results."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run optimized 2-step evaluation (RAGAS + DeepEval metrics)"
    )
    parser.add_argument(
        "results_path",
        help="Path to inference_results.json file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to save evaluation results",
        default=None
    )
    parser.add_argument(
        "--model",
        help="OpenAI model to use",
        default="gpt-4o-mini"
    )
    
    args = parser.parse_args()
    
    evaluator = BatchedLLMEvaluator(model=args.model)
    evaluator.evaluate_from_file(
        results_path=args.results_path,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
