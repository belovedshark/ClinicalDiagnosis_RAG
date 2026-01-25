"""
Base interface for all model evaluators.

All evaluators (RAG, Base Model, Fine-tuned, Fine-tuned + RAG) must implement
this interface to ensure consistent input/output formats.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
from tqdm import tqdm

from .utils import (
    load_jsonl,
    prepare_evaluation_cases,
    validate_case,
    save_results,
    load_checkpoint,
    save_checkpoint,
    is_diagnosis_match,
    normalize_diagnosis
)
from .config import SAVE_INTERVAL


class BaseEvaluator(ABC):
    """Abstract base class for all model evaluators.
    
    All evaluators must implement the run_inference method to process
    individual cases. The evaluate_all method provides common logic
    for batch processing with checkpointing.
    """
    
    # Subclasses should override this
    MODEL_TYPE: str = "base"
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    @abstractmethod
    def run_inference(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on a single case.
        
        Args:
            case: Dictionary containing:
                - case_id: Unique identifier for the case
                - question: The clinical case prompt
                - ground_truth: Expected diagnosis
                - diagnostic_reasoning: Reference reasoning
                
        Returns:
            Dictionary containing:
                - case_id: Same as input
                - question: Same as input
                - contexts: List of retrieved contexts (empty for non-RAG models)
                - answer: Model's diagnosis
                - ground_truth: Same as input
                - metadata: Dict with model_type, num_contexts, etc.
                - diagnostic_reasoning: Same as input
        """
        pass
    
    def evaluate_all(
        self,
        input_path: str,
        output_path: str,
        checkpoint_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Run inference on all cases and save results.
        
        Args:
            input_path: Path to input JSONL file with test cases
            output_path: Path to save output JSON results
            checkpoint_path: Optional path for checkpoint file
            
        Returns:
            List of inference results
        """
        print("="*80)
        print(f"{self.MODEL_TYPE.upper()} EVALUATION")
        print("="*80)
        
        # Load evaluation dataset
        print(f"\n📂 Loading evaluation dataset from: {input_path}")
        raw_data = load_jsonl(input_path)
        print(f"✅ Loaded {len(raw_data)} cases")
        
        # Prepare cases
        print("\n🔧 Preparing evaluation cases...")
        cases = prepare_evaluation_cases(raw_data)
        print(f"✅ Prepared {len(cases)} cases for evaluation")
        
        # Check for existing checkpoint
        resume_from = 0
        existing_results = []
        
        if checkpoint_path:
            checkpoint = load_checkpoint(checkpoint_path)
            if checkpoint:
                resume_from = checkpoint.get("last_processed_index", -1) + 1
                existing_results = checkpoint.get("results", [])
                print(f"\n♻️  Resuming from checkpoint at index {resume_from}")
                print(f"   Already processed: {len(existing_results)} cases")
        
        # Run inference
        print(f"\n🚀 Starting {self.MODEL_TYPE} inference...")
        results = []
        
        for idx, case in enumerate(tqdm(cases[resume_from:], 
                                       desc=f"Running {self.MODEL_TYPE} inference",
                                       initial=resume_from,
                                       total=len(cases))):
            actual_idx = resume_from + idx
            
            # Validate case
            if not validate_case(case):
                print(f"⚠️  Skipping invalid case at index {actual_idx}")
                continue
            
            # Run inference
            result = self.run_inference(case)
            results.append(result)
            
            # Save checkpoint periodically
            if checkpoint_path and (actual_idx + 1) % SAVE_INTERVAL == 0:
                checkpoint_data = {
                    "last_processed_index": actual_idx,
                    "results": existing_results + results,
                    "timestamp": time.time(),
                    "model_type": self.MODEL_TYPE
                }
                save_checkpoint(checkpoint_data, checkpoint_path)
                print(f"💾 Checkpoint saved at index {actual_idx}")
        
        # Combine with existing results
        all_results = existing_results + results
        
        # Save final results
        print(f"\n💾 Saving final results...")
        save_results(all_results, output_path)
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(
        self, 
        results: List[Dict[str, Any]], 
        use_semantic: bool = True,
        semantic_threshold: float = 0.85
    ):
        """
        Print detailed evaluation summary with hybrid matching.
        
        Args:
            results: List of inference results
            use_semantic: Whether to use semantic similarity matching
            semantic_threshold: Threshold for semantic similarity
        """
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"Model type: {self.MODEL_TYPE}")
        print(f"Total cases processed: {len(results)}")
        
        if not results:
            print("No results to evaluate.")
            return
        
        # Compute detailed metrics using hybrid matching
        exact_matches = 0
        alias_matches = 0
        semantic_matches = 0
        no_matches = 0
        errors = 0
        
        mismatched_cases = []
        semantic_match_details = []
        
        for r in results:
            answer = r.get("answer", "")
            ground_truth = r.get("ground_truth", "")
            
            # Skip error cases
            if answer.startswith("ERROR"):
                errors += 1
                continue
            
            # Use hybrid matching
            is_match, match_type, confidence = is_diagnosis_match(
                answer, 
                ground_truth,
                use_semantic=use_semantic,
                semantic_threshold=semantic_threshold
            )
            
            if is_match:
                if match_type == "exact":
                    exact_matches += 1
                elif match_type == "alias":
                    alias_matches += 1
                elif match_type == "semantic":
                    semantic_matches += 1
                    semantic_match_details.append({
                        "case_id": r.get("case_id"),
                        "answer": answer,
                        "ground_truth": ground_truth,
                        "similarity": confidence
                    })
            else:
                no_matches += 1
                mismatched_cases.append({
                    "case_id": r.get("case_id"),
                    "answer": answer,
                    "ground_truth": ground_truth,
                    "answer_normalized": normalize_diagnosis(answer),
                    "truth_normalized": normalize_diagnosis(ground_truth),
                    "similarity": confidence
                })
        
        total_valid = len(results) - errors
        total_correct = exact_matches + alias_matches + semantic_matches
        accuracy = total_correct / total_valid * 100 if total_valid else 0
        
        # Print summary
        print(f"\n📊 Evaluation Summary (Hybrid Matching):")
        print(f"  {'─'*50}")
        print(f"  Total valid cases: {total_valid}")
        print(f"  ✅ Total correct: {total_correct}/{total_valid} ({accuracy:.1f}%)")
        print(f"  {'─'*50}")
        print(f"  Match breakdown:")
        if total_valid > 0:
            print(f"    • Exact matches:    {exact_matches:3d} ({exact_matches/total_valid*100:.1f}%)")
            print(f"    • Alias matches:    {alias_matches:3d} ({alias_matches/total_valid*100:.1f}%)")
            if use_semantic:
                print(f"    • Semantic matches: {semantic_matches:3d} ({semantic_matches/total_valid*100:.1f}%) [threshold={semantic_threshold}]")
            print(f"    • No match:         {no_matches:3d} ({no_matches/total_valid*100:.1f}%)")
        else:
            print(f"    • No valid cases to evaluate (all cases had errors)")
        if errors:
            print(f"  ❌ Errors: {errors}")
        
        # Show semantic match details
        if semantic_match_details:
            print(f"\n🔍 Semantic Matches (accepted via similarity):")
            for detail in semantic_match_details[:5]:  # Show first 5
                print(f"    [{detail['case_id']}] \"{detail['answer']}\" ≈ \"{detail['ground_truth']}\" (sim={detail['similarity']:.3f})")
            if len(semantic_match_details) > 5:
                print(f"    ... and {len(semantic_match_details) - 5} more")
        
        # Show mismatched cases
        if mismatched_cases:
            print(f"\n❌ Mismatched Cases ({len(mismatched_cases)} total):")
            for case in mismatched_cases[:5]:  # Show first 5
                print(f"    [{case['case_id']}]")
                print(f"      Answer:       \"{case['answer']}\" → normalized: \"{case['answer_normalized']}\"")
                print(f"      Ground truth: \"{case['ground_truth']}\" → normalized: \"{case['truth_normalized']}\"")
                print(f"      Similarity:   {case['similarity']:.3f}")
            if len(mismatched_cases) > 5:
                print(f"    ... and {len(mismatched_cases) - 5} more")
        
        print(f"\n{'='*80}")
