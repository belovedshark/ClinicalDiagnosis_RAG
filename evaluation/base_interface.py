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
    save_checkpoint
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
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print evaluation summary."""
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"Model type: {self.MODEL_TYPE}")
        print(f"Total cases processed: {len(results)}")
        
        # Calculate accuracy
        correct = sum(1 for r in results 
                     if r.get("answer", "").lower().strip() == 
                        r.get("ground_truth", "").lower().strip())
        accuracy = correct / len(results) * 100 if results else 0
        
        print(f"\n📊 Summary:")
        print(f"  ✅ Correct: {correct}/{len(results)} ({accuracy:.1f}%)")
        
        # Count errors
        errors = sum(1 for r in results if r.get("answer", "").startswith("ERROR"))
        if errors:
            print(f"  ❌ Errors: {errors}")
