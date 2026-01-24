"""
Base Model Evaluator - Evaluates the LLM without RAG retrieval.

This evaluator calls the LLM directly with only the patient case,
without any retrieved context, to establish a baseline performance.
"""
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.base_interface import BaseEvaluator
from evaluation.utils import clean_diagnosis_output
from evaluation.config import (
    TEST_CASES_PATH,
    BASE_MODEL_RESULTS_FILE,
    BASE_MODEL_CHECKPOINT_FILE,
    DEVICE
)
from rag.generator import Generator
from rag.config import LLM_MODEL


class BaseModelEvaluator(BaseEvaluator):
    """Evaluator for base LLM model without RAG retrieval."""
    
    MODEL_TYPE = "base_model"
    
    def __init__(self, device: str = DEVICE):
        """
        Initialize the base model evaluator.
        
        Args:
            device: Device to run model on ("auto", "cuda", "mps", "cpu")
        """
        super().__init__()
        print(f"🚀 Initializing base model ({LLM_MODEL})...")
        
        # Determine device
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.generator = Generator(device=device)
        self.model_name = LLM_MODEL
        print("✅ Base model initialized successfully")
    
    def run_inference(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run base model inference on a single case (no RAG context).
        
        Args:
            case: Evaluation case with question and ground_truth
            
        Returns:
            Dictionary with inference results
        """
        question = case["question"]
        
        print(f"\n📋 Processing case: {case['case_id']}")
        print(f"❓ Question: {question[:100]}...")
        
        try:
            # Create prompt for base model (no context)
            base_model_prompt = """You are an expert clinical diagnostician. Analyze the patient case and provide a diagnosis.

PATIENT CASE:
{question}

Based on the symptoms, travel history, and clinical presentation described above, provide the most likely diagnosis.

RESPONSE FORMAT:
You must respond with ONLY the disease name. No explanations, no formatting, no punctuation.

DIAGNOSIS:"""
            
            # Generate the diagnosis using the generator
            # Note: We pass empty context since this is base model evaluation
            answer = self.generator.generate(
                context="",  # No context for base model
                question=question,
                custom_prompt=base_model_prompt
            )
            
            # Clean up the answer
            answer = clean_diagnosis_output(answer)
            
            print(f"💡 Generated diagnosis: {answer}")
            print(f"✅ Ground truth: {case['ground_truth']}")
            
            result = {
                "case_id": case["case_id"],
                "question": question,
                "contexts": [],  # Empty - no retrieval for base model
                "answer": answer,
                "ground_truth": case["ground_truth"],
                "metadata": {
                    "model_type": self.MODEL_TYPE,
                    "model_name": self.model_name,
                    "num_contexts": 0
                },
                "diagnostic_reasoning": case.get("diagnostic_reasoning", "")
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing case {case['case_id']}: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "case_id": case["case_id"],
                "question": question,
                "contexts": [],
                "answer": f"ERROR: {str(e)}",
                "ground_truth": case["ground_truth"],
                "metadata": {
                    "model_type": self.MODEL_TYPE,
                    "model_name": self.model_name,
                    "num_contexts": 0,
                    "error": str(e)
                },
                "diagnostic_reasoning": case.get("diagnostic_reasoning", "")
            }


def main():
    """Main execution function for base model evaluation."""
    evaluator = BaseModelEvaluator()
    evaluator.evaluate_all(
        input_path=TEST_CASES_PATH,
        output_path=BASE_MODEL_RESULTS_FILE,
        checkpoint_path=BASE_MODEL_CHECKPOINT_FILE
    )


if __name__ == "__main__":
    main()
