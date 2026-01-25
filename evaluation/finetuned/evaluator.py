"""
Fine-tuned Model Evaluator - Evaluates the LoRA fine-tuned LLM without RAG retrieval.

This evaluator loads the fine-tuned LoRA adapter and runs inference
without any retrieved context, to measure the improvement from fine-tuning alone.
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
    FINETUNED_RESULTS_FILE,
    FINETUNED_CHECKPOINT_FILE,
    DEVICE
)
from rag.generator import Generator
from rag.config import LORA_BASE_MODEL, LORA_ADAPTER_PATH


class FinetunedModelEvaluator(BaseEvaluator):
    """Evaluator for fine-tuned LoRA model without RAG retrieval."""
    
    MODEL_TYPE = "finetuned"
    
    def __init__(self, device: str = DEVICE, lora_path: str = None):
        """
        Initialize the fine-tuned model evaluator.
        
        Args:
            device: Device to run model on ("auto", "cuda", "mps", "cpu")
            lora_path: Path to LoRA adapter (defaults to config value)
        """
        super().__init__()
        self.lora_path = lora_path or LORA_ADAPTER_PATH
        
        print(f"🚀 Initializing fine-tuned model...")
        print(f"   Base model: {LORA_BASE_MODEL}")
        print(f"   LoRA adapter: {self.lora_path}")
        
        # Determine device
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.generator = Generator(device=device, use_lora=True, lora_path=self.lora_path)
        self.model_name = f"{LORA_BASE_MODEL} + LoRA"
        print("✅ Fine-tuned model initialized successfully")
    
    def run_inference(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run fine-tuned model inference on a single case (no RAG context).
        
        Args:
            case: Evaluation case with question and ground_truth
            
        Returns:
            Dictionary with inference results
        """
        question = case["question"]
        
        print(f"\n📋 Processing case: {case['case_id']}")
        print(f"❓ Question: {question[:100]}...")
        
        try:
            # Create prompt for fine-tuned model using the instruction format it was trained on
            finetuned_prompt = """### Instruction:
You are an expert clinical diagnostician. Analyze the following patient case and provide the most likely diagnosis.

PATIENT CASE:
{question}

Based on the symptoms, travel history, and clinical presentation described above, provide the most likely diagnosis. Respond with ONLY the disease name.

### Response:
"""
            
            # Generate the diagnosis using the fine-tuned generator
            answer = self.generator.generate(
                context="",  # No context for standalone fine-tuned model
                question=question,
                custom_prompt=finetuned_prompt
            )
            
            # Clean up the answer
            answer = clean_diagnosis_output(answer)
            
            print(f"💡 Generated diagnosis: {answer}")
            print(f"✅ Ground truth: {case['ground_truth']}")
            
            result = {
                "case_id": case["case_id"],
                "question": question,
                "contexts": [],  # Empty - no retrieval for fine-tuned model alone
                "answer": answer,
                "ground_truth": case["ground_truth"],
                "metadata": {
                    "model_type": self.MODEL_TYPE,
                    "model_name": self.model_name,
                    "lora_path": self.lora_path,
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
                    "lora_path": self.lora_path,
                    "num_contexts": 0,
                    "error": str(e)
                },
                "diagnostic_reasoning": case.get("diagnostic_reasoning", "")
            }


def main():
    """Main execution function for fine-tuned model evaluation."""
    evaluator = FinetunedModelEvaluator()
    evaluator.evaluate_all(
        input_path=TEST_CASES_PATH,
        output_path=FINETUNED_RESULTS_FILE,
        checkpoint_path=FINETUNED_CHECKPOINT_FILE
    )


if __name__ == "__main__":
    main()
