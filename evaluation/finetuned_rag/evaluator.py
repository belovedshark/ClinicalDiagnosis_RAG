"""
Fine-tuned + RAG Evaluator - Evaluates the LoRA fine-tuned LLM with RAG retrieval.

This evaluator combines the fine-tuned LoRA model with RAG retrieval
to measure the combined effect of fine-tuning and retrieval augmentation.
"""
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.base_interface import BaseEvaluator
from evaluation.utils import clean_diagnosis_output
from evaluation.config import (
    TOP_K_RETRIEVAL,
    TEST_CASES_PATH,
    FINETUNED_RAG_RESULTS_FILE,
    FINETUNED_RAG_CHECKPOINT_FILE,
    DEVICE
)
from rag.pipeline import RAGPipeline
from rag.config import LORA_BASE_MODEL, LORA_ADAPTER_PATH


class FinetunedRAGEvaluator(BaseEvaluator):
    """Evaluator for fine-tuned LoRA model with RAG retrieval."""
    
    MODEL_TYPE = "finetuned_rag"
    
    def __init__(self, device: str = DEVICE, top_k: int = TOP_K_RETRIEVAL, lora_path: str = None):
        """
        Initialize the fine-tuned + RAG evaluator.
        
        Args:
            device: Device to run model on ("auto", "cuda", "mps", "cpu")
            top_k: Number of contexts to retrieve for each query
            lora_path: Path to LoRA adapter (defaults to config value)
        """
        super().__init__()
        self.top_k = top_k
        self.lora_path = lora_path or LORA_ADAPTER_PATH
        
        print(f"🚀 Initializing fine-tuned model + RAG pipeline...")
        print(f"   Base model: {LORA_BASE_MODEL}")
        print(f"   LoRA adapter: {self.lora_path}")
        print(f"   Top-k retrieval: {top_k}")
        
        # Initialize RAG pipeline with fine-tuned model
        self.rag_pipeline = RAGPipeline(use_lora=True, lora_path=self.lora_path)
        self.model_name = f"{LORA_BASE_MODEL} + LoRA + RAG"
        print("✅ Fine-tuned + RAG pipeline initialized successfully")
    
    def run_inference(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run fine-tuned + RAG inference on a single case.
        
        Args:
            case: Evaluation case with question and ground_truth
            
        Returns:
            Dictionary with inference results
        """
        question = case["question"]
        
        print(f"\n📋 Processing case: {case['case_id']}")
        print(f"❓ Question: {question[:100]}...")
        
        try:
            # Retrieve relevant contexts
            query_emb = self.rag_pipeline.retriever.embed_query(query_text=question)
            
            results = self.rag_pipeline.retriever.client.query_points(
                collection_name=self.rag_pipeline.retriever.collection,
                query=query_emb.tolist(),
                using='text',
                limit=self.top_k,
                with_payload=True,
            )
            
            points = results.points if hasattr(results, 'points') else results
            
            contexts = []
            similarity_scores = []
            source_documents = []
            
            for p in points:
                if isinstance(p, dict):
                    payload = p.get('payload', {}) or {}
                    score = p.get('score', 0.0)
                else:
                    payload = getattr(p, 'payload', {}) or {}
                    score = getattr(p, 'score', 0.0)
                
                text = payload.get('text', '')
                source = payload.get('source', 'unknown')
                
                contexts.append(text)
                similarity_scores.append(float(score))
                source_documents.append(source)
            
            # Combine contexts
            context_text = "\n\n".join(contexts)
            
            # Create prompt for fine-tuned model with RAG context
            # Using the instruction format the model was trained on
            finetuned_rag_prompt = """### Instruction:
You are an expert clinical diagnostician. Based on the reference literature provided, analyze the patient case and provide the most likely diagnosis.

REFERENCE LITERATURE:
{context}

PATIENT CASE:
{question}

Based on the symptoms described and the reference literature above, what is the most likely diagnosis? Respond with ONLY the disease name.

### Response:
"""
            
            # Generate the diagnosis
            answer = self.rag_pipeline.generator.generate(
                context=context_text,
                question=question,
                custom_prompt=finetuned_rag_prompt
            )
            
            # Clean up the answer
            answer = clean_diagnosis_output(answer)
            
            print(f"🔍 Retrieved {len(contexts)} contexts")
            print(f"💡 Generated diagnosis: {answer}")
            print(f"✅ Ground truth: {case['ground_truth']}")
            
            result = {
                "case_id": case["case_id"],
                "question": question,
                "contexts": contexts,
                "answer": answer,
                "ground_truth": case["ground_truth"],
                "metadata": {
                    "model_type": self.MODEL_TYPE,
                    "model_name": self.model_name,
                    "lora_path": self.lora_path,
                    "similarity_scores": similarity_scores,
                    "source_documents": source_documents,
                    "num_contexts": len(contexts)
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
                    "similarity_scores": [],
                    "source_documents": [],
                    "num_contexts": 0,
                    "error": str(e)
                },
                "diagnostic_reasoning": case.get("diagnostic_reasoning", "")
            }


def main():
    """Main execution function for fine-tuned + RAG evaluation."""
    evaluator = FinetunedRAGEvaluator()
    evaluator.evaluate_all(
        input_path=TEST_CASES_PATH,
        output_path=FINETUNED_RAG_RESULTS_FILE,
        checkpoint_path=FINETUNED_RAG_CHECKPOINT_FILE
    )


if __name__ == "__main__":
    main()
