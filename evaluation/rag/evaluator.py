"""
RAG Evaluator - Evaluates the RAG (Retrieval-Augmented Generation) system.

This evaluator retrieves relevant context from the vector database
and uses it to augment the LLM's diagnosis.
"""
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.base_interface import BaseEvaluator
from evaluation.config import (
    TOP_K_RETRIEVAL,
    TEST_CASES_PATH,
    RAG_RESULTS_FILE,
    RAG_CHECKPOINT_FILE
)
from rag.pipeline import RAGPipeline


class RAGEvaluator(BaseEvaluator):
    """Evaluator for RAG (Retrieval-Augmented Generation) system."""
    
    MODEL_TYPE = "rag"
    
    def __init__(self, top_k: int = TOP_K_RETRIEVAL):
        """
        Initialize the RAG evaluator.
        
        Args:
            top_k: Number of contexts to retrieve for each query
        """
        super().__init__()
        self.top_k = top_k
        print("🚀 Initializing RAG pipeline...")
        self.rag_pipeline = RAGPipeline()
        print("✅ RAG pipeline initialized successfully")
    
    def run_inference(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run RAG inference on a single case.
        
        Args:
            case: Evaluation case with question and ground_truth
            
        Returns:
            Dictionary with inference results
        """
        question = case["question"]
        
        print(f"\n📋 Processing case: {case['case_id']}")
        print(f"❓ Question: {question[:100]}...")
        
        try:
            # Get contexts with metadata using retriever
            retriever = self.rag_pipeline.retriever
            query_emb = retriever.embed_query(query_text=question)
            
            # Retrieve with more details
            results = retriever.client.query_points(
                collection_name=retriever.collection,
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
            
            # Generate answer using the full RAG pipeline
            context_text = "\n\n".join(contexts)
            
            # Create diagnosis prompt - relies on retrieved context for reasoning
            custom_prompt_template = """You are an expert clinical diagnostician. Based on the reference literature provided, analyze the patient case and provide the most likely diagnosis.

REFERENCE LITERATURE:
{context}

PATIENT CASE:
{question}

Based on the symptoms described and the reference literature above, what is the most likely diagnosis?

Respond with ONLY the disease name. No explanations.

DIAGNOSIS:"""
            
            # Generate the diagnosis
            answer = self.rag_pipeline.generator.generate(
                context_text, 
                question,
                custom_prompt=custom_prompt_template
            )
            
            # Clean up the answer
            answer = answer.strip()
            if len(answer) > 100:
                answer = answer.split('\n')[0].strip()
                if '.' in answer:
                    answer = answer.split('.')[0].strip()
            
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
                    "similarity_scores": [],
                    "source_documents": [],
                    "num_contexts": 0,
                    "error": str(e)
                },
                "diagnostic_reasoning": case.get("diagnostic_reasoning", "")
            }


def main():
    """Main execution function for RAG evaluation."""
    evaluator = RAGEvaluator()
    evaluator.evaluate_all(
        input_path=TEST_CASES_PATH,
        output_path=RAG_RESULTS_FILE,
        checkpoint_path=RAG_CHECKPOINT_FILE
    )


if __name__ == "__main__":
    main()
