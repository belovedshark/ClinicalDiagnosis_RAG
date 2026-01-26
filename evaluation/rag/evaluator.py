"""
RAG Evaluator - Evaluates the RAG (Retrieval-Augmented Generation) system.

This evaluator retrieves relevant context from the vector database
and uses it to augment the LLM's diagnosis.

Enhanced features:
- Hybrid search (semantic + BM25 keyword matching)
- Cross-encoder reranking for better context relevance
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
    
    def __init__(self, 
                 top_k: int = TOP_K_RETRIEVAL,
                 use_hybrid_search: bool = True,
                 use_reranking: bool = True,
                 semantic_weight: float = 0.7,
                 bm25_weight: float = 0.3):
        """
        Initialize the RAG evaluator.
        
        Args:
            top_k: Number of contexts to retrieve for each query
            use_hybrid_search: Enable BM25 + semantic hybrid search
            use_reranking: Enable cross-encoder reranking
            semantic_weight: Weight for semantic similarity (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
        """
        super().__init__()
        self.top_k = top_k
        self.use_hybrid_search = use_hybrid_search
        self.use_reranking = use_reranking
        
        print("🚀 Initializing RAG pipeline...")
        print(f"   Hybrid search: {use_hybrid_search}")
        print(f"   Reranking: {use_reranking}")
        if use_hybrid_search:
            print(f"   Weights: semantic={semantic_weight}, BM25={bm25_weight}")
        
        self.rag_pipeline = RAGPipeline(
            use_hybrid_search=use_hybrid_search,
            use_reranking=use_reranking,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight
        )
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
            # Use the enhanced retrieve_with_scores method
            retriever = self.rag_pipeline.retriever
            retrieved_results = retriever.retrieve_with_scores(
                query_text=question,
                top_k=self.top_k
            )
            
            contexts = []
            similarity_scores = []
            bm25_scores = []
            rerank_scores = []
            source_documents = []
            
            for r in retrieved_results:
                contexts.append(r['text'])
                similarity_scores.append(r['semantic_score'])
                bm25_scores.append(r.get('bm25_score', 0.0))
                rerank_scores.append(r.get('rerank_score', 0.0))
                source_documents.append(r.get('source', 'unknown'))
            
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
            if self.use_reranking and rerank_scores:
                print(f"📊 Top rerank score: {rerank_scores[0]:.3f}")
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
                    "bm25_scores": bm25_scores if self.use_hybrid_search else [],
                    "rerank_scores": rerank_scores if self.use_reranking else [],
                    "source_documents": source_documents,
                    "num_contexts": len(contexts),
                    "use_hybrid_search": self.use_hybrid_search,
                    "use_reranking": self.use_reranking
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
                    "bm25_scores": [],
                    "rerank_scores": [],
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
