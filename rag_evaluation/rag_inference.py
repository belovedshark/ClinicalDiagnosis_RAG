"""
RAG Inference Pipeline for Evaluation

This script runs the RAG system on each case from the WHO evaluation dataset,
collects the retrieved contexts, generated diagnosis, and saves results for
RAGAS evaluation.
"""
import sys
import os
import time
from typing import List, Dict, Any
from tqdm import tqdm

# Add parent directory to path to import rag module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline import RAGPipeline
from rag.retriever import Retriever
from config.evaluation_config import (
    EVALUATION_DATA_PATH,
    TOP_K_RETRIEVAL,
    RAG_INFERENCE_RESULTS_FILE,
    RAG_INFERENCE_CHECKPOINT_FILE,
    SAVE_INTERVAL
)
from data_preparation import (
    load_jsonl,
    prepare_evaluation_cases,
    validate_case,
    save_results,
    load_checkpoint,
    save_checkpoint
)


class RAGInferencePipeline:
    """Pipeline for running RAG inference on evaluation dataset."""
    
    def __init__(self, top_k: int = TOP_K_RETRIEVAL):
        """
        Initialize the RAG inference pipeline.
        
        Args:
            top_k: Number of contexts to retrieve for each query
        """
        self.top_k = top_k
        print("🚀 Initializing RAG pipeline...")
        self.rag_pipeline = RAGPipeline()
        print("✅ RAG pipeline initialized successfully")
        
    def run_inference_single(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run RAG inference on a single case.
        
        Args:
            case: Evaluation case dictionary
            
        Returns:
            Dictionary with inference results
        """
        question = case["question"]
        
        # Retrieve contexts using the retriever directly for metadata
        print(f"\n📋 Processing case: {case['case_id']}")
        print(f"❓ Question: {question[:100]}...")
        
        try:
            # Get contexts with metadata
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
            # Note: We modify the query method temporarily to get raw diagnosis
            context_text = "\n\n".join(contexts)
            
            # Create a custom prompt for diagnosis only
            diagnosis_prompt = f"""
You are a clinical diagnostic assistant. Based on the patient case and medical literature provided, 
give ONLY the final diagnosis. Be concise and specific.

Patient case:
{question}

Answer with just the diagnosis name (e.g., "Dengue fever", "Malaria", etc.):
"""
            
            # Generate the diagnosis
            answer = self.rag_pipeline.generator.generate(context_text, diagnosis_prompt)
            
            # Clean up the answer (remove extra text if present)
            answer = answer.strip()
            if len(answer) > 100:  # If answer is too long, try to extract diagnosis
                # Take first line or first sentence
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
                "retrieval_metadata": {
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
            
            # Return error result
            return {
                "case_id": case["case_id"],
                "question": question,
                "contexts": [],
                "answer": f"ERROR: {str(e)}",
                "ground_truth": case["ground_truth"],
                "retrieval_metadata": {
                    "similarity_scores": [],
                    "source_documents": [],
                    "num_contexts": 0,
                    "error": str(e)
                },
                "diagnostic_reasoning": case.get("diagnostic_reasoning", "")
            }
    
    def run_inference_batch(self, cases: List[Dict[str, Any]], 
                           resume_from: int = 0) -> List[Dict[str, Any]]:
        """
        Run RAG inference on a batch of cases.
        
        Args:
            cases: List of evaluation cases
            resume_from: Index to resume from (for checkpoint recovery)
            
        Returns:
            List of inference results
        """
        results = []
        
        print(f"\n📊 Processing {len(cases)} cases (starting from index {resume_from})")
        
        for idx, case in enumerate(tqdm(cases[resume_from:], 
                                       desc="Running RAG inference",
                                       initial=resume_from,
                                       total=len(cases))):
            actual_idx = resume_from + idx
            
            # Validate case
            if not validate_case(case):
                print(f"⚠️  Skipping invalid case at index {actual_idx}")
                continue
            
            # Run inference
            result = self.run_inference_single(case)
            results.append(result)
            
            # Save checkpoint periodically
            if (actual_idx + 1) % SAVE_INTERVAL == 0:
                checkpoint_data = {
                    "last_processed_index": actual_idx,
                    "results": results,
                    "timestamp": time.time()
                }
                save_checkpoint(checkpoint_data, RAG_INFERENCE_CHECKPOINT_FILE)
                print(f"💾 Checkpoint saved at index {actual_idx}")
        
        return results


def main():
    """Main execution function for RAG inference pipeline."""
    print("="*80)
    print("RAG INFERENCE PIPELINE - Phase 1")
    print("="*80)
    
    # Load evaluation dataset
    print(f"\n📂 Loading evaluation dataset from: {EVALUATION_DATA_PATH}")
    raw_data = load_jsonl(EVALUATION_DATA_PATH)
    print(f"✅ Loaded {len(raw_data)} cases from WHO dataset")
    
    # Prepare cases
    print("\n🔧 Preparing evaluation cases...")
    cases = prepare_evaluation_cases(raw_data)
    print(f"✅ Prepared {len(cases)} cases for evaluation")
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint(RAG_INFERENCE_CHECKPOINT_FILE)
    resume_from = 0
    existing_results = []
    
    if checkpoint:
        resume_from = checkpoint.get("last_processed_index", -1) + 1
        existing_results = checkpoint.get("results", [])
        print(f"\n♻️  Resuming from checkpoint at index {resume_from}")
        print(f"   Already processed: {len(existing_results)} cases")
    
    # Initialize RAG pipeline
    inference_pipeline = RAGInferencePipeline(top_k=TOP_K_RETRIEVAL)
    
    # Run inference
    print(f"\n🚀 Starting RAG inference...")
    new_results = inference_pipeline.run_inference_batch(cases, resume_from=resume_from)
    
    # Combine with existing results if resuming
    all_results = existing_results + new_results
    
    # Save final results
    print(f"\n💾 Saving final results...")
    save_results(all_results, RAG_INFERENCE_RESULTS_FILE)
    
    # Print summary
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print(f"Total cases processed: {len(all_results)}")
    print(f"Results saved to: {RAG_INFERENCE_RESULTS_FILE}")
    
    # Calculate basic statistics
    successful = sum(1 for r in all_results if not r["answer"].startswith("ERROR"))
    failed = len(all_results) - successful
    
    print(f"\n📊 Summary:")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    
    if successful > 0:
        avg_contexts = sum(r["retrieval_metadata"]["num_contexts"] 
                          for r in all_results if not r["answer"].startswith("ERROR")) / successful
        print(f"  📚 Average contexts retrieved: {avg_contexts:.1f}")
    
    print("\n✅ Phase 1 complete! Results ready for RAGAS evaluation.")
    

if __name__ == "__main__":
    main()
