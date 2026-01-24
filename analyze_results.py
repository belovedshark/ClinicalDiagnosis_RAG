"""
Quick analysis script to view RAG inference results.

Usage:
    python analyze_results.py
"""
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from rag_evaluation.utils.evaluation_utils import (
    load_results,
    print_sample_results,
    generate_summary_report,
    calculate_exact_match_accuracy,
    calculate_partial_match_accuracy,
    analyze_retrieval_quality
)
from rag_evaluation.config.evaluation_config import RAG_INFERENCE_RESULTS_FILE


def main():
    """Analyze and display RAG inference results."""
    print("="*80)
    print("RAG INFERENCE RESULTS ANALYSIS")
    print("="*80)
    
    # Check if results file exists
    if not os.path.exists(RAG_INFERENCE_RESULTS_FILE):
        print(f"\n❌ Results file not found: {RAG_INFERENCE_RESULTS_FILE}")
        print("   Please run the inference pipeline first:")
        print("   python run_rag_inference.py")
        return
    
    # Load results
    print(f"\n📂 Loading results from: {RAG_INFERENCE_RESULTS_FILE}")
    results = load_results(RAG_INFERENCE_RESULTS_FILE)
    print(f"✅ Loaded {len(results)} cases")
    
    # Generate and print summary report
    report = generate_summary_report(results)
    print(report)
    
    # Print sample results
    print_sample_results(results, num_samples=3)
    
    # Additional detailed analysis
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    # Find best and worst cases by similarity scores
    cases_with_avg_score = []
    for r in results:
        scores = r["retrieval_metadata"].get("similarity_scores", [])
        if scores:
            avg_score = sum(scores) / len(scores)
            cases_with_avg_score.append((r["case_id"], avg_score, r))
    
    if cases_with_avg_score:
        cases_with_avg_score.sort(key=lambda x: x[1], reverse=True)
        
        print("\n🏆 Top 3 Cases (Highest Retrieval Scores):")
        for case_id, avg_score, result in cases_with_avg_score[:3]:
            print(f"  {case_id}: {avg_score:.4f}")
            print(f"    Answer: {result['answer']}")
            print(f"    Ground Truth: {result['ground_truth']}")
        
        print("\n⚠️  Bottom 3 Cases (Lowest Retrieval Scores):")
        for case_id, avg_score, result in cases_with_avg_score[-3:]:
            print(f"  {case_id}: {avg_score:.4f}")
            print(f"    Answer: {result['answer']}")
            print(f"    Ground Truth: {result['ground_truth']}")
    
    # Check for exact and partial matches
    exact_matches = []
    partial_matches = []
    no_matches = []
    
    for r in results:
        answer = r["answer"].lower().strip()
        gt = r["ground_truth"].lower().strip()
        
        if answer == gt:
            exact_matches.append(r["case_id"])
        elif gt in answer or answer in gt:
            partial_matches.append(r["case_id"])
        else:
            no_matches.append(r["case_id"])
    
    print(f"\n📊 Match Distribution:")
    print(f"  Exact matches: {len(exact_matches)}/{len(results)}")
    print(f"  Partial matches: {len(partial_matches)}/{len(results)}")
    print(f"  No matches: {len(no_matches)}/{len(results)}")
    
    if no_matches and len(no_matches) <= 10:
        print(f"\n  Cases with no match: {', '.join(no_matches)}")
    
    print("\n✅ Analysis complete!")
    print("   Results are ready for RAGAS evaluation (Phase 2)")
    

if __name__ == "__main__":
    main()
