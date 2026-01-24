"""
Utility functions for RAG evaluation.
"""
import json
from typing import List, Dict, Any
import numpy as np


def calculate_exact_match_accuracy(results: List[Dict[str, Any]]) -> float:
    """
    Calculate exact match accuracy between generated answers and ground truth.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Accuracy score (0-1)
    """
    if not results:
        return 0.0
    
    matches = 0
    for result in results:
        answer = result.get("answer", "").lower().strip()
        ground_truth = result.get("ground_truth", "").lower().strip()
        
        if answer == ground_truth:
            matches += 1
    
    return matches / len(results)


def calculate_partial_match_accuracy(results: List[Dict[str, Any]]) -> float:
    """
    Calculate partial match accuracy (ground truth contained in answer or vice versa).
    
    Args:
        results: List of evaluation results
        
    Returns:
        Partial accuracy score (0-1)
    """
    if not results:
        return 0.0
    
    matches = 0
    for result in results:
        answer = result.get("answer", "").lower().strip()
        ground_truth = result.get("ground_truth", "").lower().strip()
        
        # Check if one is contained in the other
        if ground_truth in answer or answer in ground_truth:
            matches += 1
    
    return matches / len(results)


def analyze_retrieval_quality(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze retrieval quality metrics.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary with retrieval statistics
    """
    if not results:
        return {}
    
    all_scores = []
    num_contexts = []
    
    for result in results:
        metadata = result.get("retrieval_metadata", {})
        scores = metadata.get("similarity_scores", [])
        num = metadata.get("num_contexts", 0)
        
        all_scores.extend(scores)
        num_contexts.append(num)
    
    stats = {
        "avg_similarity_score": np.mean(all_scores) if all_scores else 0,
        "min_similarity_score": np.min(all_scores) if all_scores else 0,
        "max_similarity_score": np.max(all_scores) if all_scores else 0,
        "std_similarity_score": np.std(all_scores) if all_scores else 0,
        "avg_num_contexts": np.mean(num_contexts) if num_contexts else 0,
        "total_contexts_retrieved": sum(num_contexts)
    }
    
    return stats


def print_sample_results(results: List[Dict[str, Any]], num_samples: int = 3):
    """
    Print sample results for inspection.
    
    Args:
        results: List of evaluation results
        num_samples: Number of samples to print
    """
    print("\n" + "="*80)
    print(f"SAMPLE RESULTS (showing {num_samples} cases)")
    print("="*80)
    
    for i, result in enumerate(results[:num_samples]):
        print(f"\n--- Case {i+1}: {result['case_id']} ---")
        print(f"Question: {result['question'][:150]}...")
        print(f"\nRetrieved {len(result['contexts'])} contexts")
        print(f"First context preview: {result['contexts'][0][:200] if result['contexts'] else 'None'}...")
        print(f"\nGenerated Answer: {result['answer']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"\nSimilarity Scores: {result['retrieval_metadata']['similarity_scores']}")
        print("-" * 80)


def generate_summary_report(results: List[Dict[str, Any]]) -> str:
    """
    Generate a summary report of the evaluation results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Summary report as string
    """
    exact_acc = calculate_exact_match_accuracy(results)
    partial_acc = calculate_partial_match_accuracy(results)
    retrieval_stats = analyze_retrieval_quality(results)
    
    report = f"""
RAG EVALUATION SUMMARY REPORT
{'='*80}

Total Cases: {len(results)}

Diagnostic Accuracy:
- Exact Match Accuracy: {exact_acc:.2%}
- Partial Match Accuracy: {partial_acc:.2%}

Retrieval Quality:
- Average Similarity Score: {retrieval_stats.get('avg_similarity_score', 0):.4f}
- Min Similarity Score: {retrieval_stats.get('min_similarity_score', 0):.4f}
- Max Similarity Score: {retrieval_stats.get('max_similarity_score', 0):.4f}
- Std Similarity Score: {retrieval_stats.get('std_similarity_score', 0):.4f}
- Average Contexts per Query: {retrieval_stats.get('avg_num_contexts', 0):.1f}
- Total Contexts Retrieved: {retrieval_stats.get('total_contexts_retrieved', 0)}

{'='*80}
"""
    
    return report


def load_results(file_path: str) -> List[Dict[str, Any]]:
    """
    Load results from JSON file.
    
    Args:
        file_path: Path to results file
        
    Returns:
        List of results
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
