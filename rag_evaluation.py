"""RAG evaluation using RAGAS only.

This script does the following for a single gold JSON file (use `--file`):
1. Loads the gold JSON (questions, gold answers, gold evidence).
2. Uses the in-repo `RAGPipeline` to retrieve contexts and generate an answer per question.
3. Builds a RAGAS Dataset with columns expected by RAGAS:
   - `question`, `answer` (generated), `ground_truth` (gold), `contexts`
4. Calls `ragas.evaluate(..., metrics=[...])` with the 3 metrics and saves results.

Note: This script requires `ragas` and `pandas` to be installed. It does not
implement any fallback — an ImportError will be raised if `ragas` is missing.
"""
import os
import json
import glob
import argparse
from typing import List
from rag.pipeline import RAGPipeline
import pandas as pd
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
)
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

def build_ragas_rows(pipeline: RAGPipeline, questions: List[dict], k: int = 5):
    retriever = pipeline.retriever
    generator = pipeline.generator
    rows = []

    for qobj in questions:
        qtext = qobj.get('question')
        gold_answer = qobj.get('gold_answer', '')
        gold_evidence = qobj.get('gold_evidence', [])

        # Retrieve contexts and generate
        contexts = retriever.retrieve(query_text=qtext, top_k=k)
        context_text = "\n\n".join(contexts)
        prediction = generator.generate(context_text, qtext)

        rows.append({
            'question': qtext,
            'answer': prediction,
            'contexts': contexts,
            'ground_truth': gold_answer,
        })

    return rows


def main(args):
    if args.file:
        if not os.path.isfile(args.file):
            raise FileNotFoundError(f"Specified file not found: {args.file}")
        files = [args.file]
    else:
        gold_dir = args.gold_dir
        files = sorted(glob.glob(os.path.join(gold_dir, '*.json')))
        if not files:
            raise FileNotFoundError(f"No gold files found in {gold_dir}")

    print("Initializing RAG pipeline (models may load to GPU/cpu)...")
    pipeline = RAGPipeline()

    all_rows = []
    for fpath in files:
        print(f"Processing: {fpath}")
        with open(fpath, 'r', encoding='utf-8') as fh:
            j = json.load(fh)

        questions = j.get('questions', [])
        rows = build_ragas_rows(pipeline, questions, k=args.k)
        # Attach record id for traceability
        for r in rows:
            r['record_id'] = j.get('record_id', os.path.basename(fpath))
            r['file'] = os.path.basename(fpath)
        all_rows.extend(rows)

    if not all_rows:
        print("No question rows produced. Exiting.")
        return

    dataset = Dataset.from_list(all_rows)
    print(f"Running RAGAS evaluation on {len(dataset)} rows...")

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_retries=3, timeout=30)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    result = ragas_evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    try:
        res_df = result.to_pandas()
        out_path = args.out or os.path.join('evaluation_kits', 'ragas_evaluation_results.json')
        res_df.to_json(out_path, orient='records', force_ascii=False, indent=2)
        print(f"Saved RAGAS evaluation results to {out_path}")
        
        print("\n=== RAGAS Evaluation Results ===")
        for metric in ['faithfulness', 'answer_relevancy', 'context_recall']:
            if metric in res_df.columns:
                print(f"{metric}: {res_df[metric].mean():.4f}")
    except Exception as e:
        # If .to_pandas() not available, save string repr
        print(f"Warning: Could not convert to pandas: {e}")
        out_path = args.out or os.path.join('evaluation_kits', 'ragas_evaluation_results.txt')
        with open(out_path, 'w', encoding='utf-8') as fh:
            fh.write(str(result))
        print(f"Saved RAGAS raw result to {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gold-dir', default=os.path.join('evaluation_kits', 'generated_gold'), help='Directory with gold JSON files')
    p.add_argument('--file', default=None, help='Path to a single gold JSON file to evaluate (overrides --gold-dir)')
    p.add_argument('--k', type=int, default=5, help='Number of contexts to retrieve/generate with')
    p.add_argument('--out', default=None, help='Output file path for RAGAS results')
    args = p.parse_args()
    main(args)