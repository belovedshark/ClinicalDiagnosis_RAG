#!/usr/bin/env python3
"""
Evaluate RAG retrieval + generation quality using the metrics you specified.

Metrics supported (automated):
- Retrieval: Precision@K, Recall@K, MRR@K
- Generation: simple lexical overlap (ROUGE-L if package installed), and an optional
  NLI-based faithfulness check using a seq2seq/NLI model (optional, requires transformers).

Also produces a CSV for human raters with columns needed to score
- Faithfulness (binary or Likert)
- Answer Relevancy (Likert)

Input format (JSONL): each line is a JSON object with keys:
{
  "id": "unique_query_id",
  "query": "user question",
  "gold_evidence": ["evidence_text_1", "evidence_text_2", ...],
  "retrieved": ["doc text 1", "doc text 2", ...],  # ordered by model
  "answer": "generated answer text",
  "gold_answer": "(optional) human answer for comparison"
}

Usage:
  python scripts/evaluate_rag.py --pred scripts/sample_predictions.jsonl --k 5 --out results.json

"""

import argparse
import json
import math
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

# Optional imports for advanced metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except Exception:
    ROUGE_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    preds = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            preds.append(json.loads(line))
    return preds


def is_evidence_overlap(retrieved_text: str, gold_evidences: List[str]) -> bool:
    """Simple overlap test: check if any gold evidence substring is contained in retrieved text.

    This conservative test assumes gold_evidence contains representative strings or short spans.
    For better recall matching you can normalize whitespace and punctuation or use fuzzy matching.
    """
    rt = ' '.join(retrieved_text.split()).lower()
    for g in gold_evidences:
        if not g:
            continue
        if g.lower().strip() in rt:
            return True
    return False


def precision_at_k(retrieved: List[str], gold_evidence: List[str], k: int) -> float:
    topk = retrieved[:k]
    if not topk:
        return 0.0
    hits = sum(1 for d in topk if is_evidence_overlap(d, gold_evidence))
    return hits / len(topk)


def recall_at_k(retrieved: List[str], gold_evidence: List[str], k: int) -> float:
    # how many gold evidence items are covered by top-k retrieved docs
    if not gold_evidence:
        return 0.0
    topk = retrieved[:k]
    covered = 0
    for g in gold_evidence:
        found = any(g.lower().strip() in (' '.join(t.split()).lower()) for t in topk)
        if found:
            covered += 1
    return covered / len(gold_evidence)


def mrr_at_k(retrieved: List[str], gold_evidence: List[str], k: int) -> float:
    # reciprocal rank of first retrieved doc that contains any gold evidence
    for i, d in enumerate(retrieved[:k], start=1):
        if is_evidence_overlap(d, gold_evidence):
            return 1.0 / i
    return 0.0


def aggregate_retrieval_metrics(items: List[Dict[str, Any]], k: int) -> Dict[str, float]:
    p_sum = 0.0
    r_sum = 0.0
    mrr_sum = 0.0
    n = len(items)
    for it in items:
        retrieved = it.get('retrieved', [])
        gold = it.get('gold_evidence', []) or []
        p_sum += precision_at_k(retrieved, gold, k)
        r_sum += recall_at_k(retrieved, gold, k)
        mrr_sum += mrr_at_k(retrieved, gold, k)
    return {
        'precision@{}'.format(k): p_sum / n if n else 0.0,
        'recall@{}'.format(k): r_sum / n if n else 0.0,
        'mrr@{}'.format(k): mrr_sum / n if n else 0.0,
    }


def compute_rouge(answer: str, gold: Optional[str]) -> Dict[str, float]:
    if not ROUGE_AVAILABLE or gold is None:
        return {}
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    sc = scorer.score(gold, answer)
    return {'rougeL_f': sc['rougeL'].fmeasure}


def nli_faithfulness_check(answer: str, contexts: List[str], nli_model=None) -> Optional[float]:
    """Simple NLI-based faithfulness proxy: check if contexts entail the answer.

    We concatenate the top-k contexts and ask an NLI model to score entailment of the answer
    against the contexts. This is a heuristic — better pipelines generate Q/A pairs and verify
    grounded answers.

    Returns entailment probability if model available, otherwise None.
    """
    if nli_model is None:
        return None
    premise = '\n'.join(contexts[:5])
    # Many NLI models expect inputs as: premise -> hypothesis
    out = nli_model({'premise': premise, 'hypothesis': answer})
    # pipeline output format varies; return score for 'ENTAILMENT' if available
    if isinstance(out, list) and out:
        # huggingface textual entailment pipelines often return label and score
        first = out[0]
        # try multiple possible keys
        label = first.get('label') or first.get('entailment_label')
        score = first.get('score') or first.get('probability')
        if label and score is not None:
            # many label strings: 'ENTAILMENT' or 'entailment'
            if label.upper().startswith('ENTAIL'):
                return float(score)
            else:
                # if not entailment, return negative score mapping
                return 0.0
    return None


def generate_human_eval_csv(items: List[Dict[str, Any]], out_csv: Path) -> None:
    """Create CSV with columns for human raters to score faithfulness and relevancy.

    Columns:
      id, query, answer, gold_answer, retrieved_context_preview (first N chars),
      instructions (short)
    """
    fields = [
        'id', 'query', 'answer', 'gold_answer', 'retrieved_preview', 'instructions'
    ]
    with out_csv.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for it in items:
            rid = it.get('id')
            q = it.get('query', '')
            ans = it.get('answer', '')
            gold = it.get('gold_answer', '')
            retrieved_preview = ''
            retrieved = it.get('retrieved', [])
            if retrieved:
                retrieved_preview = ' ||| '.join(t.replace('\n', ' ') for t in retrieved[:3])
                if len(retrieved_preview) > 300:
                    retrieved_preview = retrieved_preview[:297] + '...'
            instr = (
                "Rate the answer's Faithfulness (0-3): 0=no support, 1=partially supported, "
                "2=mostly supported, 3=fully supported; and Relevancy (0-3): 0=not relevant -> 3=highly relevant"
            )
            w.writerow({
                'id': rid, 'query': q, 'answer': ans, 'gold_answer': gold,
                'retrieved_preview': retrieved_preview, 'instructions': instr
            })


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred', required=True, help='JSONL file with predictions (one JSON per line)')
    p.add_argument('--k', type=int, default=5, help='K for precision@K / recall@K / mrr@K')
    p.add_argument('--out', help='Write aggregated results JSON to this path')
    p.add_argument('--csv', help='Write human-eval CSV to this path')
    p.add_argument('--nli', action='store_true', help='Attempt NLI-based faithfulness scoring (requires transformers)')
    args = p.parse_args()

    preds = load_predictions(Path(args.pred))
    print(f"Loaded {len(preds)} prediction items from {args.pred}")

    ret_metrics = aggregate_retrieval_metrics(preds, args.k)
    print('\nRetrieval metrics:')
    for k, v in ret_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Generation stats: rouge if available; store per-item summaries
    per_item = []
    rouge_enabled = ROUGE_AVAILABLE
    nli_enabled = args.nli and TRANSFORMERS_AVAILABLE
    nli_model = None
    if nli_enabled:
        try:
            # user can change model to a better NLI model; this uses a pipeline if available
            nli_model = pipeline('text-classification', model='facebook/bart-large-mnli')
        except Exception as e:
            print(f"Could not initialize NLI pipeline: {e}")
            nli_model = None

    for it in preds:
        ans = it.get('answer', '')
        gold_ans = it.get('gold_answer')
        contexts = it.get('retrieved', [])
        r = {}
        if rouge_enabled and gold_ans:
            r.update(compute_rouge(ans, gold_ans))
        if nli_enabled and nli_model:
            try:
                r['nli_entailment_score'] = nli_faithfulness_check(ans, contexts, nli_model)
            except Exception as e:
                r['nli_entailment_score'] = None
        per_item.append({'id': it.get('id'), 'retrieval': r})

    # Optionally write CSV for human raters
    if args.csv:
        generate_human_eval_csv(preds, Path(args.csv))
        print(f"Wrote human-eval CSV to {args.csv}")

    out = {'retrieval': ret_metrics, 'per_item': per_item}
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Wrote results to {args.out}")
    else:
        print('\nSummary JSON:')
        print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
