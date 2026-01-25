#!/usr/bin/env python3
"""Evaluate generated QA kits using the repo's `rag` pipeline (Qdrant + generator).

For each JSON kit in `evaluation_kits/generated_gold`, this script will:
- load the questions
- retrieve top-k contexts from Qdrant (k default 3)
- generate a RAG answer using the repo `rag.generator`
- append `retrieved_context` and `rag_answer` to each question
- save augmented kits to `evaluation_kits/evaluated/{basename}.json`

Usage:
  python3 rag_evaluation/evaluate_rag.py [--input-dir ...] [--output-dir ...] [--k 3] [--dry-run] [--force]
"""

import argparse
import json
import logging
import time
from pathlib import Path


def load_files(input_dir: Path):
    return sorted(input_dir.expanduser().resolve().glob("*.json"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', default='evaluation_kits/generated_gold')
    p.add_argument('--output-dir', default='evaluation_kits/evaluated')
    p.add_argument('--k', type=int, default=3, help='Top-k retrievals (default 3)')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--log-level', default='INFO')
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s %(levelname)s %(message)s')

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = load_files(input_dir)
    if not files:
        logging.info('No generated kits found in %s', input_dir)
        return

    # Import RAG pipeline from repo. Ensure repo root is on sys.path when running
    try:
        from rag.pipeline import RAGPipeline
    except Exception:
        import sys
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        try:
            from rag.pipeline import RAGPipeline
        except Exception as exc:
            logging.error('Failed to import rag pipeline after adding repo root: %s', exc)
            return

    pipeline = RAGPipeline()

    for path in files:
        basename = path.stem
        out_path = output_dir / f"{basename}.json"
        if out_path.exists() and not args.force:
            logging.info('Skipping %s (already evaluated). Use --force to override.', path.name)
            continue

        logging.info('Processing %s', path.name)
        try:
            kit = json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            logging.error('Failed to load %s: %s', path, exc)
            continue

        questions = kit.get('questions', [])
        context_md = kit.get('context_markdown')

        for q in questions:
            qtext = q.get('question')
            if not qtext:
                continue

            # Retrieve top-k contexts (use retriever directly to avoid double prompting)
            try:
                contexts = pipeline.retriever.retrieve(query_text=qtext, top_k=args.k)
            except Exception as exc:
                logging.warning('Retriever failed for %s: %s', qtext, exc)
                contexts = []

            # Append retrieved_context as list of strings
            q['retrieved_context'] = contexts

            if args.dry_run:
                q['rag_answer'] = None
                continue

            # Build context_text and call generator
            if contexts:
                context_text = '\n\n'.join(contexts)
            else:
                context_text = context_md or ''

            # generator.generate expects (context, question)
            try:
                rag_answer = pipeline.generator.generate(context_text, qtext)
            except Exception as exc:
                logging.warning('Generator failed for %s: %s', qtext, exc)
                rag_answer = None

            q['rag_answer'] = rag_answer

            # brief throttle
            time.sleep(float(__import__('os').getenv('INTER_REQUEST_SECONDS', '0.3')))

        # Save augmented kit
        try:
            out_path.write_text(json.dumps(kit, ensure_ascii=False, indent=2), encoding='utf-8')
            logging.info('Wrote evaluated kit %s', out_path)
        except Exception as exc:
            logging.error('Failed to write %s: %s', out_path, exc)


if __name__ == '__main__':
    main()
