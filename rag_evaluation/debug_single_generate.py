#!/usr/bin/env python3
"""Debug Gemini output for a single context markdown file.

Saves raw model output to `evaluation_kits/debug_raw/{basename}.txt` and prints it.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Optional

try:
    from scripts.table_restructure import make_model
except Exception:
    import sys
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from scripts.table_restructure import make_model
    except Exception:
        import importlib.util
        tr_path = repo_root / 'scripts' / 'table_restructure.py'
        spec = importlib.util.spec_from_file_location('table_restructure', str(tr_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        make_model = getattr(module, 'make_model')


PROMPT_TEMPLATE = (
    "Given this record content:\n\n{CONTEXT}\n\n"
    "Produce up to 3 clinically meaningful questions that can be answered ONLY by information inside this record. "
    "For each question return JSON only with fields `question`, `gold_answer`, and `gold_evidence` (exact supporting sentence(s) from the record). "
    "The `question` should include: most likely diagnosis, key differential(s) for fever when relevant, and a brief recommended management approach when applicable. Return JSON only."
)


def generate_with_gemini(model, prompt: str, max_retries: int = 3, backoff: float = 1.0) -> Optional[str]:
    for attempt in range(1, max_retries + 1):
        try:
            resp = model.generate_content(prompt)
            out = getattr(resp, 'text', None) or getattr(resp, 'content', None)
            if isinstance(out, bytes):
                out = out.decode('utf-8')
            return out
        except Exception as exc:
            logging.warning("Gemini call failed (attempt %d/%d): %s", attempt, max_retries, exc)
            if attempt == max_retries:
                logging.error("Max retries reached for Gemini call. Skipping.")
                return None
            time.sleep(backoff * attempt)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('md_path', help='Path to the markdown context file')
    p.add_argument('--out-dir', default='evaluation_kits/debug_raw')
    args = p.parse_args()

    md_file = Path(args.md_path)
    if not md_file.exists():
        print('File not found:', md_file)
        return

    raw_text = md_file.read_text(encoding='utf-8')
    prompt = PROMPT_TEMPLATE.format(CONTEXT=raw_text)

    api_key = os.getenv('GOOGLE_API_KEY')
    model = make_model(api_key)

    raw_out = generate_with_gemini(model, prompt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    basename = md_file.stem
    out_path = out_dir / f"{basename}.txt"
    if raw_out is None:
        print('No output from model')
        out_path.write_text('', encoding='utf-8')
    else:
        # ensure a string
        if isinstance(raw_out, bytes):
            raw_out = raw_out.decode('utf-8')
        out_path.write_text(raw_out, encoding='utf-8')
        print('Raw model output saved to', out_path)
        print('---BEGIN RAW OUTPUT---')
        print(raw_out)
        print('---END RAW OUTPUT---')


if __name__ == '__main__':
    main()
