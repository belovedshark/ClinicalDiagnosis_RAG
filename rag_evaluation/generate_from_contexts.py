#!/usr/bin/env python3
"""Generate QA kits from per-record markdown contexts using Gemini.

This script iterates `.md` files in `evaluation_kits/context_packs_md/`, calls Gemini
to generate questions, gold answers and gold evidence (JSON-only), and writes the
results to `evaluation_kits/generated_gold/{basename}.json`.

It reuses the Gemini model initialization from `scripts/table_restructure.py`.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Optional

try:
    # normal package-style import (works when running as module)
    from scripts.table_restructure import make_model
except Exception:
    # fallback: ensure repo root is on sys.path then try again
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from scripts.table_restructure import make_model
    except Exception:
        # final fallback: load the file by path as a module
        import importlib.util
        tr_path = repo_root / 'scripts' / 'table_restructure.py'
        spec = importlib.util.spec_from_file_location('table_restructure', str(tr_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        make_model = getattr(module, 'make_model')


PROMPT_TEMPLATE = (
    """
    Given this record content:\n\n{CONTEXT}\n\n
    You are creating evaluation data for a medical RAG system. Your job is to generate questions that are 100% answerable ONLY using the content of the provided medical record.

    Generate **exactly 2** clinically meaningful questions (include a short patient situation (age, background, symptoms) that:
    - MUST be fully answerable *using only sentences that are explicitly present in the record*.
    - MUST NOT require any external medical knowledge, assumptions, or inference.
    - MUST be short (ideal length: 1-3 sentences).
    - MUST NOT include anything that is not literally stated in the record.

    The question MUST combine:
    1. a patient-specific situation summary (using exact wording from the record)
    2. a question about:
    - a diagnosis explicitly given in the record, OR
    - a treatment explicitly given in the record, OR
    - a clinical action explicitly described in the record
    You MUST NOT ask about "most likely diagnosis" or "recommended treatment" unless that information appears *verbatim* in the record.

    For each question, produce:

    1. "question":  
    - Must use ONLY facts from the record.  
    - MUST NOT introduce any medical reasoning.  
    - MUST be answerable with a quote from the record.

    2. "gold_answer":  
    - MUST be copied or minimally summarized from information explicitly stated in the record.  
    - MUST NOT rely on medical knowledge or logic not present in the text.

    3. "gold_evidence":  
    - MUST be an array of exact sentences *copied verbatim* from the record.  
    - Each evidence span must appear exactly in the record with no changes.  
    - MUST include only the minimal text needed to answer the question.

    STRICT RULES:
    - DO NOT infer.  
    - DO NOT paraphrase evidence.  
    - DO NOT provide answers drawn from the Summary Box unless the question is explicitly based on it.  
    - DO NOT use any knowledge outside the provided record.  
    - If a detail is not explicitly in the record, you MUST NOT mention it.  
    - If the question cannot be answered using EXACT sentences from the text, DO NOT generate that question.
    """
   )

# Make the model output strictly JSON only and show an explicit example to improve parsability.
PROMPT_TEMPLATE = PROMPT_TEMPLATE + (
"\n\nOUTPUT FORMAT (IMPORTANT): Return ONLY valid JSON with a top-level key \"questions\" whose value is a list of exactly 3 objects.\n"
"Each object must have the keys \"question\", \"gold_answer\", and \"gold_evidence\". Do not include any extra text, commentary, or markdown.\n"
"Example output (exact JSON structure):\n\n"
"{{\n"
"  \"questions\": [\n"
"    {{\n"
"      \"question\": \"What was the patient's temperature on admission?\",\n"
"      \"gold_answer\": \"39.6°C\",\n"
"      \"gold_evidence\": \"Vital signs: temperature 39.6°C\"\n"
"    }},\n"
"    {{ ... }},\n"
"    {{ ... }}\n"
"  ]\n"
"}}\n"
)


def safe_parse_json(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None

    t = text.strip()

    # remove triple-backtick fences if present (``` or ```json ... ```)
    if t.startswith('```') and t.endswith('```'):
        parts = t.split('\n')
        if parts and parts[0].startswith('```'):
            parts = parts[1:]
        if parts and parts[-1].strip().startswith('```'):
            parts = parts[:-1]
        t = '\n'.join(parts).strip()

    # try direct parse first
    try:
        return json.loads(t)
    except Exception:
        pass

    # Try to robustly extract the first JSON object/array substring
    def _extract_and_parse(s: str):
        for start_char, end_char in (("{", "}"), ("[", "]")):
            start = s.find(start_char)
            if start == -1:
                continue
            stack = []
            in_string = False
            escape = False
            for i in range(start, len(s)):
                ch = s[i]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == '\\':
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                else:
                    if ch == '"':
                        in_string = True
                        continue
                    if ch == start_char:
                        stack.append(ch)
                    elif ch == end_char:
                        if stack:
                            stack.pop()
                            if not stack:
                                candidate = s[start:i+1]
                                try:
                                    return json.loads(candidate)
                                except Exception:
                                    # fail and continue searching
                                    break
            # nothing parsed for this bracket type, continue
        return None

    parsed = _extract_and_parse(t)
    if parsed is not None:
        return parsed

    # Fallback: attempt to heuristically extract the outermost JSON substring
    # by finding the first '{' or '[' and the last matching '}' or ']' and parsing that.
    for start_char, end_char in (('{', '}'), ('[', ']')):
        start = t.find(start_char)
        end = t.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            candidate = t[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                continue

    return None


def generate_with_gemini(model, prompt: str, max_retries: int = 3, backoff: float = 1.0) -> Optional[str]:
    """Call model.generate_content(prompt) with retries and return raw text output or None."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = model.generate_content(prompt)
            out = getattr(resp, 'text', None) or getattr(resp, 'content', None)
            if isinstance(out, bytes):
                out = out.decode('utf-8')
            return out.strip() if out is not None else None
        except Exception as exc:
            logging.warning("Gemini call failed (attempt %d/%d): %s", attempt, max_retries, exc)
            if attempt == max_retries:
                logging.error("Max retries reached for Gemini call. Skipping.")
                return None
            time.sleep(backoff * attempt)


def process_all(input_dir: Path, output_dir: Path, api_key: Optional[str], dry_run: bool = False):
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = make_model(api_key)

    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        logging.info("No markdown files found in %s", input_dir)
        return

    logging.info("Found %d markdown files in %s", len(md_files), input_dir)
    request_count = 0
    max_requests_per_minute = int(os.getenv('MAX_REQUESTS_PER_MINUTE', '15'))
    cooldown_seconds = int(os.getenv('COOLDOWN_SECONDS', '65'))

    for idx, f in enumerate(md_files, start=1):
        logging.info("Processing %s (%d/%d)", f.name, idx, len(md_files))
        raw_text = f.read_text(encoding='utf-8')

        if not raw_text or raw_text.strip() == "":
            logging.info("Skipping %s: empty", f.name)
            continue

        if dry_run:
            logging.info("Dry run: skipping API call for %s", f.name)
            continue

        prompt = PROMPT_TEMPLATE.format(CONTEXT=raw_text)
        raw_out = generate_with_gemini(model, prompt)

        if raw_out is None:
            logging.error("No output from Gemini for %s", f.name)
            result = {"questions": [], "raw_response": None}
        else:
            parsed = safe_parse_json(raw_out)
            if parsed is None:
                logging.warning("Response for %s not JSON-parsable; saving raw output", f.name)
                result = {"questions": [], "raw_response": raw_out}
            else:
                # If model returned a bare list of question objects, wrap it
                if isinstance(parsed, list):
                    result = {"questions": parsed, "raw_response": raw_out}
                elif isinstance(parsed, dict) and 'questions' in parsed:
                    result = {"questions": parsed.get('questions', []), "raw_response": raw_out}
                else:
                    # unexpected shape: store under questions if possible
                    if isinstance(parsed, dict):
                        result = {"questions": parsed.get('questions', []), "raw_response": raw_out}
                    else:
                        result = {"questions": [], "raw_response": raw_out}

        # Build kit and save
        basename = f.stem
        kit = {
            "record_id": f.name,
            "source_files": {"context_md": str(f)},
            "context_markdown": raw_text,
            "questions": result.get('questions', []),
            "generator_meta": {
                "generator": "gemini",
                "prompt_version": "v1",
                "date_created": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            }
        }

        out_path = output_dir / f"{basename}.json"
        out_path.write_text(json.dumps(kit, ensure_ascii=False, indent=2), encoding='utf-8')
        logging.info("Wrote generated kit %s", out_path)

        request_count += 1
        time.sleep(float(os.getenv('INTER_REQUEST_SECONDS', '0.3')))

        if request_count >= max_requests_per_minute:
            logging.info("Reached %d requests. Sleeping for %d seconds...", max_requests_per_minute, cooldown_seconds)
            time.sleep(cooldown_seconds)
            request_count = 0


def main():
    import argparse

    p = argparse.ArgumentParser(description="Generate QA kits from context markdown using Gemini")
    p.add_argument('--input-dir', default='evaluation_kits/context_packs_md', help='Directory with per-record .md context files')
    p.add_argument('--output-dir', default='evaluation_kits/generated_gold', help='Directory to write generated JSON kits')
    p.add_argument('--dry-run', action='store_true', help='Do not call the API')
    p.add_argument('--log-level', default='INFO')
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s %(levelname)s %(message)s')

    api_key = os.getenv('GOOGLE_API_KEY')
    process_all(Path(args.input_dir), Path(args.output_dir), api_key, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
