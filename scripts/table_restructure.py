import os
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path
# import os
import time
import argparse
import logging
# from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path
from typing import Optional


load_dotenv()

# Configure Gemini model (lazy-init after reading API key)
_MODEL_NAME = "gemini-1.5-flash"


def make_model(api_key: Optional[str]):
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in environment")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(_MODEL_NAME)


def extract_table_with_gemini(model, text: str, max_retries: int = 3, backoff: float = 1.0) -> Optional[str]:
    """Use Gemini to reformat the following raw text into a Markdown table.

    Retries on transient failures and returns None on permanent failure.
    """
    prompt = f"""
You are a medical data formatter.
Convert the following unstructured text-based medical table into a clean Markdown table.
Preserve clinical and diagnostic details. If the input does not contain a table,
return an empty string.

Input:
{text}

Output:
Markdown table only.
"""

    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt)
            # Some API clients return an object with `text` or `content` fields
            out = getattr(response, "text", None) or getattr(response, "content", None)
            if isinstance(out, bytes):
                out = out.decode("utf-8")
            return out.strip() if out is not None else None
        except Exception as exc:
            logging.warning("Gemini call failed (attempt %d/%d): %s", attempt, max_retries, exc)
            if attempt == max_retries:
                logging.error("Max retries reached for Gemini call. Skipping this file.")
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
    for f in md_files:
        logging.info("Processing %s", f.name)
        raw_text = f.read_text(encoding="utf-8")
        if dry_run:
            logging.info("Dry run: skipping API call for %s", f.name)
            continue

        formatted = extract_table_with_gemini(model, raw_text)
        if formatted is None:
            logging.error("Failed to process %s: no output from model", f.name)
            continue

        out_path = output_dir / f.name
        out_path.write_text(formatted or "", encoding="utf-8")
        logging.info("Wrote restructured output to %s", out_path)
        # small pause to be nice to the API
        time.sleep(0.3)


def parse_args():
    p = argparse.ArgumentParser(description="Restructure extracted markdown tables using Gemini")
    p.add_argument("--input-dir", default="out_all", help="Directory containing extracted markdown files")
    p.add_argument("--output-dir", default="out_all_structured", help="Directory to write restructured tables")
    p.add_argument("--dry-run", action="store_true", help="Don't call the API; just list files")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    api_key = os.getenv("GOOGLE_API_KEY")
    process_all(Path(args.input_dir), Path(args.output_dir), api_key, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
