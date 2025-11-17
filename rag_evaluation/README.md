RAG evaluation utilities

This package contains a small tool to assemble per-record context packs from `Processed/` and generate question/answer/evidence kits using Gemini or a deterministic mock.

Usage

1. Install dependencies (if you plan to call a real Gemini endpoint you will need network access and an API key):

```bash
pip install -r requirements.txt
```

2. Run the assembler + generator for the first 10 records:

```bash
python -m rag_evaluation.assemble_and_generate --limit 10
```

3. Output:
- `evaluation_kits/context_packs/` — per-record intermediate JSON context packs
- `evaluation_kits/generated_gold/` — per-record generated QA kits (JSON)

Environment
- `GEMINI_API_KEY`: if set, the script will try to call the configured Gemini API endpoint. If not set, a deterministic mock generator is used.
- `GEMINI_MODEL`: optional model name used when calling the Gemini endpoint.
- `GEMINI_API_ENDPOINT`: optional full endpoint URL to override the default.

Notes
- The Gemini call uses a conservative POST to a generative language endpoint; adapt `call_gemini` in `assemble_and_generate.py` to match your provider's exact API.
- The mock generator is deterministic and useful for offline testing and review.
