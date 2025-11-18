RAG Evaluation Helper
=====================

This small evaluation toolkit helps compute retrieval and basic generation metrics for
RAG runs, focusing on the metrics you specified:

Generation (Quality of Answers):
- Faithfulness (measured via optional NLI entailment proxy and human eval CSV)
- Answer Relevancy (collected via human-eval CSV)

Retrieval (Quality of Context):
- Context Precision@K
- Context Recall@K
- MRR@K

Files
-----
- `scripts/evaluate_rag.py` : main evaluation script
- `scripts/sample_predictions.jsonl` : (optionally) sample input format

Input format (JSONL)
--------------------
Each line is a JSON object with these keys:
- `id` : unique id for the query
- `query` : the user question
- `gold_evidence` : list of ground-truth evidence spans (strings)
- `retrieved` : ordered list of retrieved document text (strings)
- `answer` : model-generated answer text
- `gold_answer` : (optional) human-written answer for automatic comparison

Example JSONL line:
{"id":"q1","query":"What is the likely diagnosis?","gold_evidence":["viral haemorrhagic fever","Ebola virus"],"retrieved":["...Ebola virus disease...","...malaria..."],"answer":"The presentation is most consistent with Ebola virus disease.","gold_answer":"Most consistent with Ebola virus disease."}

How to run
----------
1. Basic retrieval+generation summary

```powershell
python scripts/evaluate_rag.py --pred scripts/sample_predictions.jsonl --k 5 --out results.json
```

2. Also produce a CSV file for human raters

```powershell
python scripts/evaluate_rag.py --pred scripts/sample_predictions.jsonl --k 5 --csv human_eval.csv
```

3. Attempt NLI-based faithfulness scoring (requires `transformers` and model download)

```powershell
python -m pip install transformers torch
python scripts/evaluate_rag.py --pred scripts/sample_predictions.jsonl --k 5 --nli --out results.json
```

Notes
-----
- Precision/Recall calculation uses a simple substring overlap test between retrieved docs and
  `gold_evidence`. If your ground-truth evidence is stored as indices or doc IDs, adapt the script
  to match on IDs instead.
- The NLI-based faithfulness check is only a heuristic. For robust faithfulness evaluation, prefer
  a human rating for faithfulness or a QA-consistency evaluation (generate QA pairs from the answer
  and check answers against the retrieved contexts).

Human Evaluation
----------------
- Use the CSV produced by `--csv` as an input to your human rating workflow (spreadsheet or annotation
  tool). The `instructions` column contains suggested rating guidelines for Faithfulness (0-3) and
  Relevancy (0-3).

Advanced ideas
--------------
- Implement QA-based faithfulness: generate questions from the model's answer (using a QG model),
  answer them using the retrieved contexts, and compare answers; this is stronger than direct NLI.
- Use fuzzy matching (e.g., RapidFuzz) or tokenized overlap to improve retrieval metrics when exact
  substring matching is too strict.

