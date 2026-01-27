# Evaluation Framework

Unified evaluation framework for comparing different clinical diagnosis model configurations.

## Overview

This framework provides a consistent interface for evaluating:

| Evaluator | Status | Description |
|-----------|--------|-------------|
| `rag` | ✅ Ready | RAG (Retrieval-Augmented Generation) - retrieves context from vector DB |
| `base_model` | ✅ Ready | Base LLM without retrieval |
| `finetuned` | 📝 Placeholder | Fine-tuned model without retrieval |
| `finetuned_rag` | 📝 Placeholder | Fine-tuned model with RAG retrieval |

All evaluators use the same input format (`test_cases.jsonl`) and produce the same output format for easy comparison.

## Quick Start

### Prerequisites

1. **For RAG evaluation**: Ensure Qdrant is running
   ```bash
   docker start qdrant
   # Or: docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Dependencies**: Make sure all required packages are installed
   ```bash
   pip install -r requirements.txt
   ```

### Running Evaluations

From the project root directory:

```bash
# Run RAG evaluation
python -m evaluation.run_evaluation rag

# Run Base Model evaluation
python -m evaluation.run_evaluation base_model

# Run all available evaluators
python -m evaluation.run_evaluation all
```

Or run evaluators directly:

```bash
python -m evaluation.rag.evaluator
python -m evaluation.base_model.evaluator
```

## Output

Results are saved to each evaluator's `results/` folder:

```
evaluation/
├── rag/results/
│   ├── inference_results.json    # Final results
│   └── checkpoint.json           # Checkpoint for resume
├── base_model/results/
│   ├── inference_results.json
│   └── checkpoint.json
```

### Output Format

Each result entry contains:

```json
{
  "case_id": "who_case_001",
  "question": "A 21-year-old male presents with...",
  "contexts": ["..."],           
  "answer": "Dengue fever",
  "ground_truth": "Dengue fever",
  "metadata": {
    "model_type": "rag",
    "num_contexts": 5
  },
  "diagnostic_reasoning": "..."
}
```

- `contexts`: Retrieved contexts (empty for base_model)
- `metadata`: Model-specific metadata

## Analyzing Results

Use the metrics module to analyze results:

```python
from evaluation.metrics import (
    load_results,
    generate_summary_report,
    compare_models
)

# Load and analyze single model
results = load_results("evaluation/rag/results/inference_results.json")
print(generate_summary_report(results))

# Compare multiple models
all_results = {
    "rag": load_results("evaluation/rag/results/inference_results.json"),
    "base_model": load_results("evaluation/base_model/results/inference_results.json"),
}
print(compare_models(all_results))
```

## Folder Structure

```
evaluation/
├── README.md               # This file
├── test_cases.jsonl        # Input test cases (10 WHO clinical cases)
├── base_interface.py       # Abstract BaseEvaluator class
├── config.py               # Shared configuration
├── utils.py                # Data loading/saving utilities
├── metrics.py              # Evaluation metrics
├── run_evaluation.py       # Main entry point
│
├── rag/                    # RAG evaluator
│   ├── evaluator.py
│   └── results/
│
├── base_model/             # Base model evaluator
│   ├── evaluator.py
│   └── results/
│
├── finetuned/              # Fine-tuned evaluator (placeholder)
│   └── evaluator.py
│
└── finetuned_rag/          # Fine-tuned + RAG evaluator (placeholder)
    └── evaluator.py
```

## Adding New Evaluators

To add a new evaluator, implement the `BaseEvaluator` interface:

```python
from evaluation.base_interface import BaseEvaluator

class MyEvaluator(BaseEvaluator):
    MODEL_TYPE = "my_model"
    
    def __init__(self):
        super().__init__()
        # Initialize your model
    
    def run_inference(self, case: dict) -> dict:
        # Process single case and return result
        return {
            "case_id": case["case_id"],
            "question": case["question"],
            "contexts": [],  # or retrieved contexts
            "answer": "predicted diagnosis",
            "ground_truth": case["ground_truth"],
            "metadata": {"model_type": self.MODEL_TYPE},
            "diagnostic_reasoning": case.get("diagnostic_reasoning", "")
        }
```

## Checkpointing

Evaluations automatically checkpoint progress every 5 cases. To resume an interrupted evaluation, simply run the same command again - it will continue from where it left off.

To start fresh, delete the checkpoint file:
```bash
rm evaluation/rag/results/checkpoint.json
```
