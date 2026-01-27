# Evaluation Framework

Unified evaluation framework for comparing different clinical diagnosis model configurations.

## Overview

This framework provides a consistent interface for evaluating:

### Inference Evaluators

| Evaluator | Status | Description |
|-----------|--------|-------------|
| `rag` | ✅ Ready | RAG (Retrieval-Augmented Generation) - retrieves context from vector DB |
| `base_model` | ✅ Ready | Base LLM without retrieval |
| `finetuned` | 📝 Placeholder | Fine-tuned model without retrieval |
| `finetuned_rag` | 📝 Placeholder | Fine-tuned model with RAG retrieval |

### Framework Evaluators (Post-Inference)

| Framework | Status | Description |
|-----------|--------|-------------|
| `ragas` | ✅ Ready | RAGAS metrics for RAG quality (faithfulness, context precision, etc.) |
| `deepeval` | ✅ Ready | DeepEval G-Eval for clinical reasoning evaluation |
| `frameworks` | ✅ Ready | Run both RAGAS and DeepEval together |

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

### Running Inference Evaluations

From the project root directory:

```bash
# Run RAG evaluation
python -m evaluation.run_evaluation rag

# Run Base Model evaluation
python -m evaluation.run_evaluation base_model

# Run all inference evaluators
python -m evaluation.run_evaluation all
```

Or run evaluators directly:

```bash
python -m evaluation.rag.evaluator
python -m evaluation.base_model.evaluator
```

### Running Framework Evaluations (RAGAS & DeepEval)

Framework evaluations run on existing inference results to compute additional quality metrics:

```bash
# Run RAGAS evaluation on RAG results
python -m evaluation.run_evaluation ragas --results evaluation/rag/results/inference_results.json

# Run DeepEval reasoning evaluation
python -m evaluation.run_evaluation deepeval --results evaluation/rag/results/inference_results.json

# Run both frameworks together
python -m evaluation.run_evaluation frameworks --results evaluation/rag/results/inference_results.json
```

Or run framework evaluators directly:

```bash
# RAGAS
python -m evaluation.frameworks.ragas_evaluator evaluation/rag/results/inference_results.json -o ragas_results.json

# DeepEval
python -m evaluation.frameworks.deepeval_evaluator evaluation/rag/results/inference_results.json -o deepeval_results.json
```

**Note**: Framework evaluations require an OpenAI API key set as `OPENAI_API_KEY` environment variable.

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
├── frameworks/results/           # Framework evaluation results
│   ├── ragas_inference_results.json
│   ├── deepeval_inference_results.json
│   └── combined_inference_results.json
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
├── test_cases.jsonl        # Input test cases (WHO clinical cases)
├── base_interface.py       # Abstract BaseEvaluator class
├── config.py               # Shared configuration
├── utils.py                # Data loading/saving utilities
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
├── finetuned_rag/          # Fine-tuned + RAG evaluator (placeholder)
│   └── evaluator.py
│
└── frameworks/             # External evaluation frameworks
    ├── __init__.py
    ├── ragas_evaluator.py      # RAGAS metrics wrapper
    ├── deepeval_evaluator.py   # DeepEval reasoning metrics
    └── results/                # Framework evaluation outputs
```

## Framework Metrics

### RAGAS Metrics

RAGAS (Retrieval Augmented Generation Assessment) evaluates RAG pipeline quality:

| Metric | Description |
|--------|-------------|
| `faithfulness` | Is the answer factually grounded in the retrieved contexts? |
| `answer_relevancy` | How relevant is the answer to the question? |
| `context_precision` | How much of the retrieved context is actually relevant? |
| `context_recall` | Does the context contain the information needed to answer? |

### DeepEval Reasoning Metrics

DeepEval uses G-Eval (LLM-as-judge) for clinical reasoning evaluation:

| Metric | Description |
|--------|-------------|
| `coherence` | Is the diagnostic reasoning logically structured? |
| `correctness` | Does reasoning align with reference diagnostic reasoning? |
| `factual_accuracy` | Does the model avoid fabricating clinical facts? |
| `relevancy` | Is the answer relevant to the clinical question? |

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

## Configuration

Framework settings can be configured in `config.py`:

```python
# RAGAS settings
RAGAS_LLM_MODEL = "gpt-4o-mini"      # OpenAI model for RAGAS
RAGAS_EMBEDDING_MODEL = "text-embedding-3-small"
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# DeepEval settings
DEEPEVAL_MODEL = "gpt-4o-mini"       # Model for G-Eval
DEEPEVAL_THRESHOLD = 0.5             # Minimum threshold for passing
DEEPEVAL_INCLUDE_HALLUCINATION = True
DEEPEVAL_INCLUDE_RELEVANCY = True
```

## Environment Variables

For framework evaluations, set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```
