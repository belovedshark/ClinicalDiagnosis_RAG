# Evaluation Framework

Unified evaluation framework for comparing different clinical diagnosis model configurations.

## Overview

This framework provides a consistent interface for evaluating:

### Inference Evaluators

| Evaluator | Status | Description |
|-----------|--------|-------------|
| `rag` | ✅ Ready | RAG (Retrieval-Augmented Generation) - retrieves context from vector DB |
| `base_model` | ✅ Ready | Base LLM without retrieval |
| `finetuned` | ✅ Ready | Fine-tuned model without retrieval |
| `finetuned_rag` | ✅ Ready | Fine-tuned model with RAG retrieval |

### Framework Evaluation (Post-Inference)

| Command | Description |
|---------|-------------|
| `frameworks` | Fast batched evaluation combining RAGAS + DeepEval metrics (2 API calls per case) |

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

3. **OpenAI API Key**: Required for framework evaluation
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

### Running Inference Evaluations

From the project root directory:

```bash
# Run RAG evaluation
python -m evaluation.run_evaluation rag

# Run Base Model evaluation
python -m evaluation.run_evaluation base_model

# Run Fine-tuned evaluation
python -m evaluation.run_evaluation finetuned

# Run Fine-tuned + RAG evaluation
python -m evaluation.run_evaluation finetuned_rag

# Run all inference evaluators
python -m evaluation.run_evaluation all
```

### Running Framework Evaluation (RAGAS + DeepEval)

Framework evaluation runs on existing inference results to compute quality metrics:

```bash
# Basic usage (auto-detects model name from path)
python -m evaluation.run_evaluation frameworks --results evaluation/rag/results/inference_results.json
# → Output: rag_evaluation.json

# Specify custom output name
python -m evaluation.run_evaluation frameworks --results evaluation/rag/results/inference_results.json --name my_experiment
# → Output: my_experiment_evaluation.json

# Specify full output path
python -m evaluation.run_evaluation frameworks --results evaluation/rag/results/inference_results.json --output custom_path.json
```

#### Command Options

| Option | Description | Example |
|--------|-------------|---------|
| `--results`, `-r` | Path to inference_results.json | `--results evaluation/rag/results/inference_results.json` |
| `--name`, `-n` | Model name for output file | `--name rag_v2` |
| `--output`, `-o` | Full output path | `--output my_results.json` |

## Evaluation Methodology

### 2-Step Batched Evaluation

The framework uses an optimized 2-step approach that combines RAGAS and DeepEval methodologies:

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Extract Claims (RAGAS Faithfulness Methodology)     │
│                                                             │
│ Input:  "Dengue fever based on fever, rash, mosquito bite"  │
│ Output: ["Patient has dengue fever",                        │
│          "Symptoms include fever and rash",                 │
│          "Exposure to mosquito bite"]                       │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Verify Claims + Compute All Metrics                 │
│                                                             │
│ RAGAS Metrics:                                              │
│   • faithfulness (claims supported / total claims)          │
│   • answer_relevancy                                        │
│   • context_precision                                       │
│   • context_recall                                          │
│                                                             │
│ DeepEval Metrics:                                           │
│   • reasoning_coherence                                     │
│   • correctness                                             │
└─────────────────────────────────────────────────────────────┘
```

**Performance**: ~2 API calls per case = ~6 minutes for 63 cases (vs 2+ hours with separate frameworks)

## Framework Metrics

### RAGAS Metrics (Retrieval-Augmented Generation Assessment)

| Metric | Description | Score Range |
|--------|-------------|-------------|
| `faithfulness` | Is the answer factually grounded in the retrieved contexts? | 0.0 - 1.0 |
| `answer_relevancy` | Does the answer directly address the clinical question? | 0.0 - 1.0 |
| `context_precision` | How much of the retrieved context is actually relevant? | 0.0 - 1.0 |
| `context_recall` | Does the context contain the information needed for correct diagnosis? | 0.0 - 1.0 |

### DeepEval Metrics (G-Eval Methodology)

| Metric | Description | Score Range |
|--------|-------------|-------------|
| `reasoning_coherence` | Is the diagnostic reasoning logical and clinically sound? | 0.0 - 1.0 |
| `correctness` | Does the answer match the ground truth diagnosis? | 0.0 - 1.0 |

## Output

### Folder Structure

```
evaluation/
├── README.md                    # This file
├── EVALUATION_REPORT.md         # Comparison report of all models
├── test_cases.jsonl             # Input test cases (63 WHO clinical cases)
├── base_interface.py            # Abstract BaseEvaluator class
├── config.py                    # Shared configuration
├── utils.py                     # Data loading/saving utilities
├── run_evaluation.py            # Main entry point
│
├── rag/                         # RAG evaluator
│   ├── evaluator.py
│   └── results/
│       └── inference_results.json
│
├── base_model/                  # Base model evaluator
│   ├── evaluator.py
│   └── results/
│       └── inference_results.json
│
├── finetuned/                   # Fine-tuned evaluator
│   ├── evaluator.py
│   └── results/
│       └── inference_results.json
│
├── finetuned_rag/               # Fine-tuned + RAG evaluator
│   ├── evaluator.py
│   └── results/
│       └── inference_results.json
│
└── frameworks/                  # Framework evaluation
    ├── __init__.py
    ├── batched_evaluator.py     # Fast 2-step evaluator (RAGAS + DeepEval)
    ├── ragas_evaluator.py       # Standalone RAGAS (slower)
    ├── deepeval_evaluator.py    # Standalone DeepEval (slower)
    └── results/
        ├── rag_evaluation.json
        ├── base_model_evaluation.json
        ├── finetuned_evaluation.json
        └── finetuned_rag_evaluation.json
```

### Inference Results Format

```json
{
  "case_id": "who_case_001",
  "question": "A 21-year-old male presents with...",
  "contexts": ["retrieved context 1", "..."],
  "answer": "Dengue fever",
  "ground_truth": "Dengue fever",
  "metadata": {
    "model_type": "rag",
    "num_contexts": 5
  },
  "diagnostic_reasoning": "..."
}
```

### Framework Evaluation Output Format

```json
{
  "ragas": {
    "average_scores": {
      "faithfulness": 0.933,
      "answer_relevancy": 0.468,
      "context_precision": 0.679,
      "context_recall": 0.640
    },
    "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
    "description": "Retrieval-Augmented Generation Assessment"
  },
  "deepeval": {
    "average_scores": {
      "reasoning_coherence": 0.484,
      "correctness": 0.405
    },
    "metrics": ["reasoning_coherence", "correctness"],
    "description": "LLM Evaluation with G-Eval methodology"
  },
  "per_case_scores": [
    {
      "case_id": "who_case_001",
      "ragas": {
        "faithfulness": 1.0,
        "faithfulness_detail": "3 of 3 claims supported",
        "answer_relevancy": 1.0,
        "context_precision": 1.0,
        "context_recall": 1.0
      },
      "deepeval": {
        "reasoning_coherence": 1.0,
        "correctness": 1.0
      },
      "claims_extracted": 3,
      "explanation": "The diagnosis is well-supported..."
    }
  ],
  "metadata": {
    "num_cases": 63,
    "model": "gpt-4o-mini",
    "method": "2-step: extract_claims + verify_and_evaluate",
    "timestamp": "2026-01-26T..."
  }
}
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

## Configuration

Framework settings can be configured in `config.py`:

```python
# Model for framework evaluation (RAGAS + DeepEval)
DEEPEVAL_MODEL = "gpt-4o-mini"       # Cost-effective, good quality
DEEPEVAL_THRESHOLD = 0.5             # Minimum threshold for passing
```

## See Also

- [EVALUATION_REPORT.md](EVALUATION_REPORT.md) - Detailed comparison of all model configurations
