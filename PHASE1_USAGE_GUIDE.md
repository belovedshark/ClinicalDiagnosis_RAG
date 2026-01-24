# Phase 1 Implementation: RAG Inference Pipeline

## Overview

Phase 1 has been successfully implemented! This phase runs your RAG system on all 48 WHO clinical cases from the evaluation dataset and collects the necessary data for RAGAS evaluation.

## What Was Implemented

### 1. Core Modules Created

#### **rag_evaluation/config/evaluation_config.py**
- Configuration settings for the evaluation pipeline
- Paths to data files and results
- RAG system parameters (top-k retrieval, device settings)
- Checkpoint and save intervals

#### **rag_evaluation/data_preparation.py**
- Functions to load and parse the JSONL evaluation dataset
- Case preparation and validation
- Results saving and checkpoint management
- Handles the duplicate case_prompt field in your data

#### **rag_evaluation/rag_inference.py**
- Main inference pipeline class
- Runs RAG system on each case
- Collects contexts, similarity scores, and generated diagnoses
- Implements checkpoint recovery for long-running evaluations
- Saves results in RAGAS-compatible format

#### **rag_evaluation/utils/evaluation_utils.py**
- Helper functions for result analysis
- Exact and partial match accuracy calculations
- Retrieval quality metrics
- Summary report generation

### 2. Runner Scripts

#### **run_rag_inference.py**
Main script to execute the inference pipeline

#### **analyze_results.py**
Script to analyze and visualize results after inference

## How to Use

### Step 1: Ensure Qdrant is Running

Your RAG system requires Qdrant vector database. Make sure it's running:

```bash
# Check if Qdrant is running
curl http://localhost:6333

# If not, start Qdrant (adjust based on your setup)
docker start qdrant  # if using Docker
# OR
qdrant-server  # if installed locally
```

### Step 2: Install Dependencies

```bash
pip install tqdm pillow
# Other dependencies should already be installed
```

### Step 3: Run the Inference Pipeline

```bash
cd /Users/macintoshhd/Downloads/Project-Dr.Khanh/Clinical-RAG
python run_rag_inference.py
```

This will:
- Load all 48 WHO cases from `evaluate_RAG.jsonl`
- Run your RAG system on each case
- Retrieve top-5 contexts from the vector database
- Generate diagnoses using your Gemma model
- Save results to `rag_evaluation/results/rag_inference_results.json`
- Create checkpoints every 5 cases (resumable if interrupted)

### Step 4: Analyze Results

```bash
python analyze_results.py
```

This will display:
- Summary statistics (accuracy, retrieval quality)
- Sample results from the evaluation
- Best and worst performing cases
- Match distribution analysis

## Output Format

The inference pipeline produces `rag_inference_results.json` with this structure:

```json
[
  {
    "case_id": "who_case_001",
    "question": "A 21-year-old male presents with 4 days of high fever...",
    "contexts": [
      "Retrieved context chunk 1...",
      "Retrieved context chunk 2...",
      ...
    ],
    "answer": "Dengue fever",
    "ground_truth": "Dengue fever",
    "retrieval_metadata": {
      "similarity_scores": [0.891, 0.856, 0.823, 0.789, 0.745],
      "source_documents": ["doc1.md", "doc2.md", ...],
      "num_contexts": 5
    },
    "diagnostic_reasoning": "Expert reasoning from dataset..."
  },
  ...
]
```

This format is **ready for RAGAS evaluation** in Phase 2!

## Configuration Options

Edit `rag_evaluation/config/evaluation_config.py` to customize:

```python
TOP_K_RETRIEVAL = 5          # Number of contexts to retrieve
RAG_DEVICE = "auto"          # Device: "auto", "cuda", "mps", "cpu"
SAVE_INTERVAL = 5            # Checkpoint frequency
```

## Features

### ✅ Checkpoint Recovery
If the process is interrupted (Ctrl+C, error, etc.), simply run it again:
```bash
python run_rag_inference.py
```
It will automatically resume from the last saved checkpoint.

### ✅ Progress Tracking
Real-time progress bar shows:
- Current case being processed
- Case ID and question preview
- Retrieved contexts count
- Generated diagnosis vs. ground truth

### ✅ Error Handling
If a case fails:
- Error is logged
- Case is marked with ERROR status
- Pipeline continues with remaining cases

### ✅ Comprehensive Metadata
Each result includes:
- Similarity scores for each retrieved context
- Source document information
- Number of contexts retrieved
- Original diagnostic reasoning from dataset

## Expected Runtime

- **With GPU**: ~5-10 minutes for 48 cases
- **With CPU**: ~15-30 minutes for 48 cases
- **With MPS (Mac M1/M2)**: ~10-20 minutes for 48 cases

Times vary based on:
- Model size (gemma-2b vs gemma-7b)
- Hardware specifications
- Retrieval performance

## Troubleshooting

### Problem: Import errors

```bash
# Make sure you're in the project root
cd /Users/macintoshhd/Downloads/Project-Dr.Khanh/Clinical-RAG
python run_rag_inference.py
```

### Problem: Qdrant connection failed

```
Error: Could not connect to Qdrant at http://localhost:6333
```

**Solution**: Start Qdrant service
```bash
docker start qdrant
# or check QDRANT_URL environment variable
```

### Problem: Out of memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Use CPU or reduce batch size
```python
# In evaluation_config.py
RAG_DEVICE = "cpu"
TOP_K_RETRIEVAL = 3  # Reduce from 5 to 3
```

### Problem: Model loading slow

**Solution**: Models are cached after first run. Subsequent runs will be faster.

## Next Steps: Phase 2

After Phase 1 completes successfully, you're ready for Phase 2:

1. **RAGAS Framework Setup**
   ```bash
   pip install ragas langchain openai  # or anthropic for Claude
   ```

2. **RAGAS Evaluation**
   - Use the generated `rag_inference_results.json`
   - Implement RAGAS metrics:
     - Context Relevance
     - Faithfulness
     - Answer Relevance

3. **Follow the plan**: See `plans/rag_evaluation_with_ragas_framework.md` for Phase 2 details

## Validation

Before proceeding to Phase 2, verify:

✅ All 48 cases processed successfully
✅ Results file exists: `rag_evaluation/results/rag_inference_results.json`
✅ Each result has non-empty contexts and answer
✅ Similarity scores are reasonable (typically 0.5-1.0)
✅ Run `python analyze_results.py` to see summary

## Files Structure

```
Clinical-RAG/
├── run_rag_inference.py              # ← Run this to start
├── analyze_results.py                # ← Run this to analyze
├── evaluate_RAG.jsonl                # Input: WHO cases
├── rag_evaluation/
│   ├── config/
│   │   └── evaluation_config.py     # Configuration
│   ├── data_preparation.py          # Data loading
│   ├── rag_inference.py             # Main pipeline
│   ├── utils/
│   │   └── evaluation_utils.py      # Analysis tools
│   └── results/
│       ├── rag_inference_results.json      # ← Output here
│       └── rag_inference_checkpoint.json   # Checkpoints
└── rag/                              # Your existing RAG system
    ├── pipeline.py
    ├── retriever.py
    └── generator.py
```

## Summary

Phase 1 is **complete and ready to run**! 🎉

The implementation:
- ✅ Integrates seamlessly with your existing RAG system in the `rag/` folder
- ✅ Handles the WHO evaluation dataset format
- ✅ Collects all data needed for RAGAS evaluation
- ✅ Includes checkpoint recovery and error handling
- ✅ Provides analysis tools for result inspection
- ✅ Produces RAGAS-compatible output format

**To start Phase 1 evaluation:**
```bash
python run_rag_inference.py
```

Good luck with your evaluation! 🚀
