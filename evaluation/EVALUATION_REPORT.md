# Clinical Diagnosis RAG - Evaluation Report

## Overview

Comparison of 4 model configurations evaluated on 63 WHO clinical diagnosis cases using RAGAS and DeepEval metrics.

---

## Model Configurations

### 1. RAG (Retrieval-Augmented Generation)
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Query     │ ──► │  Retriever  │ ──► │  Base LLM   │ ──► Diagnosis
│ (Symptoms)  │     │ (Vector DB) │     │ + Context   │
└─────────────┘     └─────────────┘     └─────────────┘
```
- **How it works**: Retrieves relevant medical literature from vector database, then uses base LLM to generate diagnosis based on retrieved context
- **Strengths**: Access to external knowledge, grounded responses
- **Weaknesses**: Dependent on retrieval quality

### 2. Base Model (No Retrieval)
```
┌─────────────┐     ┌─────────────┐
│   Query     │ ──► │  Base LLM   │ ──► Diagnosis
│ (Symptoms)  │     │  (No RAG)   │
└─────────────┘     └─────────────┘
```
- **How it works**: Uses only the LLM's pre-trained knowledge to generate diagnosis
- **Strengths**: Fast, no retrieval latency
- **Weaknesses**: Limited to training data, may hallucinate

### 3. Finetuned (No Retrieval)
```
┌─────────────┐     ┌─────────────┐
│   Query     │ ──► │ Finetuned   │ ──► Diagnosis
│ (Symptoms)  │     │    LLM      │
└─────────────┘     └─────────────┘
```
- **How it works**: LLM fine-tuned on clinical diagnosis data, no retrieval
- **Strengths**: Domain-specific knowledge embedded in weights
- **Weaknesses**: May overfit, lose generalization

### 4. Finetuned + RAG
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Query     │ ──► │  Retriever  │ ──► │ Finetuned   │ ──► Diagnosis
│ (Symptoms)  │     │ (Vector DB) │     │ LLM+Context │
└─────────────┘     └─────────────┘     └─────────────┘
```
- **How it works**: Combines fine-tuned LLM with RAG retrieval
- **Strengths**: Domain knowledge + external context
- **Weaknesses**: More complex, potential conflicts between learned and retrieved knowledge

---

## Metrics Explained

### RAGAS Framework (Retrieval-Augmented Generation Assessment)

#### 1. Faithfulness
```
Question: Is the answer factually grounded in the retrieved contexts?

Example:
  Context: "Dengue fever presents with high fever, rash, and joint pain"
  Answer: "The patient has dengue fever based on fever and rash"
  
  ✓ Faithful - claims supported by context
  
  Context: "Dengue fever presents with high fever, rash, and joint pain"  
  Answer: "The patient has malaria based on the symptoms"
  
  ✗ Not Faithful - diagnosis not supported by context
```
- **Score 1.0**: All claims in the answer are supported by context
- **Score 0.0**: Answer contradicts or is unsupported by context

#### 2. Answer Relevancy
```
Question: Does the answer directly address the clinical question?

Example:
  Question: "Patient has fever, rash after mosquito bite. Diagnosis?"
  Answer: "Dengue fever"
  
  ✓ Relevant - directly answers the diagnostic question
  
  Question: "Patient has fever, rash after mosquito bite. Diagnosis?"
  Answer: "The patient should drink more water"
  
  ✗ Not Relevant - doesn't provide a diagnosis
```
- **Score 1.0**: Answer perfectly addresses the question
- **Score 0.0**: Answer is completely off-topic

#### 3. Context Precision
```
Question: How much of the retrieved context is actually relevant?

Example:
  Question: "Symptoms of dengue?"
  Retrieved Contexts:
    [1] "Dengue causes fever and rash" ✓ Relevant
    [2] "Malaria is transmitted by mosquitoes" ✗ Irrelevant
    [3] "Dengue can cause joint pain" ✓ Relevant
    
  Context Precision = 2/3 = 0.67
```
- **Score 1.0**: All retrieved contexts are relevant
- **Score 0.0**: No retrieved contexts are relevant
- **Note**: Models without RAG score ~0 (no context retrieved)

#### 4. Context Recall
```
Question: Does the context contain ALL information needed for correct diagnosis?

Example:
  Ground Truth: "Dengue fever" (requires: fever + rash + mosquito exposure)
  Context contains: fever ✓, rash ✓, mosquito exposure ✓
  
  Context Recall = 1.0 (all needed info present)
  
  Context contains: fever ✓, rash ✓, mosquito exposure ✗
  
  Context Recall = 0.67 (missing key info)
```
- **Score 1.0**: Context has everything needed for correct answer
- **Score 0.0**: Context missing critical information

### DeepEval Framework (G-Eval Methodology)

#### 5. Reasoning Coherence
```
Question: Is the diagnostic reasoning logical and clinically sound?

Example of Good Reasoning (Score: 1.0):
  "The patient presents with high fever, retro-orbital pain, and rash 
   following mosquito exposure in a tropical region. These symptoms 
   are characteristic of dengue fever. The combination of fever with 
   rash and mosquito vector exposure strongly suggests arboviral 
   infection, with dengue being most likely given the symptom profile."

Example of Poor Reasoning (Score: 0.0):
  "Dengue because patient is sick"
```
- **Score 1.0**: Clear, logical clinical reasoning
- **Score 0.0**: No reasoning or illogical reasoning

#### 6. Correctness
```
Question: Does the diagnosis match the ground truth?

Example:
  Ground Truth: "Dengue fever"
  Model Answer: "Dengue fever" → Score: 1.0 (exact match)
  Model Answer: "Dengue" → Score: 1.0 (equivalent)
  Model Answer: "Severe dengue" → Score: 0.5 (related)
  Model Answer: "Malaria" → Score: 0.0 (wrong)
```
- **Score 1.0**: Exact or clinically equivalent match
- **Score 0.5**: Partially correct (related condition)
- **Score 0.0**: Completely wrong diagnosis

---

## Summary Comparison

### RAGAS Metrics (Retrieval-Augmented Generation Assessment)

| Metric | RAG | Base Model | Finetuned | Finetuned+RAG | Best |
|--------|-----|------------|-----------|---------------|------|
| **Faithfulness** | 0.933 | 0.921 | 0.683 | 0.857 | RAG |
| **Answer Relevancy** | 0.468 | 0.349 | 0.214 | 0.341 | RAG |
| **Context Precision** | 0.679 | 0.016 | 0.016 | 0.640 | RAG |
| **Context Recall** | 0.640 | 0.016 | 0.016 | 0.608 | RAG |

### DeepEval Metrics (LLM Evaluation with G-Eval)

| Metric | RAG | Base Model | Finetuned | Finetuned+RAG | Best |
|--------|-----|------------|-----------|---------------|------|
| **Reasoning Coherence** | 0.484 | 0.349 | 0.222 | 0.349 | RAG |
| **Correctness** | 0.405 | 0.310 | 0.143 | 0.302 | RAG |

---

## Visual Comparison

### Faithfulness (Is the answer grounded in context?)

```
RAG           ████████████████████████████████████████████████░░░░ 93.3%
Base Model    ██████████████████████████████████████████████░░░░░░ 92.1%
Finetuned+RAG █████████████████████████████████████████░░░░░░░░░░░ 85.7%
Finetuned     ██████████████████████████████████░░░░░░░░░░░░░░░░░░ 68.3%
```

### Answer Relevancy (Is the answer relevant to the question?)

```
RAG           ███████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 46.8%
Base Model    █████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 34.9%
Finetuned+RAG █████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 34.1%
Finetuned     ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 21.4%
```

### Context Precision (How relevant is the retrieved context?)

```
RAG           ██████████████████████████████████░░░░░░░░░░░░░░░░░░ 67.9%
Finetuned+RAG ████████████████████████████████░░░░░░░░░░░░░░░░░░░░ 64.0%
Base Model    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.6%
Finetuned     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.6%
```

### Context Recall (Does context contain needed info?)

```
RAG           ████████████████████████████████░░░░░░░░░░░░░░░░░░░░ 64.0%
Finetuned+RAG ██████████████████████████████░░░░░░░░░░░░░░░░░░░░░░ 60.8%
Base Model    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.6%
Finetuned     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.6%
```

### Reasoning Coherence (Is the reasoning logical?)

```
RAG           ████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 48.4%
Base Model    █████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 34.9%
Finetuned+RAG █████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 34.9%
Finetuned     ███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 22.2%
```

### Correctness (Does answer match ground truth?)

```
RAG           ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 40.5%
Base Model    ███████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 31.0%
Finetuned+RAG ███████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 30.2%
Finetuned     ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 14.3%
```

---

## Key Findings

### 1. RAG Outperforms All Other Configurations
- **Best on all 6 metrics**
- Highest faithfulness (93.3%) - answers well-grounded in context
- Highest correctness (40.5%) - most accurate diagnoses

### 2. Context Retrieval is Critical
- Models without RAG (Base Model, Finetuned) score near 0% on context metrics
- This is expected since they have no retrieval component
- RAG configurations show ~65% context quality

### 3. Fine-tuning Alone Hurts Performance
- Finetuned model performs **worst** across all metrics
- Suggests fine-tuning without RAG may cause overfitting or loss of generalization
- Fine-tuning + RAG recovers most of the performance

### 4. Reasoning Quality Needs Improvement
- All models score below 50% on reasoning coherence
- Correctness ranges from 14% to 40%
- Suggests room for improvement in diagnostic reasoning

---

## Recommendations

1. **Use RAG for Production**: RAG configuration shows best overall performance
2. **Investigate Fine-tuning Strategy**: Current fine-tuning degrades performance
3. **Improve Retrieval Quality**: Context precision/recall at ~65% suggests retrieval can be optimized
4. **Enhance Reasoning**: Consider chain-of-thought prompting or reasoning-focused fine-tuning

---

## Methodology

### Evaluation Framework
- **RAGAS**: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- **DeepEval**: Reasoning Coherence, Correctness (using G-Eval methodology)

### Approach
- 2-step batched evaluation (extract claims → verify + evaluate)
- Model: GPT-4o-mini
- 63 WHO clinical diagnosis test cases

### Metrics Explained

| Metric | Description | Scale |
|--------|-------------|-------|
| Faithfulness | Is the answer factually grounded in retrieved contexts? | 0-1 |
| Answer Relevancy | Does the answer address the clinical question? | 0-1 |
| Context Precision | How much retrieved context is relevant? | 0-1 |
| Context Recall | Does context contain info needed for diagnosis? | 0-1 |
| Reasoning Coherence | Is the diagnostic reasoning logical? | 0-1 |
| Correctness | Does the answer match the ground truth diagnosis? | 0-1 |
