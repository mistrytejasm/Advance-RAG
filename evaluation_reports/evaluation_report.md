# Phase 9 -- RAG Evaluation Report

**Generated:** 2026-04-19 07:02 UTC  
**Samples evaluated:** 34  
**Judge model:** `llama-3.3-70b-versatile`  

---

## Overall RAGAS Scores

| Metric | Score | Meaning |
|--------|-------|---------|
| **faithfulness** | `0.8242` | Answer grounded in retrieved chunks (no hallucination) |
| **answer_relevancy** | `0.8423` | Answer directly addresses the question |
| **context_precision** | `0.9881` | Signal-to-noise ratio of retrieved chunks |
| **context_recall** | `0.9773` | Chunks retrieved cover the ground truth |

> **Interpretation:** 0.0 = worst, 1.0 = perfect.
> - Faithfulness < 0.7 means high hallucination risk.
> - Context Recall < 0.6 means retrieval is missing relevant chunks.

---

## Breakdown by Query Type

| Query Type | faithfulness | answer_relevancy | context_precision | context_recall |
|---------|---------|---------|---------|---------|
| definition | 0.875 | 0.8276 | 0.9667 | 1.0 |
| explanation | 0.6529 | 0.7416 | 1.0 | 0.9167 |
| list | 0.8968 | 0.9343 | 1.0 | 1.0 |
| procedural | nan | 0.7935 | nan | 1.0 |

---

## Breakdown by Difficulty

| Difficulty | faithfulness | answer_relevancy | context_precision | context_recall |
|---------|---------|---------|---------|---------|
| easy | 0.8214 | 0.8425 | 0.9861 | 0.9737 |
| medium | 0.8381 | 0.8412 | 1.0 | 1.0 |

---

## Lowest Faithfulness Samples (Top Hallucination Risk)

| Question | Faithfulness | Type | Difficulty |
|----------|-------------|------|------------|
| What is the rule for combining data size and model size? | `0.0` | explanation | easy |
| Where are the parameters stored in practice according to the pro tip? | `0.6667` | definition | easy |
| What topics are covered in Week 9: LLM Fundamentals? | `0.7` | list | easy |
| Why is offloading important when the GPU has only 4 GB of VRAM? | `0.7143` | explanation | medium |
| Why is establishing a baseline important before fine‑tuning an adapter | `0.75` | explanation | easy |

---

## Next Steps

- **Faithfulness < 0.7**: Tighten grounding instructions in `llm_generator.py`.
- **Context Recall < 0.6**: Increase `top_k` or lower `min_score` threshold.
- **Context Precision < 0.6**: Raise `min_score` or tune reranker.
- **Answer Relevancy < 0.7**: Review the query-understanding / rewrite layer.

*Full per-question scores: `evaluation_reports/eval_scores.csv`*