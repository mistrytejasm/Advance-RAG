"""
evaluation/scorer.py  -- Phase 9 RAGAS Scorer.

Reads data/eval_results_raw.json (produced by run_eval.py), computes
RAGAS metrics using Groq as the judge LLM, and writes:
  - data/eval_scores.csv          -- per-question scores
  - data/evaluation_report.md     -- human-readable summary report

RAGAS Metrics used:
  faithfulness        -- Is the answer supported by the retrieved contexts?
  answer_relevancy    -- Does the answer address the question?
  context_precision   -- Are the top-ranked contexts relevant? (signal:noise)
  context_recall      -- Did retrieval find the chunks needed for the answer?

Usage:
    python evaluation/scorer.py [OPTIONS]

Options:
    --input   PATH   Path to raw results JSON (default: data/eval_results_raw.json)
    --limit   INT    Score only the first N records (useful for testing)
"""

import argparse
import json
import warnings
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Suppress deprecation warnings from RAGAS and LangChain wrappers.
# We intentionally use the old-style ragas.metrics._* API because it is the
# only one evaluate() accepts in RAGAS 0.4.x. Both libraries emit deprecation
# warnings for the wrappers we need -- suppress them to keep output clean.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*LangChain.*", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.eval_config import (
    GROQ_API_KEY,
    EVAL_JUDGE_MODEL,
    EVAL_RESULTS_RAW_PATH,
    EVAL_RESULTS_SCORES_PATH,
    EVAL_REPORT_PATH,
)
from evaluation.logger import get_logger

logger = get_logger("scorer")


# ── LLM + Embeddings setup ────────────────────────────────────────────

def _build_ragas_llm():
    """
    Build a RAGAS-compatible LangchainLLMWrapper using ChatGroq.
    LangchainLLMWrapper is what ragas.metrics (old-style) expect.
    """
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file before running the scorer."
        )

    llm = ChatGroq(
        model=EVAL_JUDGE_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )
    wrapped = LangchainLLMWrapper(llm)
    logger.info(f"[Scorer] RAGAS judge LLM ready: {EVAL_JUDGE_MODEL}")
    return wrapped


def _build_ragas_embeddings():
    """
    Build RAGAS-compatible embeddings using LangchainEmbeddingsWrapper.

    The old-style ragas.metrics._answer_relevance.AnswerRelevancy needs
    an embeddings object that exposes embed_query. The ragas.embeddings
    HuggingFaceEmbeddings lacks this -- we use LangchainEmbeddingsWrapper
    around a langchain_community.embeddings.HuggingFaceEmbeddings instead.
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings as LCEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper

        emb = LCEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
        )
        wrapped = LangchainEmbeddingsWrapper(emb)
        logger.info("[Scorer] Local BGE embeddings (LangchainEmbeddingsWrapper) ready.")
        return wrapped
    except Exception as exc:
        logger.warning(
            f"[Scorer] Could not load BGE embeddings ({exc}). "
            "AnswerRelevancy will be skipped."
        )
        return None


# ── Dataset builder ───────────────────────────────────────────────────

def _build_ragas_dataset(records: list[dict]):
    """
    Convert raw result records into a RAGAS EvaluationDataset.

    RAGAS expects each sample to have:
        user_input          str
        response            str
        retrieved_contexts  list[str]
        reference           str   (ground truth -- needed for recall)
    """
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    samples = []
    for r in records:
        if not r.get("response") or not r.get("retrieved_contexts"):
            logger.warning(
                f"[Scorer] Skipping record with missing response/contexts "
                f"(chunk_id={r.get('chunk_id', '?')[:8]}...)"
            )
            continue

        samples.append(SingleTurnSample(
            user_input=r["user_input"],
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r.get("reference", ""),
        ))

    if not samples:
        raise ValueError(
            "No valid samples found. "
            "Run `python evaluation/run_eval.py` first."
        )

    logger.info(f"[Scorer] Built RAGAS dataset with {len(samples)} valid samples.")
    return EvaluationDataset(samples=samples)


# ── Scoring ───────────────────────────────────────────────────────────

def _run_ragas(dataset, llm, embeddings) -> object:
    """
    Instantiate RAGAS metrics and run evaluate().

    We use the ragas.metrics._* internal classes (old-style). These are
    what ragas.evaluate() actually expects — the new collections API
    produces incompatible types that evaluate() rejects.
    """
    from ragas import evaluate
    from ragas.metrics._faithfulness       import Faithfulness
    from ragas.metrics._answer_relevance   import AnswerRelevancy
    from ragas.metrics._context_precision  import ContextPrecision
    from ragas.metrics._context_recall     import ContextRecall

    # Build each metric and inject the shared LLM.
    f  = Faithfulness(llm=llm)
    cp = ContextPrecision(llm=llm)
    cr = ContextRecall(llm=llm)

    metrics = [f, cp, cr]

    # AnswerRelevancy also needs embeddings (for semantic similarity).
    # We set strictness=1 because Groq does not support n>1 (fixes BadRequestError).
    if embeddings is not None:
        ar = AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1)
        metrics.insert(1, ar)
    else:
        logger.warning("[Scorer] AnswerRelevancy skipped — embeddings unavailable.")

    logger.info(
        f"[Scorer] Running RAGAS with {len(metrics)} metrics: "
        f"{[m.name for m in metrics]} | judge={EVAL_JUDGE_MODEL}"
    )

    # Concurrency control to avoid blowing past Groq Rate Limits
    from ragas.run_config import RunConfig
    
    run_config = RunConfig(
        timeout=180,           # Allow complex judgments to take their time
        max_workers=2,         # Low concurrency for Groq rate limits
        max_retries=5,
        max_wait=60            # Wait up to 60s between retries
    )

    return evaluate(
        dataset=dataset, 
        metrics=metrics, 
        run_config=run_config,
        raise_exceptions=False # Print warnings instead of crashing entire run
    )


# ── Output writers ────────────────────────────────────────────────────

def _save_scores_csv(
    ragas_result,
    raw_records: list[dict],
    path: str,
) -> pd.DataFrame:
    """
    Merge RAGAS per-question scores with pipeline metadata and save CSV.
    """
    df_scores = ragas_result.to_pandas()

    valid_records = [
        r for r in raw_records
        if r.get("response") and r.get("retrieved_contexts")
    ]
    meta_rows = [{
        "chunk_id":     r.get("chunk_id", ""),
        "query_type":   r.get("query_type", ""),
        "difficulty":   r.get("difficulty", ""),
        "answer_type":  r.get("answer_type", ""),
        "page":         r.get("page"),
        "section":      r.get("section", ""),
        "is_grounded":  r.get("is_grounded", False),
        "confidence":   r.get("confidence", 0.0),
        "latency_ms":   r.get("total_latency_ms", 0),
        "num_contexts": r.get("num_contexts", 0),
    } for r in valid_records]

    df_meta = pd.DataFrame(meta_rows)

    if len(df_scores) == len(df_meta):
        df = pd.concat(
            [df_scores.reset_index(drop=True), df_meta.reset_index(drop=True)],
            axis=1,
        )
    else:
        logger.warning(
            f"[Scorer] Row count mismatch (RAGAS={len(df_scores)}, "
            f"meta={len(df_meta)}). Saving scores without metadata."
        )
        df = df_scores

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"[Scorer] Per-question scores saved to {path}.")
    return df


def _write_report(df: pd.DataFrame, path: str, n_samples: int) -> None:
    """Write a structured markdown evaluation report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    metric_cols = [
        c for c in
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        if c in df.columns
    ]
    overall = {col: df[col].mean() for col in metric_cols}

    by_type: dict = defaultdict(dict)
    if "query_type" in df.columns:
        for col in metric_cols:
            for qt, val in df.groupby("query_type")[col].mean().items():
                by_type[qt][col] = round(val, 4)

    by_diff: dict = defaultdict(dict)
    if "difficulty" in df.columns:
        for col in metric_cols:
            for d, val in df.groupby("difficulty")[col].mean().items():
                by_diff[d][col] = round(val, 4)

    bottom_faith = pd.DataFrame()
    if "faithfulness" in df.columns and "user_input" in df.columns:
        bottom_faith = (
            df[["user_input", "faithfulness", "query_type", "difficulty"]]
            .nsmallest(5, "faithfulness")
        )

    metric_docs = {
        "faithfulness":      "Answer grounded in retrieved chunks (no hallucination)",
        "answer_relevancy":  "Answer directly addresses the question",
        "context_precision": "Signal-to-noise ratio of retrieved chunks",
        "context_recall":    "Chunks retrieved cover the ground truth",
    }

    lines = [
        "# Phase 9 -- RAG Evaluation Report",
        "",
        f"**Generated:** {now}  ",
        f"**Samples evaluated:** {n_samples}  ",
        f"**Judge model:** `{EVAL_JUDGE_MODEL}`  ",
        "",
        "---",
        "",
        "## Overall RAGAS Scores",
        "",
        "| Metric | Score | Meaning |",
        "|--------|-------|---------|",
    ]
    for col in metric_cols:
        lines.append(
            f"| **{col}** | `{round(overall[col], 4)}` "
            f"| {metric_docs.get(col, '')} |"
        )

    lines += [
        "",
        "> **Interpretation:** 0.0 = worst, 1.0 = perfect.",
        "> - Faithfulness < 0.7 means high hallucination risk.",
        "> - Context Recall < 0.6 means retrieval is missing relevant chunks.",
        "",
        "---",
        "",
        "## Breakdown by Query Type",
        "",
    ]
    if by_type:
        lines += [
            "| Query Type | " + " | ".join(metric_cols) + " |",
            "|" + "---------|" * (len(metric_cols) + 1),
        ]
        for qt, scores in sorted(by_type.items()):
            row = " | ".join(str(scores.get(c, "N/A")) for c in metric_cols)
            lines.append(f"| {qt} | {row} |")

    lines += ["", "---", "", "## Breakdown by Difficulty", ""]
    if by_diff:
        lines += [
            "| Difficulty | " + " | ".join(metric_cols) + " |",
            "|" + "---------|" * (len(metric_cols) + 1),
        ]
        for d, scores in sorted(by_diff.items()):
            row = " | ".join(str(scores.get(c, "N/A")) for c in metric_cols)
            lines.append(f"| {d} | {row} |")

    if not bottom_faith.empty:
        lines += [
            "",
            "---",
            "",
            "## Lowest Faithfulness Samples (Top Hallucination Risk)",
            "",
            "| Question | Faithfulness | Type | Difficulty |",
            "|----------|-------------|------|------------|",
        ]
        for _, row in bottom_faith.iterrows():
            q  = str(row.get("user_input", ""))[:70]
            fv = round(row.get("faithfulness", 0), 4)
            qt = row.get("query_type", "")
            d  = row.get("difficulty", "")
            lines.append(f"| {q} | `{fv}` | {qt} | {d} |")

    lines += [
        "",
        "---",
        "",
        "## Next Steps",
        "",
        "- **Faithfulness < 0.7**: Tighten grounding instructions in `llm_generator.py`.",
        "- **Context Recall < 0.6**: Increase `top_k` or lower `min_score` threshold.",
        "- **Context Precision < 0.6**: Raise `min_score` or tune reranker.",
        "- **Answer Relevancy < 0.7**: Review the query-understanding / rewrite layer.",
        "",
        f"*Full per-question scores: `{EVAL_RESULTS_SCORES_PATH}`*",
    ]

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"[Scorer] Evaluation report saved to {path}.")


# ── Main ──────────────────────────────────────────────────────────────

def run(input_path: str, limit: int | None) -> None:
    p = Path(input_path)
    if not p.exists():
        logger.error(
            f"[Scorer] Raw results file not found: {input_path}. "
            "Run `python evaluation/run_eval.py` first."
        )
        return

    with p.open("r", encoding="utf-8") as f:
        raw_records = json.load(f)

    if limit:
        raw_records = raw_records[:limit]

    logger.info(f"[Scorer] Loaded {len(raw_records)} raw results from {input_path}.")

    llm        = _build_ragas_llm()
    embeddings = _build_ragas_embeddings()
    dataset    = _build_ragas_dataset(raw_records)

    logger.info("[Scorer] Starting RAGAS evaluation (this may take a few minutes)...")
    ragas_result = _run_ragas(dataset, llm, embeddings)

    df = _save_scores_csv(ragas_result, raw_records, EVAL_RESULTS_SCORES_PATH)
    _write_report(df, EVAL_REPORT_PATH, n_samples=len(dataset.samples))

    metric_cols = [
        c for c in
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        if c in df.columns
    ]
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION COMPLETE")
    print("=" * 60)
    for col in metric_cols:
        print(f"  {col:<25}: {df[col].mean():.4f}")
    print(f"\n  Full report  : {EVAL_REPORT_PATH}")
    print(f"  Per-query    : {EVAL_RESULTS_SCORES_PATH}")
    print("=" * 60 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 9 RAGAS Scorer -- compute RAG evaluation metrics.",
    )
    parser.add_argument(
        "--input", type=str, default=EVAL_RESULTS_RAW_PATH,
        help=f"Path to raw results JSON (default: {EVAL_RESULTS_RAW_PATH}).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Score only the first N records (default: all).",
    )
    args = parser.parse_args()
    run(input_path=args.input, limit=args.limit)
