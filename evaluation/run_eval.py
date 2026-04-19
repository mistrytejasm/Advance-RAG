"""
evaluation/run_eval.py  -- Phase 9 Evaluation Runner.

Calls the internal RAG pipeline (retrieval + generation) for every sample
in data/eval_dataset.json and saves the results to data/eval_results_raw.json.

Crucially, it saves results AFTER EVERY QUESTION so that a crash at question
45 does NOT lose the first 44 results.  It also skips questions that are
already present in the output file, making it safe to re-run (idempotent).

Usage:
    python evaluation/run_eval.py [OPTIONS]

Options:
    --limit   INT  Max questions to evaluate (default: all).
    --reset        Delete existing raw results and start fresh.

Examples:
    # Dry-run: evaluate first 5 questions only
    python evaluation/run_eval.py --limit 5

    # Full evaluation run
    python evaluation/run_eval.py

    # Restart from scratch
    python evaluation/run_eval.py --reset
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on sys.path so `app` and `evaluation` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.eval_config import (
    EVAL_DATASET_PATH,
    EVAL_RESULTS_RAW_PATH,
    EVAL_TOP_K,
    EVAL_RERANK_TOP_K,
    EVAL_MIN_SCORE,
    EVAL_REQUEST_DELAY_SEC,
)
from evaluation.logger import get_logger

logger = get_logger("run_eval")


# ── File helpers ──────────────────────────────────────────────────────

def _load_json(path: str) -> list[dict]:
    """Load a JSON array from disk. Returns [] if file missing or corrupt."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def _save_json(records: list[dict], path: str) -> None:
    """Atomically write a JSON array to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    tmp.replace(p)


# ── Single-question pipeline call ───────────────────────────────────

def _run_single(sample: dict) -> dict:
    """
    Run the full RAG pipeline for one eval sample.

    We import the pipeline lazily (inside the function) so that the heavy
    PaddleOCR / Pinecone / MongoDB startup happens only once, when the
    first question is processed -- not at module-import time.

    Returns a result dict that matches the RAGAS expected schema PLUS
    extra metadata fields for analysis.
    """
    from app.services.retrieval.retrieval_pipeline import run_retrieval_pipeline
    from app.generation.answer_service import generate_answer

    query       = sample["query"]
    document_id = sample.get("document_id")

    retrieval = run_retrieval_pipeline(
        query=query,
        document_id=document_id,
        top_k=EVAL_TOP_K,
        rerank_top_k=EVAL_RERANK_TOP_K,
        min_score=EVAL_MIN_SCORE,
    )

    full_response = generate_answer(
        query=query,
        retrieval_result=retrieval,
    )

    # Extract retrieved chunks' text -- RAGAS needs them as a list of strings
    retrieved_chunks = retrieval.get("results", [])
    contexts = [c.get("content", "") for c in retrieved_chunks]

    return {
        # RAGAS fields (exact names RAGAS expects)
        "user_input":         query,
        "response":           full_response.get("answer", ""),
        "retrieved_contexts": contexts,
        "reference":          sample.get("ground_truth", ""),

        # Extra metadata (for per-query analysis / debugging)
        "chunk_id":           sample.get("chunk_id"),
        "document_id":        document_id,
        "query_type":         sample.get("query_type"),
        "difficulty":         sample.get("difficulty"),
        "answer_type":        sample.get("answer_type"),
        "page":               sample.get("page"),
        "section":            sample.get("section"),
        "is_grounded":        full_response.get("is_grounded", False),
        "confidence":         full_response.get("confidence", 0.0),
        "total_latency_ms":   full_response.get("total_latency_ms", 0),
        "num_contexts":       len(contexts),
        "evaluated_at":       datetime.now(timezone.utc).isoformat(),
    }


# ── Main loop ─────────────────────────────────────────────────────────

def run(limit: int | None, reset: bool) -> None:
    # -- Load eval dataset
    dataset = _load_json(EVAL_DATASET_PATH)
    if not dataset:
        logger.error(f"[Runner] No samples found in {EVAL_DATASET_PATH}. Aborting.")
        return
    logger.info(f"[Runner] Loaded {len(dataset)} evaluation samples.")

    # -- Handle reset
    if reset:
        logger.warning("[Runner] RESET: deleting existing raw results.")
        p = Path(EVAL_RESULTS_RAW_PATH)
        if p.exists():
            p.unlink()

    # -- Load existing results (for resume / idempotency)
    existing_results = _load_json(EVAL_RESULTS_RAW_PATH)
    already_done = {r["chunk_id"] for r in existing_results if r.get("chunk_id")}
    logger.info(
        f"[Runner] {len(already_done)} questions already answered. "
        f"Will skip them."
    )

    # -- Apply limit
    to_process = [s for s in dataset if s.get("chunk_id") not in already_done]
    if limit is not None:
        to_process = to_process[:limit]

    if not to_process:
        logger.info("[Runner] Nothing to process. All questions already evaluated.")
        return

    logger.info(f"[Runner] Processing {len(to_process)} questions ...")
    results = list(existing_results)  # start with already-done results

    success_count = 0
    error_count   = 0

    for i, sample in enumerate(to_process, start=1):
        chunk_id = sample.get("chunk_id", "?")
        query    = sample.get("query", "")

        logger.info(
            f"[Runner] [{i}/{len(to_process)}] "
            f"chunk={chunk_id[:8]}... | q={query[:60]}"
        )

        try:
            result = _run_single(sample)
            results.append(result)
            success_count += 1

            # Save after EVERY question -- crash-safe
            _save_json(results, EVAL_RESULTS_RAW_PATH)
            logger.info(
                f"[Runner] [{i}/{len(to_process)}] OK "
                f"({result['num_contexts']} contexts, "
                f"{result['total_latency_ms']}ms)"
            )

        except Exception as exc:
            logger.exception(
                f"[Runner] [{i}/{len(to_process)}] FAILED for "
                f"chunk={chunk_id[:8]}...: {exc}"
            )
            error_count += 1

        # Rate-limit guard: small delay between pipeline calls
        if i < len(to_process):
            time.sleep(EVAL_REQUEST_DELAY_SEC)

    # -- Summary
    logger.info("=" * 60)
    logger.info("EVALUATION RUNNER SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total samples in dataset : {len(dataset)}")
    logger.info(f"  Processed this run       : {len(to_process)}")
    logger.info(f"  Succeeded                : {success_count}")
    logger.info(f"  Errors                   : {error_count}")
    logger.info(f"  Total in results file    : {len(results)}")
    logger.info(f"  Results saved to         : {EVAL_RESULTS_RAW_PATH}")
    logger.info("=" * 60)

    if success_count > 0:
        logger.info(
            "[Runner] Next step: run `python evaluation/scorer.py` "
            "to compute RAGAS metrics."
        )


# ── CLI ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 9 Evaluation Runner -- calls the RAG pipeline for each eval sample.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max questions to process this run (default: all remaining).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Delete existing raw results and start from scratch.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(limit=args.limit, reset=args.reset)
