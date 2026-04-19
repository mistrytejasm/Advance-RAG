"""
generate_eval_dataset.py -- Phase 8 Incremental Evaluation Dataset Generator.

Usage:
    python generate_eval_dataset.py [OPTIONS]

Options:
    --limit INT       Max evaluation samples to generate in this run. Default: 50.
    --batch_size INT  How many chunks to process per MongoDB batch. Default: 10.
    --append          Append to existing dataset (DEFAULT - always safe to pass).
    --document_id STR Scope to one document. Default: all documents.
    --reset           Delete the dataset file and reset all chunk flags. Regenerate from scratch.
    --dry_run         Simulate the run. Do not write any file or mark MongoDB. Print stats only.

Examples:
    # Incremental run -- process the next 50 un-evaluated chunks
    python generate_eval_dataset.py --limit 50 --append --batch_size 10

    # Reset and rebuild the entire dataset
    python generate_eval_dataset.py --reset --limit 200

    # Preview what would happen without writing anything
    python generate_eval_dataset.py --dry_run --limit 20

Pipeline Flow:
    MongoDB chunks
    -> detect unevaluated chunks (evaluation_generated = False / missing)
    -> batch-load chunks
    -> generate evaluation Q&A via Groq LLM
    -> validate the record (dedup, grounding, schema)
    -> append to data/eval_dataset.json (atomic per-record write)
    -> mark chunk as evaluated in MongoDB
    -> log pipeline statistics
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on the path so `evaluation` and `app` packages resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.config import (
    EVAL_DATASET_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LIMIT,
    MIN_CHUNK_LENGTH,
)
from evaluation.mongo_loader import MongoLoader
from evaluation.llm_generator import EvalLLMGenerator, EvalGenerationError
from evaluation.validator import EvalValidator
from evaluation.logger import get_logger, PipelineStats

logger = get_logger("generate_eval")
stats  = PipelineStats()


# ── Dataset file helpers ──────────────────────────────────────────────

def _load_existing_dataset(path: str) -> list[dict]:
    """Load existing JSON dataset (returns [] if file doesn't exist)."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error(f"Dataset at {path} is not a JSON array. Treating as empty.")
            return []
        logger.info(f"[Dataset] Loaded {len(data)} existing records from {path}.")
        return data
    except (json.JSONDecodeError, IOError) as exc:
        logger.error(f"[Dataset] Failed to read {path}: {exc}. Treating as empty.")
        return []


def _save_dataset(records: list[dict], path: str, dry_run: bool = False) -> None:
    """Write the full records list to disk atomically."""
    if dry_run:
        logger.info(f"[DRY RUN] Would write {len(records)} records to {path}.")
        return

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file then rename -- prevents corruption on failure
    tmp_path = p.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    tmp_path.replace(p)   # atomic on POSIX; near-atomic on Windows NTFS
    logger.debug(f"[Dataset] Saved {len(records)} records to {path}.")


def _reset_dataset(path: str, mongo: "MongoLoader", dry_run: bool) -> None:
    """Delete dataset file and clear all evaluation flags in MongoDB."""
    logger.warning("[RESET] Deleting eval dataset file and resetting all chunk flags.")

    if dry_run:
        logger.info("[DRY RUN] Would delete dataset file and reset MongoDB flags.")
        return

    p = Path(path)
    if p.exists():
        p.unlink()
        logger.info(f"[RESET] Deleted {path}.")

    count = mongo.reset_all()
    logger.info(f"[RESET] Reset {count} chunks to evaluation_generated=False.")


# ── Core record assembler ─────────────────────────────────────────────

def _assemble_record(chunk: dict, llm_output: dict) -> dict:
    """
    Merge the LLM output with pipeline metadata to produce a full record.
    This is the canonical dataset schema.
    page and section are top-level fields in the chunks collection (not nested in metadata).
    """
    return {
        "query":              llm_output["query"].strip(),
        "ground_truth":       llm_output["ground_truth"].strip(),
        "document_id":        chunk.get("document_id", ""),
        "chunk_id":           chunk.get("chunk_id", ""),
        "relevant_chunk_ids": [chunk.get("chunk_id", "")],
        "query_type":         llm_output["query_type"],
        "difficulty":         llm_output["difficulty"],
        "answer_type":        llm_output["answer_type"],   # added: drives Phase 9 eval strategy
        "source":             "synthetic",
        "page":               chunk.get("page"),          # top-level field
        "section":            chunk.get("section", ""),   # top-level field
        "generated_at":       datetime.now(timezone.utc).isoformat(),
    }


# ── Main pipeline ─────────────────────────────────────────────────────

def run(
    limit:       int,
    batch_size:  int,
    document_id: str | None,
    reset:       bool,
    dry_run:     bool,
) -> None:
    """
    Main pipeline entry point.

    Args:
        limit:       Total evaluation samples to generate this run.
        batch_size:  Chunks to load per MongoDB query.
        document_id: Scope to one document (None = all documents).
        reset:       If True, wipe the dataset and restart from zero.
        dry_run:     If True, simulate without writing anything.
    """
    mongo     = MongoLoader()
    generator = EvalLLMGenerator()
    validator = EvalValidator()

    # -- 0. Reset mode
    if reset:
        _reset_dataset(EVAL_DATASET_PATH, mongo, dry_run)

    # -- 1. Load existing dataset
    all_records = [] if reset and not dry_run else _load_existing_dataset(EVAL_DATASET_PATH)

    # Seed the validator with queries already in the dataset to prevent
    # duplicates across runs, not just within this run.
    for r in all_records:
        q = str(r.get("query", "")).strip().lower()
        if q:
            validator._seen_queries.add(q)

    # -- 2. Count available work
    total_unevaluated = mongo.count_unevaluated(document_id)
    logger.info(
        f"[Pipeline] {total_unevaluated} unevaluated chunks available. "
        f"Target: {limit} new samples."
    )

    if total_unevaluated == 0:
        logger.info("[Pipeline] No unevaluated chunks found. Nothing to do.")
        stats.summary(logger)
        mongo.close()
        return

    # -- 3. Batch processing loop
    generated_this_run = 0
    batch_offset       = 0
    # Tracks chunk_ids processed this run -- used in dry_run mode to prevent
    # re-fetching the same chunks that were not marked in MongoDB.
    dry_run_seen_ids: set[str] = set()

    while generated_this_run < limit:
        remaining  = limit - generated_this_run
        fetch_size = min(batch_size, remaining)

        logger.info(
            f"[Pipeline] Fetching batch (offset~{batch_offset} size={fetch_size}) ..."
        )
        # In dry_run mode, MongoDB is never updated, so we must exclude
        # already-seen chunks from the query ourselves to avoid infinite loop.
        raw_chunks = mongo.load_unevaluated_chunks(
            limit=fetch_size + len(dry_run_seen_ids) if dry_run else fetch_size,
            document_id=document_id,
        )
        if dry_run:
            chunks = [c for c in raw_chunks if c.get("chunk_id") not in dry_run_seen_ids][:fetch_size]
        else:
            chunks = raw_chunks
        stats.chunks_scanned += len(chunks)

        if not chunks:
            logger.info("[Pipeline] No more unevaluated chunks available. Done.")
            break

        # Maps chunk_id -> eval_skip_reason (None = successfully generated).
        # Using a dict makes the bulk_mark_evaluated call carry per-chunk context.
        newly_evaluated: dict[str, str | None] = {}

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "unknown")
            content  = (chunk.get("content") or "").strip()  # real field name

            # Track in memory for dry_run deduplication
            dry_run_seen_ids.add(chunk_id)

            # -- Length guard: skip header-only / title chunks.
            # Chunks shorter than MIN_CHUNK_LENGTH generate trivial questions
            # ("What is the title of the section?") that don't test RAG quality.
            # We mark them as evaluated immediately so they never resurface.
            if len(content) < MIN_CHUNK_LENGTH:
                logger.debug(
                    f"[Pipeline] Skipping short chunk {chunk_id[:8]}... "
                    f"({len(content)} chars < MIN_CHUNK_LENGTH={MIN_CHUNK_LENGTH})"
                )
                stats.chunks_skipped += 1
                newly_evaluated[chunk_id] = "short_chunk"
                continue

            try:
                # 3a. Generate from LLM
                logger.info(f"[Pipeline] Generating sample for chunk: {chunk_id[:8]}...")
                llm_output = generator.generate_eval_sample(chunk)

                # 3b. Assemble full record
                record = _assemble_record(chunk, llm_output)

                # 3c. Validate the assembled record
                is_valid, val_reason = validator.validate(record, content)
                if not is_valid:
                    logger.warning(
                        f"[Pipeline] Record rejected for chunk {chunk_id[:8]}...: {val_reason}"
                    )
                    stats.chunks_skipped += 1
                    newly_evaluated[chunk_id] = "validation_failed"
                    continue

                # 3d. Accumulate (None reason = success)
                all_records.append(record)
                newly_evaluated[chunk_id] = None
                generated_this_run += 1
                stats.samples_generated += 1
                stats.chunks_processed  += 1

                logger.info(
                    f"[Pipeline] [OK] Sample {generated_this_run}/{limit} -- "
                    f"type={record['query_type']} diff={record['difficulty']} | "
                    f"chunk={chunk_id[:8]}..."
                )

                # Save after every record (append-safe)
                _save_dataset(all_records, EVAL_DATASET_PATH, dry_run)

                if generated_this_run >= limit:
                    break

            except EvalGenerationError as exc:
                logger.error(
                    f"[Pipeline] Generation failed for chunk {chunk_id[:8]}...: {exc}"
                )
                stats.errors += 1
                newly_evaluated[chunk_id] = "llm_error"

            except Exception as exc:
                logger.exception(
                    f"[Pipeline] Unexpected error for chunk {chunk_id[:8]}...: {exc}"
                )
                stats.errors += 1
                newly_evaluated[chunk_id] = "unexpected_error"

        # 3e. Bulk-mark all processed chunks in MongoDB with their reasons
        if newly_evaluated and not dry_run:
            mongo.bulk_mark_evaluated(newly_evaluated)
        elif dry_run:
            logger.info(
                f"[DRY RUN] Would mark {len(newly_evaluated)} chunks as evaluated."
            )

        batch_offset += len(chunks)

        if len(chunks) < fetch_size:
            # Fewer chunks than requested -- collection is exhausted
            break

    # -- 4. Final save + summary
    _save_dataset(all_records, EVAL_DATASET_PATH, dry_run)

    logger.info(
        f"[Pipeline] Generated {generated_this_run} new samples. "
        f"Total dataset size: {len(all_records)} records."
    )
    stats.summary(logger)
    mongo.close()


# ── CLI ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 8 -- Incremental Evaluation Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max evaluation samples to generate (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Chunks per MongoDB batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=True,
        help="Append to existing dataset (default behaviour).",
    )
    parser.add_argument(
        "--document_id",
        type=str,
        default=None,
        help="Scope generation to one document ID (default: all documents).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Delete the dataset and reset all chunk flags before running.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Simulate the run without writing files or updating MongoDB.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE -- No files will be written or flags updated.")
        logger.info("=" * 60)

    if args.reset:
        logger.warning(
            "RESET MODE enabled. Existing dataset will be deleted and all "
            "chunk evaluation flags will be cleared."
        )

    logger.info(
        f"Starting eval generation: limit={args.limit} "
        f"batch_size={args.batch_size} "
        f"document_id={args.document_id or 'ALL'} "
        f"reset={args.reset} dry_run={args.dry_run}"
    )

    run(
        limit=args.limit,
        batch_size=args.batch_size,
        document_id=args.document_id,
        reset=args.reset,
        dry_run=args.dry_run,
    )
