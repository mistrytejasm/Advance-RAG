"""
config.py — All Phase 8 evaluation generation settings.

Every value is read from the environment (with a safe default) so that
nothing is hardcoded and the pipeline works across dev / CI / prod by
just overriding env vars — no code changes required.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Groq LLM (shared with Phase 7 settings) ──────────────────────────
GROQ_API_KEY            = os.getenv("GROQ_API_KEY")
GROQ_MODEL              = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ── LLM generation knobs ──────────────────────────────────────────────
EVAL_LLM_TEMPERATURE    = float(os.getenv("EVAL_LLM_TEMPERATURE",    "0.7"))   # Higher = more creative questions
EVAL_LLM_MAX_TOKENS     = int(os.getenv("EVAL_LLM_MAX_TOKENS",       "512"))
EVAL_LLM_TIMEOUT        = int(os.getenv("EVAL_LLM_TIMEOUT",          "30"))
EVAL_LLM_MAX_RETRIES    = int(os.getenv("EVAL_LLM_MAX_RETRIES",      "3"))
EVAL_LLM_RETRY_DELAY    = float(os.getenv("EVAL_LLM_RETRY_DELAY",    "2.0"))   # seconds

# ── MongoDB ───────────────────────────────────────────────────────────
MONGODB_URI             = os.getenv("MONGODB_URI")
DATABASE_NAME           = os.getenv("DATABASE_NAME",                  "rag_db")
CHUNKS_COLLECTION       = os.getenv("CHUNKS_COLLECTION",              "chunks")

# ── Output ────────────────────────────────────────────────────────────
EVAL_DATASET_PATH       = os.getenv("EVAL_DATASET_PATH", "evaluation_data/eval_dataset.json")

# ── Pipeline defaults ─────────────────────────────────────────────────
DEFAULT_BATCH_SIZE      = int(os.getenv("EVAL_BATCH_SIZE",            "10"))
DEFAULT_LIMIT           = int(os.getenv("EVAL_DEFAULT_LIMIT",         "50"))

# ── Supported query types and difficulty levels ───────────────────────
QUERY_TYPES = [
    "definition",
    "procedural",
    "explanation",
    "comparison",
    "troubleshooting",
    "list",
    "conceptual",
]

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

# How the ground_truth answer was derived from the chunk.
# This drives downstream evaluation strategy in Phase 9:
#   extractive  -> direct span from chunk  (low hallucination risk)
#   abstractive -> paraphrase / summary    (medium risk)
#   reasoning   -> inference from chunk    (higher risk, logic checks needed)
#   multi_hop   -> synthesised from 2+     (hardest to auto-evaluate)
ANSWER_TYPES = ["extractive", "abstractive", "reasoning", "multi_hop"]

# Minimum chunk content length to evaluate.
# Chunks shorter than this are almost always section headers or titles
# (e.g. "Section: PART 2: Deep Learning") that generate trivial questions
# like "What is the title of the section?" -- useless for evaluating RAG quality.
# 120 chars empirically filters headers (<80 chars) while preserving substantive paragraphs.
MIN_CHUNK_LENGTH        = int(os.getenv("EVAL_MIN_CHUNK_LENGTH",      "120"))

# The JSON keys the LLM must return
REQUIRED_LLM_KEYS = {"query", "ground_truth", "query_type", "difficulty", "answer_type"}
