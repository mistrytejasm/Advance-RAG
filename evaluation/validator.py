"""
validator.py — Post-generation validation of evaluation samples.

Why a separate validator?
  The LLM validator (in llm_generator.py) checks that the LLM's raw JSON
  output is structurally correct.  THIS validator checks the full assembled
  evaluation record — including pipeline-level rules:

    1. No duplicate queries in the dataset being built this run.
    2. Query is not empty.
    3. Ground truth is not empty.
    4. query_type is one of the supported values.
    5. difficulty is one of the supported values.
    6. chunk_id exists and was the chunk we fetched from Mongo.
    7. Grounding check: ground_truth text must share at least one
       meaningful token with the chunk text (catches hallucinated answers).

The validator is stateful — it accumulates seen queries across the run
to detect duplicates.
"""

from evaluation.config import QUERY_TYPES, DIFFICULTY_LEVELS
from evaluation.logger import get_logger

logger = get_logger("validator")

# Tokens that appear in almost every sentence and are too common to use
# as a grounding signal.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "this", "that", "it",
    "its", "and", "or", "but", "not", "no", "if", "so",
}


class EvalValidator:
    """Stateful validator for assembled evaluation records."""

    def __init__(self) -> None:
        self._seen_queries: set[str] = set()

    def validate(self, record: dict, chunk_text: str) -> tuple[bool, str]:
        """
        Validate a fully assembled evaluation record.

        Args:
            record:     The assembled sample dict (all fields present).
            chunk_text: The original chunk text used to generate the sample.

        Returns:
            (is_valid, reason) —
                is_valid=True means the record is accepted.
                reason explains the rejection if is_valid=False.
        """
        # --- 1. Query not empty ---
        query = str(record.get("query", "")).strip()
        if not query:
            return False, "Query is empty."

        # --- 2. Ground truth not empty ---
        ground_truth = str(record.get("ground_truth", "")).strip()
        if not ground_truth:
            return False, "Ground truth is empty."

        # --- 3. Supported query_type ---
        if record.get("query_type") not in QUERY_TYPES:
            return False, f"Invalid query_type: '{record.get('query_type')}'."

        # --- 4. Supported difficulty ---
        if record.get("difficulty") not in DIFFICULTY_LEVELS:
            return False, f"Invalid difficulty: '{record.get('difficulty')}'."

        # --- 5. chunk_id present ---
        if not record.get("chunk_id"):
            return False, "chunk_id is missing."

        # --- 6. Duplicate query check ---
        normalised_query = query.lower()
        if normalised_query in self._seen_queries:
            return False, f"Duplicate query detected: '{query[:80]}…'"

        # --- 7. Grounding check ---
        # The ground truth must share at least 3 meaningful tokens with
        # the chunk text — a lightweight heuristic that catches completely
        # hallucinated answers without being overly strict.
        if not self._is_grounded(ground_truth, chunk_text):
            return False, (
                "Ground truth does not appear to be grounded in the chunk text "
                "(no meaningful token overlap found)."
            )

        # All checks passed — record the query as seen
        self._seen_queries.add(normalised_query)
        return True, "Valid."

    # ── Private helpers ───────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Extract non-stopword alphabetical tokens from text."""
        return {
            w for w in text.lower().split()
            if w.isalpha() and w not in _STOPWORDS and len(w) > 2
        }

    def _is_grounded(self, ground_truth: str, chunk_text: str, min_overlap: int = 3) -> bool:
        """
        Check that the ground truth shares at least `min_overlap` meaningful
        tokens with the chunk text.
        """
        gt_tokens    = self._tokenize(ground_truth)
        chunk_tokens = self._tokenize(chunk_text)
        overlap      = gt_tokens & chunk_tokens
        return len(overlap) >= min_overlap

    @property
    def seen_count(self) -> int:
        return len(self._seen_queries)
