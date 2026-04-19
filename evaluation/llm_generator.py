"""
llm_generator.py — LLM-based evaluation question & ground truth generator.

Each call to generate_eval_sample() produces exactly ONE evaluation record
for the given chunk by asking the LLM to:
  1. Choose an appropriate query type for the chunk's content.
  2. Write a natural-language question that can be answered from the chunk.
  3. Write the ideal ground-truth answer using only the chunk's text.
  4. Assign a difficulty level.

The LLM must respond with a JSON object only — no prose, no markdown.
We enforce this by specifying it in the system prompt AND validating the
parsed JSON before accepting it.

Retry logic mirrors Phase 7's LLMGenerator — exponential backoff for
rate limits and 5xx errors, hard timeout, configurable retries.
"""

import json
import random
import time

from groq import Groq, APIStatusError, APIConnectionError, RateLimitError

from evaluation.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    EVAL_LLM_TEMPERATURE,
    EVAL_LLM_MAX_TOKENS,
    EVAL_LLM_TIMEOUT,
    EVAL_LLM_MAX_RETRIES,
    EVAL_LLM_RETRY_DELAY,
    QUERY_TYPES,
    DIFFICULTY_LEVELS,
    ANSWER_TYPES,
    REQUIRED_LLM_KEYS,
)
from evaluation.logger import get_logger

logger = get_logger("llm_generator")


class EvalGenerationError(Exception):
    """Raised when the LLM fails to produce a valid evaluation sample."""


# ── System prompt ─────────────────────────────────────────────────────
# The LLM is instructed to return ONLY JSON — this guarantees parseable output.

_SYSTEM_PROMPT = f"""You are an expert evaluation dataset generator for a RAG (Retrieval-Augmented Generation) system.

Your task is to generate exactly ONE evaluation sample from the document chunk provided by the user.

## Output Format
You MUST return ONLY a valid JSON object. No markdown, no explanation, no extra text.

Required JSON keys:
- "query":        A natural-language question that can be answered from the chunk. Must be non-empty.
- "ground_truth": The ideal, complete answer to the query using ONLY information from the chunk. Must be non-empty.
- "query_type":   One of: {', '.join(QUERY_TYPES)}
- "difficulty":   One of: {', '.join(DIFFICULTY_LEVELS)}
- "answer_type":  One of: {', '.join(ANSWER_TYPES)}

## Rules
1. The query MUST be answerable using ONLY the provided chunk text.
2. The ground_truth MUST be grounded in the chunk -- do not fabricate.
3. Choose query_type to match the question's intent:
   - definition:     "What is X?"
   - procedural:     "How to do X?"
   - explanation:    "Why does X happen?"
   - comparison:     "What is the difference between X and Y?"
   - troubleshooting:"How do you fix/debug X?"
   - list:           "What are the steps/components/types of X?"
   - conceptual:     "What is the core idea behind X?"
4. Choose difficulty based on reasoning depth required:
   - easy:   Directly stated in the chunk.
   - medium: Requires connecting two pieces of information from the chunk.
   - hard:   Requires inference or synthesis beyond a single sentence.
5. Choose answer_type based on HOW the ground_truth was derived:
   - extractive:  Ground truth is a direct quote or near-verbatim span from the chunk.
   - abstractive: Ground truth paraphrases or summarises the chunk in different words.
   - reasoning:   Ground truth requires a logical inference step not explicitly stated.
   - multi_hop:   Ground truth synthesises information from two or more distinct parts of the chunk.
6. Return ONLY the JSON object. No preamble. No trailing text."""


def _build_user_message(chunk_text: str) -> str:
    """Build the user-side message with the chunk content embedded."""
    return (
        f"Generate one evaluation sample from the following document chunk:\n\n"
        f"{'─' * 60}\n"
        f"{chunk_text.strip()}\n"
        f"{'─' * 60}\n\n"
        f"Remember: Return ONLY valid JSON."
    )


class EvalLLMGenerator:
    """Groq-based evaluation sample generator with retry/backoff."""

    def __init__(self) -> None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in environment.")
        self._client = Groq(
            api_key=GROQ_API_KEY,
            timeout=EVAL_LLM_TIMEOUT,
        )

    def generate_eval_sample(self, chunk: dict) -> dict:
        """
        Generate one evaluation sample for the given chunk.

        Args:
            chunk: A MongoDB chunk document with at minimum a "text" field.
                   May also contain "chunk_id", "document_id", "metadata".

        Returns:
            A dict with keys: query, ground_truth, query_type, difficulty
            (validated — callers add document_id, chunk_id, timestamps).

        Raises:
            EvalGenerationError: After all retries are exhausted or the
                input chunk has no usable text.
        """
        text = (chunk.get("content") or "").strip()  # chunks collection uses 'content' not 'text'
        if not text:
            raise EvalGenerationError("Chunk has no text content.")

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_message(text)},
        ]

        last_error: Exception | None = None

        for attempt in range(1, EVAL_LLM_MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    temperature=EVAL_LLM_TEMPERATURE,
                    max_tokens=EVAL_LLM_MAX_TOKENS,
                    # Force JSON-only output where supported
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content.strip()
                parsed  = self._parse_json(content)
                self._validate_llm_output(parsed)

                logger.debug(
                    f"[EvalLLMGenerator] Sample generated — "
                    f"type={parsed['query_type']} diff={parsed['difficulty']} "
                    f"attempt={attempt}"
                )
                return parsed

            except (EvalGenerationError, ValueError) as exc:
                # Validation / parsing failures: retry with a note
                last_error = exc
                wait = EVAL_LLM_RETRY_DELAY * attempt
                logger.warning(
                    f"[EvalLLMGenerator] Validation failed (attempt {attempt}/"
                    f"{EVAL_LLM_MAX_RETRIES}): {exc}. Retrying in {wait}s…"
                )
                time.sleep(wait)

            except RateLimitError as exc:
                last_error = exc
                wait = EVAL_LLM_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"[EvalLLMGenerator] Rate-limited (attempt {attempt}/"
                    f"{EVAL_LLM_MAX_RETRIES}). Retrying in {wait:.1f}s…"
                )
                time.sleep(wait)

            except APIStatusError as exc:
                if exc.status_code >= 500:
                    last_error = exc
                    wait = EVAL_LLM_RETRY_DELAY * attempt
                    logger.warning(
                        f"[EvalLLMGenerator] Server error {exc.status_code} "
                        f"(attempt {attempt}/{EVAL_LLM_MAX_RETRIES}). "
                        f"Retrying in {wait:.1f}s…"
                    )
                    time.sleep(wait)
                else:
                    raise EvalGenerationError(
                        f"Groq API client error {exc.status_code}: {exc.message}"
                    ) from exc

            except APIConnectionError as exc:
                last_error = exc
                wait = EVAL_LLM_RETRY_DELAY * attempt
                logger.warning(
                    f"[EvalLLMGenerator] Connection error (attempt {attempt}/"
                    f"{EVAL_LLM_MAX_RETRIES}). Retrying in {wait:.1f}s…"
                )
                time.sleep(wait)

        raise EvalGenerationError(
            f"Failed to generate evaluation sample after {EVAL_LLM_MAX_RETRIES} "
            f"attempts. Last error: {last_error}"
        )

    # ── Private helpers ───────────────────────────────────────────────

    @staticmethod
    def _parse_json(content: str) -> dict:
        """
        Parse JSON from LLM output.
        Strips markdown code fences if the model ignores the instruction.
        """
        # Strip potential code fences: ```json ... ```
        cleaned = content
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
            cleaned = "\n".join(
                l for l in lines
                if not l.strip().startswith("```")
            ).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM output is not valid JSON: {exc}\nOutput: {content[:300]}") from exc

    @staticmethod
    def _validate_llm_output(data: dict) -> None:
        """
        Validate the parsed LLM response has all required keys with valid values.
        Raises ValueError if any check fails.
        """
        missing = REQUIRED_LLM_KEYS - set(data.keys())
        if missing:
            raise ValueError(f"LLM JSON missing required keys: {missing}")

        if not str(data.get("query", "")).strip():
            raise ValueError("LLM returned empty 'query'.")

        if not str(data.get("ground_truth", "")).strip():
            raise ValueError("LLM returned empty 'ground_truth'.")

        if data.get("query_type") not in QUERY_TYPES:
            raise ValueError(
                f"Invalid query_type '{data.get('query_type')}'. "
                f"Must be one of: {QUERY_TYPES}"
            )

        if data.get("difficulty") not in DIFFICULTY_LEVELS:
            raise ValueError(
                f"Invalid difficulty '{data.get('difficulty')}'. "
                f"Must be one of: {DIFFICULTY_LEVELS}"
            )

        if data.get("answer_type") not in ANSWER_TYPES:
            raise ValueError(
                f"Invalid answer_type '{data.get('answer_type')}'. "
                f"Must be one of: {ANSWER_TYPES}"
            )


# Module-level singleton — one HTTP client pool for the whole script run
eval_llm_generator = EvalLLMGenerator()
