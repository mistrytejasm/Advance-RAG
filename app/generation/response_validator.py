"""response_validator.py — Single Responsibility: Validate LLM answer quality.

Why ChatGPT's proposed validator was broken:
  BAD:  `if "I don't know" in answer: return False`
  This rejects perfectly valid answers that CONTAIN that phrase as part
  of a sentence (e.g., "Scientists didn't know this until 1990").

Our validator checks three structurally correct conditions:

  1. Empty answer guard:
     The LLM returned an empty string or only whitespace → invalid.

  2. No-answer phrase detection:
     The LLM explicitly returned the LLM_NO_ANSWER_PHRASE sentinel (exact
     match from settings). This is a VALID, expected response when the context
     genuinely doesn't contain the answer — the validator marks it valid but
     sets `is_grounded=False` so the caller can adjust response metadata.

  3. Minimum length guard:
     Answers shorter than MIN_ANSWER_LENGTH characters are almost certainly
     model failures (e.g., "OK." or a single word) — mark as invalid.
"""

from app.config.settings import LLM_NO_ANSWER_PHRASE
from app.utils.logger import logger

# Shortest meaningful answer in characters (configurable here, not in settings
# because it is an internal quality heuristic, not a user-facing parameter)
_MIN_ANSWER_LENGTH = 20


def validate_response(answer: str) -> tuple[bool, bool, str]:
    """
    Validate the LLM-generated answer.

    Args:
        answer: The raw text returned by LLMGenerator.generate().

    Returns:
        (is_valid, is_grounded, reason) where:
            is_valid   — True if the answer is usable (may still be a refusal)
            is_grounded — True if the answer answers the question from context
                          False if the LLM returned the no-answer sentinel
            reason     — Human-readable string explaining the validation result
    """
    cleaned = answer.strip() if answer else ""

    # Guard 1: Empty answer
    if not cleaned:
        logger.warning("[ResponseValidator] LLM returned empty answer.")
        return False, False, "LLM returned an empty answer."

    # Guard 2: No-answer sentinel — valid (intentional refusal), not grounded
    if cleaned == LLM_NO_ANSWER_PHRASE:
        logger.info(
            "[ResponseValidator] LLM returned no-answer sentinel "
            "(context insufficient)."
        )
        return True, False, "LLM determined context is insufficient."

    # Guard 3: Minimum length
    if len(cleaned) < _MIN_ANSWER_LENGTH:
        logger.warning(
            f"[ResponseValidator] Answer too short ({len(cleaned)} chars): '{cleaned}'"
        )
        return False, False, f"Answer too short ({len(cleaned)} chars); likely a model error."

    return True, True, "Answer is valid and grounded."
