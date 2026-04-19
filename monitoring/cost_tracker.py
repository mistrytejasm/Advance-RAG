"""
monitoring/cost_tracker.py  — Token-based API cost estimation.

Prices are per 1 000 tokens (input + output blended) taken from public
pricing pages. Override via environment variables for other models.

Usage:
    from monitoring.cost_tracker import compute_cost
    usd = compute_cost(tokens=1450, model="openai/gpt-oss-120b")
"""
import os

# ── Pricing table (USD per 1 000 tokens, blended input+output estimate) ──
# Sources (April 2026 approximate public rates):
#   Groq llama-3.3-70b-versatile  → $0.00059 / 1K tokens
#   Groq openai/gpt-oss-120b      → $0.00090 / 1K tokens  (estimated)
#   OpenAI gpt-4o-mini            → $0.00015 / 1K tokens
#   OpenAI gpt-4o                 → $0.00500 / 1K tokens
#
# Add your own model by adding a key here or setting an env-var override.
MODEL_PRICING: dict[str, float] = {
    "llama-3.3-70b-versatile":  float(os.getenv("PRICE_LLAMA_70B",    "0.00059")),
    "llama-3.1-70b-versatile":  float(os.getenv("PRICE_LLAMA_70B",    "0.00059")),
    "openai/gpt-oss-120b":      float(os.getenv("PRICE_GPT_OSS_120B", "0.00090")),
    "gpt-4o-mini":              float(os.getenv("PRICE_GPT4O_MINI",   "0.00015")),
    "gpt-4o":                   float(os.getenv("PRICE_GPT4O",        "0.00500")),
}

# Fallback price used when the model is not in the table.
DEFAULT_PRICE_PER_1K = float(os.getenv("PRICE_DEFAULT", "0.00059"))


def compute_cost(tokens: int, model: str) -> float:
    """
    Estimate the API cost for a given request.

    Args:
        tokens: Total tokens consumed (input + output).
        model:  Model identifier string (must match a key in MODEL_PRICING).

    Returns:
        Estimated cost in USD, rounded to 8 decimal places.
        Returns 0.0 if tokens <= 0.
    """
    if tokens <= 0:
        return 0.0

    price_per_1k = MODEL_PRICING.get(model, DEFAULT_PRICE_PER_1K)
    return round((tokens / 1000) * price_per_1k, 8)
