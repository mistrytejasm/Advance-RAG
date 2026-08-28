"""
monitoring/cost_tracker.py  — Token-based API cost estimation.

Prices are per 1 000 tokens (input + output blended) taken from public
pricing pages. Override via environment variables for other models.

Usage:
    from monitoring.cost_tracker import compute_cost
    usd = compute_cost(tokens=1450, model="openai/gpt-oss-120b")
"""
import os

# Detailed Rates (Input / Output per 1K tokens)
MODEL_RATES: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile": {
        "input": float(os.getenv("PRICE_LLAMA_70B_IN", "0.00059")),
        "output": float(os.getenv("PRICE_LLAMA_70B_OUT", "0.00079")),
    },
    "llama-3.1-70b-versatile": {
        "input": float(os.getenv("PRICE_LLAMA_70B_IN", "0.00059")),
        "output": float(os.getenv("PRICE_LLAMA_70B_OUT", "0.00079")),
    },
    "openai/gpt-oss-120b": {
        "input": float(os.getenv("PRICE_GPT_OSS_IN", "0.00060")),
        "output": float(os.getenv("PRICE_GPT_OSS_OUT", "0.00180")),
    },
    "gpt-4o-mini": {
        "input": float(os.getenv("PRICE_GPT4O_MINI_IN", "0.00015")),
        "output": float(os.getenv("PRICE_GPT4O_MINI_OUT", "0.00060")),
    },
    "gpt-4o": {
        "input": float(os.getenv("PRICE_GPT4O_IN", "0.00250")),
        "output": float(os.getenv("PRICE_GPT4O_OUT", "0.01000")),
    },
}

DEFAULT_IN_PRICE = float(os.getenv("PRICE_DEFAULT_IN", "0.00059"))
DEFAULT_OUT_PRICE = float(os.getenv("PRICE_DEFAULT_OUT", "0.00079"))


def compute_cost(
    tokens: int = 0,
    model: str = "",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> float:
    """
    Estimate the API cost for a given request with granular input/output token pricing.

    Args:
        tokens:        Total tokens (legacy fallback if input/output not specified).
        model:         Model identifier string.
        input_tokens:  Prompt tokens count.
        output_tokens: Completion tokens count.

    Returns:
        Estimated cost in USD, rounded to 8 decimal places.
    """
    rates = MODEL_RATES.get(model, {"input": DEFAULT_IN_PRICE, "output": DEFAULT_OUT_PRICE})

    if input_tokens is not None or output_tokens is not None:
        in_tok = input_tokens or 0
        out_tok = output_tokens or 0
        cost = (in_tok / 1000.0) * rates["input"] + (out_tok / 1000.0) * rates["output"]
        return round(cost, 8)

    if tokens <= 0:
        return 0.0

    blended_rate = (rates["input"] + rates["output"]) / 2.0
    return round((tokens / 1000.0) * blended_rate, 8)

