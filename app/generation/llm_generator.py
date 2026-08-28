"""llm_generator.py — Single Responsibility: Call Groq API and return structured response.

Production features:
  - Singleton Groq client (one HTTP connection pool for the whole app lifetime)
  - Exponential-backoff retry on transient errors (429 rate-limit, 5xx server errors)
  - Hard request timeout enforced via the Groq client timeout parameter
  - Structured return dict with answer text + full token usage metadata
  - Raises LLMGenerationError on unrecoverable failure so callers can
    distinguish LLM failure from retrieval failure and respond appropriately.

Model used: configured via settings.GROQ_MODEL (default: openai/gpt-oss-120b)
API key:    read from GROQ_API_KEY environment variable via settings.
"""

import time
import asyncio

from groq import Groq, AsyncGroq, APIStatusError, APIConnectionError, RateLimitError

from app.config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_TIMEOUT_SECONDS,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAY,
)
from app.utils.logger import logger


class LLMGenerationError(Exception):
    """Raised when the LLM fails to generate after all retry attempts."""


class LLMGenerator:
    """Groq-based LLM generator with sync and async retry support."""

    def __init__(self) -> None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        self._client = Groq(
            api_key=GROQ_API_KEY,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        self._async_client = AsyncGroq(
            api_key=GROQ_API_KEY,
            timeout=LLM_TIMEOUT_SECONDS,
        )

    async def generate_async(self, messages: list[dict]) -> dict:
        """
        Asynchronously call the Groq chat completion API with exponential backoff.
        """
        last_error: Exception | None = None

        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                t0 = time.time()

                response = await self._async_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_OUTPUT_TOKENS,
                )

                latency_ms = round((time.time() - t0) * 1000, 1)
                answer = response.choices[0].message.content.strip()
                usage  = response.usage

                logger.info(
                    f"[LLMGenerator] (Async) Generated answer — "
                    f"model={GROQ_MODEL} "
                    f"tokens={usage.total_tokens} ({usage.prompt_tokens}in/"
                    f"{usage.completion_tokens}out) "
                    f"latency={latency_ms}ms"
                )

                return {
                    "answer":        answer,
                    "model":         GROQ_MODEL,
                    "input_tokens":  usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "total_tokens":  usage.total_tokens,
                    "latency_ms":    latency_ms,
                }

            except RateLimitError as exc:
                last_error = exc
                wait = LLM_RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"[LLMGenerator] Rate-limited (attempt {attempt}/{LLM_MAX_RETRIES}). "
                    f"Retrying in {wait:.1f}s…"
                )
                await asyncio.sleep(wait)

            except APIStatusError as exc:
                if exc.status_code >= 500:
                    last_error = exc
                    wait = LLM_RETRY_DELAY * attempt
                    logger.warning(
                        f"[LLMGenerator] Server error {exc.status_code} "
                        f"(attempt {attempt}/{LLM_MAX_RETRIES}). "
                        f"Retrying in {wait:.1f}s…"
                    )
                    await asyncio.sleep(wait)
                else:
                    raise LLMGenerationError(
                        f"Groq API client error {exc.status_code}: {exc.message}"
                    ) from exc

            except APIConnectionError as exc:
                last_error = exc
                wait = LLM_RETRY_DELAY * attempt
                logger.warning(
                    f"[LLMGenerator] Connection error (attempt {attempt}/{LLM_MAX_RETRIES}). "
                    f"Retrying in {wait:.1f}s…"
                )
                await asyncio.sleep(wait)

        raise LLMGenerationError(
            f"LLM generation failed after {LLM_MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    def generate(self, messages: list[dict]) -> dict:
        """
        Call the Groq chat completion API with exponential-backoff retry.

        Args:
            messages: List of role/content dicts from prompt_builder.build_messages().

        Returns:
            dict with:
                answer       (str)   — the LLM's response text
                model        (str)   — model name used
                input_tokens (int)   — prompt token count
                output_tokens(int)   — completion token count
                total_tokens (int)   — input + output
                latency_ms   (float) — wall-clock time for the API call

        Raises:
            LLMGenerationError: after all retries are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                t0 = time.time()

                response = self._client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_OUTPUT_TOKENS,
                )

                latency_ms = round((time.time() - t0) * 1000, 1)
                answer = response.choices[0].message.content.strip()
                usage  = response.usage

                logger.info(
                    f"[LLMGenerator] Generated answer — "
                    f"model={GROQ_MODEL} "
                    f"tokens={usage.total_tokens} ({usage.prompt_tokens}in/"
                    f"{usage.completion_tokens}out) "
                    f"latency={latency_ms}ms"
                )

                return {
                    "answer":        answer,
                    "model":         GROQ_MODEL,
                    "input_tokens":  usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "total_tokens":  usage.total_tokens,
                    "latency_ms":    latency_ms,
                }

            except RateLimitError as exc:
                last_error = exc
                wait = LLM_RETRY_DELAY * (2 ** (attempt - 1))  # exponential back-off
                logger.warning(
                    f"[LLMGenerator] Rate-limited (attempt {attempt}/{LLM_MAX_RETRIES}). "
                    f"Retrying in {wait:.1f}s…"
                )
                time.sleep(wait)

            except APIStatusError as exc:
                # 5xx server errors are transient; 4xx (except 429) are permanent
                if exc.status_code >= 500:
                    last_error = exc
                    wait = LLM_RETRY_DELAY * attempt
                    logger.warning(
                        f"[LLMGenerator] Server error {exc.status_code} "
                        f"(attempt {attempt}/{LLM_MAX_RETRIES}). "
                        f"Retrying in {wait:.1f}s…"
                    )
                    time.sleep(wait)
                else:
                    # 4xx client errors will not be fixed by retrying
                    raise LLMGenerationError(
                        f"Groq API client error {exc.status_code}: {exc.message}"
                    ) from exc

            except APIConnectionError as exc:
                last_error = exc
                wait = LLM_RETRY_DELAY * attempt
                logger.warning(
                    f"[LLMGenerator] Connection error (attempt {attempt}/{LLM_MAX_RETRIES}). "
                    f"Retrying in {wait:.1f}s…"
                )
                time.sleep(wait)

        raise LLMGenerationError(
            f"LLM generation failed after {LLM_MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )


# Module-level singleton — Groq client is safe to share across requests
llm_generator = LLMGenerator()
