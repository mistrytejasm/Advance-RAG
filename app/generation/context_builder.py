"""context_builder.py — Single Responsibility: Build token-budgeted LLM context.

Converts retrieved, reranked chunks into a structured text block ready for
the LLM prompt. Key production features:

  Token budget: Uses tiktoken to count tokens precisely. Adds chunks greedily
                from highest rerank_score downward until MAX_CONTEXT_TOKENS is
                exhausted. This prevents context window overflow regardless of
                chunk size.

  Hard cap:     Never exceeds MAX_CONTEXT_CHUNKS chunks even if token budget
                allows more (keeps latency predictable).

  Structured format: Each chunk is formatted with its source metadata so the
                     LLM can cite sources accurately.

  Returns both the context string AND the list of chunks actually included,
  so the citation builder and response logger work from the same subset.
"""

import tiktoken

from app.config.settings import MAX_CONTEXT_TOKENS, MAX_CONTEXT_CHUNKS
from app.utils.logger import logger

# Use cl100k_base as a conservative token count approximation —
# accurate for GPT-4 family; slightly over-counts for Llama/Mistral which
# is intentional (we prefer to stay under the limit).
_ENCODER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Return the number of tokens in `text` using the cl100k_base encoder."""
    return len(_ENCODER.encode(text))


class ContextBuilder:
    """Build a token-budgeted context string from reranked chunks."""

    def build(
        self,
        chunks: list[dict],
        max_tokens: int = MAX_CONTEXT_TOKENS,
        max_chunks: int = MAX_CONTEXT_CHUNKS,
    ) -> tuple[str, list[dict]]:
        """
        Select and format the most relevant chunks within the token budget.

        Args:
            chunks:     Reranked result dicts from the retrieval pipeline.
                        Each must have: chunk_id, content, page, section,
                        source, content_type, rerank_score.
            max_tokens: Maximum total tokens for the context block.
            max_chunks: Hard cap on number of chunks included.

        Returns:
            (context_string, used_chunks) — the formatted context text and
            the subset of chunks that were included (needed by citation builder).
        """
        if not chunks:
            return "", []

        # Sort descending by rerank_score — best evidence first.
        ranked = sorted(chunks, key=lambda c: c.get("rerank_score", 0), reverse=True)

        used_chunks: list[dict] = []
        context_parts: list[str] = []
        tokens_used: int = 0

        for chunk in ranked:
            if len(used_chunks) >= max_chunks:
                break

            chunk_text = self._format_chunk(chunk, index=len(used_chunks) + 1)
            chunk_tokens = _count_tokens(chunk_text)

            if tokens_used + chunk_tokens > max_tokens:
                # Try to add a truncated version of this chunk
                remaining = max_tokens - tokens_used
                if remaining > 50:  # Only truncate if meaningful space remains
                    chunk_text = self._truncate_chunk(chunk, remaining)
                    chunk_tokens = _count_tokens(chunk_text)
                    context_parts.append(chunk_text)
                    used_chunks.append(chunk)
                    tokens_used += chunk_tokens
                break

            context_parts.append(chunk_text)
            used_chunks.append(chunk)
            tokens_used += chunk_tokens

        context_string = "\n\n---\n\n".join(context_parts)

        logger.info(
            f"[ContextBuilder] {len(used_chunks)} chunks selected, "
            f"~{tokens_used} tokens used (budget: {max_tokens})"
        )
        return context_string, used_chunks

    # ── Private helpers ───────────────────────────────────────────────

    def _format_chunk(self, chunk: dict, index: int) -> str:
        """Format a single chunk as a labelled context block."""
        page    = chunk.get("page", "?")
        section = chunk.get("section", "").strip()
        content = chunk.get("content", "").strip()

        header = f"[Source {index}] Page {page}"
        if section:
            header += f" | Section: {section}"

        return f"{header}\n{content}"

    def _truncate_chunk(self, chunk: dict, token_limit: int) -> str:
        """Return a chunk truncated to fit within `token_limit` tokens."""
        full = self._format_chunk(chunk, index="?")
        tokens = _ENCODER.encode(full)
        truncated_tokens = tokens[:token_limit]
        return _ENCODER.decode(truncated_tokens) + " [truncated]"


# Module-level singleton
context_builder = ContextBuilder()
