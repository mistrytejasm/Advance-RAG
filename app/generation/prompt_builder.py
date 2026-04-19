"""prompt_builder.py — Single Responsibility: Build structured chat prompts for the LLM.

Design principles:
  - The system prompt enforces grounding: the LLM may ONLY answer from the
    provided context, never from its parametric knowledge.
  - The exact "no-answer phrase" is injected from settings so it can be
    changed in config without touching code.
  - The prompt follows the OpenAI ChatCompletion messages format (list of
    role/content dicts), which Groq's client accepts directly.
  - Temperature is kept low (0.1) in settings to favour factual consistency.
"""

from app.config.settings import LLM_NO_ANSWER_PHRASE

# ── System prompt ─────────────────────────────────────────────────────
# This is the single source of truth for LLM behaviour.
# All anti-hallucination rules are embedded here.
SYSTEM_PROMPT = f"""You are a precise, expert document assistant for a RAG (Retrieval-Augmented Generation) system.

Your role is to answer the user's question **exclusively** using the provided document context below.

## Rules you MUST follow:
1. **Grounding**: Answer ONLY from the provided context. Do not use any external or prior knowledge. Quote exact phrases from the context when possible to maximize accuracy.
2. **Partial-answer policy**: If the context provides partial information, answer what you can and clearly state what is missing. Only if the context is completely irrelevant, respond with exactly:
   "{LLM_NO_ANSWER_PHRASE}"
3. **Accuracy**: Never guess, infer beyond context, or fabricate details.
4. **Citations**: At the end of your answer, always list the source(s) you used in this exact format:
   Sources: Page X | Section: Y
5. **Conciseness**: Be thorough but concise. Avoid padding or restating the question.
6. **Formatting**: Use clear, structured prose. Use bullet points only when listing multiple items."""


def build_messages(query: str, context: str) -> list[dict]:
    """
    Build the chat messages payload for the Groq API.

    Args:
        query:   The original user question (not rewritten — we want the answer
                 to match what the user actually asked).
        context: The formatted context string from ContextBuilder.

    Returns:
        List of message dicts in OpenAI ChatCompletion format:
        [{"role": "system", ...}, {"role": "user", ...}]
    """
    user_content = (
        f"Question: {query}\n\n"
        f"Context from the document:\n"
        f"{'=' * 60}\n"
        f"{context}\n"
        f"{'=' * 60}\n\n"
        f"Answer the question based strictly on the context above."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
