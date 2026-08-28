"""query_rewriter.py — Single Responsibility: Clean and expand user queries.

Two-stage rewriting:
  Stage 1 — Filler removal:
    Strips polite preambles and filler phrases that add no search value.
    Example: "Can you please tell me what deep learning is?"
             → "what deep learning is"
    Semantic content words ("what", "which", "how") are PRESERVED because
    they carry intent meaning used by the classifier.

  Stage 2 — Abbreviation expansion (single-token queries only):
    If the entire query is a single known abbreviation, expand it to its
    full form plus the acronym so BM25 and vector search both benefit.
    Example: "KNN" → "K-Nearest Neighbors KNN"
    Multi-word queries are left unchanged to avoid false expansions.

Design decisions:
  - FILLER_PHRASES and ABBREVIATIONS are class-level dicts/lists.
    Longer phrases are checked before shorter ones to avoid partial matches
    (e.g., "can you tell me" must come before "tell me").
  - Abbreviation expansion preserves the original abbreviation at the end
    so BM25 can still match the raw acronym alongside the expanded form.
  - Returns a (rewritten_query, was_changed: bool) tuple so the caller
    has full observability without re-computing string equality.
"""

import re

from app.utils.logger import logger


class QueryRewriter:
    """Clean and optionally expand user queries for better retrieval."""

    # ── Filler phrases — ordered longest-first to prevent partial stripping ──
    FILLER_PHRASES: list[str] = [
        "can you please tell me",
        "could you please tell me",
        "can you tell me",
        "could you tell me",
        "i would like to know",
        "i want to know",
        "please tell me",
        "can you explain",
        "could you explain",
        "please explain",
        "tell me about",
        "tell me",
        "please",
    ]

    # ── AI/ML abbreviation expansion table ───────────────────────────────
    # Keys: lowercase abbreviation
    # Values: expanded form WITH original abbreviation appended for BM25 recall
    ABBREVIATIONS: dict[str, str] = {
        "cnn":   "Convolutional Neural Network CNN",
        "rnn":   "Recurrent Neural Network RNN",
        "lstm":  "Long Short-Term Memory LSTM",
        "gru":   "Gated Recurrent Unit GRU",
        "nlp":   "Natural Language Processing NLP",
        "llm":   "Large Language Model LLM",
        "rl":    "Reinforcement Learning RL",
        "rlhf":  "Reinforcement Learning from Human Feedback RLHF",
        "gan":   "Generative Adversarial Network GAN",
        "vae":   "Variational Autoencoder VAE",
        "bert":  "BERT Bidirectional Encoder Representations Transformers",
        "gpt":   "Generative Pre-trained Transformer GPT",
        "knn":   "K-Nearest Neighbors KNN",
        "svm":   "Support Vector Machine SVM",
        "rag":   "Retrieval-Augmented Generation RAG",
        "mlp":   "Multi-Layer Perceptron MLP",
        "ddpm":  "Denoising Diffusion Probabilistic Models DDPM",
        "dpo":   "Direct Preference Optimization DPO",
        "lora":  "Low-Rank Adaptation LoRA",
        "qlora": "Quantized Low-Rank Adaptation QLoRA",
        "ann":   "Approximate Nearest Neighbors ANN",
        "bm25":  "Best Matching 25 BM25",
        "cot":   "Chain of Thought CoT",
        "tot":   "Tree of Thoughts ToT",
        "ocr":   "Optical Character Recognition OCR",
        "ner":   "Named Entity Recognition NER",
        "pos":   "Part-of-Speech POS",
        "asr":   "Automatic Speech Recognition ASR",
        "tts":   "Text to Speech TTS",
        "pca":   "Principal Component Analysis PCA",
        "dnn":   "Deep Neural Network DNN",
        "vlm":   "Vision Language Model VLM",
        "vit":   "Vision Transformer ViT",
        "dqn":   "Deep Q-Network DQN",
        "ppo":   "Proximal Policy Optimization PPO",
        "a3c":   "Asynchronous Advantage Actor-Critic A3C",
        "gmm":   "Gaussian Mixture Model GMM",
        "umap":  "Uniform Manifold Approximation and Projection UMAP",
        "dbscan": "Density-Based Spatial Clustering DBSCAN",
    }

    def rewrite(self, query: str) -> tuple[str, bool]:
        """
        Apply filler removal and abbreviation expansion.

        Args:
            query: Raw user query string.

        Returns:
            (rewritten_query, was_changed) — the cleaned query and a boolean
            indicating whether any transformation was applied.
        """
        if not query or not query.strip():
            return query, False

        original = query.strip()
        q = original.lower()

        # Stage 1: Strip filler phrases (longest first to avoid partial hits)
        for phrase in self.FILLER_PHRASES:
            q = q.replace(phrase, " ")

        # Normalise whitespace and strip trailing punctuation
        q = re.sub(r"\s+", " ", q).strip()
        q = re.sub(r"[?!.]+$", "", q).strip()

        # Stage 2: Abbreviation expansion
        q = self._expand_abbreviations(q)

        # Preserve original casing if no transformation was applied
        rewritten = q if q else original
        was_changed = rewritten.lower() != original.lower()

        if was_changed:
            logger.debug(f"[QueryRewriter] '{original}' → '{rewritten}'")

        return rewritten, was_changed

    def _expand_abbreviations(self, query: str) -> str:
        """Expand single abbreviation or distinct acronym words."""
        tokens = query.split()
        if len(tokens) == 1 and tokens[0] in self.ABBREVIATIONS:
            return self.ABBREVIATIONS[tokens[0]]

        # In multi-token queries, expand standalone acronyms
        expanded_tokens = []
        for t in tokens:
            cleaned_token = re.sub(r"[^\w]", "", t)
            if cleaned_token in self.ABBREVIATIONS and len(cleaned_token) >= 2:
                # Add full form while keeping original token
                expanded_tokens.append(self.ABBREVIATIONS[cleaned_token])
            else:
                expanded_tokens.append(t)
        return " ".join(expanded_tokens)

    def _expand_if_single_abbreviation(self, q: str) -> str:
        """
        Expand `q` to its full form if it is a single known abbreviation.
        Multi-word queries are returned unchanged.
        """
        tokens = q.split()
        if len(tokens) == 1:
            token = tokens[0].lower()
            if token in self.ABBREVIATIONS:
                return self.ABBREVIATIONS[token]
        return q


# Module-level singleton
query_rewriter = QueryRewriter()
