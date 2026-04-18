"""Tokenizer utility for managing token counts."""
import tiktoken

class SimpleTokenizer:
    def __init__(self, model_name="cl100k_base"):
        # We use cl100k_base which is standard for OpenAI and approximate for models like BGE/MiniLM
        self._encoding = tiktoken.get_encoding(model_name)

    def encode(self, text: str):
        return self._encoding.encode(text)

    def decode(self, tokens: list):
        return self._encoding.decode(tokens)

    def count_tokens(self, text: str):
        return len(self.encode(text))