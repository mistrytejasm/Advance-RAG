"""batch_manager.py — Single Responsibility: Split chunks into batches.

Iterates over a flat list of chunks and yields fixed-size sublists.
Independent of the embedding model or vector DB — pure Python logic.
"""

from app.config.settings import EMBEDDING_BATCH_SIZE


class BatchManager:
    """Split an input list into fixed-size batches."""

    def __init__(self, batch_size: int = EMBEDDING_BATCH_SIZE):
        self.batch_size = batch_size

    def get_batches(self, items: list) -> list[list]:
        """
        Yield successive sublists of `batch_size` length.

        Example:
            items = [1..110], batch_size = 64
            → [[1..64], [65..110]]
        """
        batches = []
        for start in range(0, len(items), self.batch_size):
            batches.append(items[start: start + self.batch_size])
        return batches

    def extract_texts(self, batch: list[dict]) -> list[str]:
        """Return the raw text strings from a batch of chunk dicts."""
        return [chunk.get("content", "") for chunk in batch]
