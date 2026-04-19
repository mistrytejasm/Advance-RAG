"""embedding_model.py — Single Responsibility: Load & hold the model.

Only one job: download BAAI/bge-base-en-v1.5 once and expose the
raw SentenceTransformer instance so other modules can use it.
"""

from sentence_transformers import SentenceTransformer
from app.config.settings import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    """Lazy singleton model loader."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        return self._model

    @property
    def model_name(self) -> str:
        return EMBEDDING_MODEL_NAME

    @property
    def dimension(self) -> int:
        return 768


# Module-level singleton — import this, never re-instantiate
embedding_model = EmbeddingModel()
