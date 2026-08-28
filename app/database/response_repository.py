"""response_repository.py — MongoDB `responses` collection writer."""

from datetime import datetime, timezone
from app.database.mongodb_client import MongoDBClient


class ResponseRepository:
    """Persist LLM response metadata to the `responses` collection."""

    def __init__(self) -> None:
        self.collection = MongoDBClient().get_collection("responses")
        self._ensure_ttl_index()

    def _ensure_ttl_index(self) -> None:
        try:
            # 60-day TTL index on created_at timestamp
            self.collection.create_index("created_at", expireAfterSeconds=60 * 86400)
        except Exception:
            pass

    def insert_response(self, response_data: dict) -> None:
        """Insert one response log document. Failures are non-fatal."""
        if "created_at" not in response_data:
            response_data["created_at"] = datetime.now(timezone.utc)
        self.collection.insert_one(response_data)
