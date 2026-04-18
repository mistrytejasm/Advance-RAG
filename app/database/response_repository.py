"""response_repository.py — MongoDB `responses` collection writer."""

from app.database.mongodb_client import MongoDBClient


class ResponseRepository:
    """Persist LLM response metadata to the `responses` collection."""

    def __init__(self) -> None:
        self.collection = MongoDBClient().get_collection("responses")

    def insert_response(self, response_data: dict) -> None:
        """Insert one response log document. Failures are non-fatal."""
        self.collection.insert_one(response_data)
