from datetime import datetime, timezone
from app.database.mongodb_client import MongoDBClient
from app.config.settings import EMBEDDING_MODEL_NAME


class ChunkRepository:
    def __init__(self):
        self.collection = MongoDBClient().get_collection("chunks")

    def insert_chunk(self, chunk_data: dict):
        """Insert a chunk and initialise its embedding status fields."""
        chunk_data.setdefault("embedding_status", "pending")
        chunk_data.setdefault("embedding_model", EMBEDDING_MODEL_NAME)
        chunk_data.setdefault("embedded_at", None)
        self.collection.insert_one(chunk_data)

    def get_chunks_by_document_id(self, document_id: str) -> list:
        """Retrieve all chunks for a document (omits MongoDB _id)."""
        cursor = self.collection.find({"document_id": document_id}, {"_id": 0})
        return list(cursor)

    def get_chunk_by_id(self, chunk_id: str) -> dict | None:
        """Retrieve a single chunk by its chunk_id (omits MongoDB _id)."""
        return self.collection.find_one({"chunk_id": chunk_id}, {"_id": 0})

    def get_pending_chunks(self, document_id: str) -> list:
        """Return chunks where embedding_status != 'completed'."""
        cursor = self.collection.find(
            {
                "document_id": document_id,
                "embedding_status": {"$ne": "completed"},
            },
            {"_id": 0},
        )
        return list(cursor)

    def mark_embedded(self, chunk_id: str):
        """Set embedding_status='completed' and record the timestamp."""
        self.collection.update_one(
            {"chunk_id": chunk_id},
            {"$set": {
                "embedding_status": "completed",
                "embedded_at": datetime.now(timezone.utc).isoformat(),
            }},
        )

    def mark_failed(self, chunk_id: str):
        """Set embedding_status='failed' for observability and retry targeting."""
        self.collection.update_one(
            {"chunk_id": chunk_id},
            {"$set": {"embedding_status": "failed"}},
        )