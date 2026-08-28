from datetime import datetime, timezone
from app.database.mongodb_client import MongoDBClient

class LogRepository:

    def __init__(self):
        self.collection = MongoDBClient().get_collection(
            "logs"
        )
        self._ensure_ttl_index()

    def _ensure_ttl_index(self):
        try:
            # 30-day TTL index on created_at timestamp
            self.collection.create_index("created_at", expireAfterSeconds=30 * 86400)
        except Exception:
            pass

    def insert_log(self, log_data):
        if "created_at" not in log_data:
            log_data["created_at"] = datetime.now(timezone.utc)
        self.collection.insert_one(
            log_data
        )