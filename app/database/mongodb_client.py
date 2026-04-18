"""MongoDB Client Connection Setup"""

from pymongo import MongoClient
from app.config.settings import (MONGODB_URI, DATABASE_NAME)

class MongoDBClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.client = MongoClient(MONGODB_URI)
            cls._instance.db = cls._instance.client[DATABASE_NAME]
        return cls._instance

    def get_collection(self, name: str):
        return self.db[name]

    def close(self):
        if self._instance:
            self._instance.client.close()
            self._instance = None

# Create a single instance of MongoDBClient
mongo_client = MongoDBClient()