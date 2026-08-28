"""MongoDB Client Connection Setup"""

from pymongo import MongoClient
from app.config.settings import (MONGODB_URI, DATABASE_NAME)

class MongoDBClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = None
            cls._instance._db = None
        return cls._instance

    @property
    def client(self):
        if self._client is None:
            self._client = MongoClient(
                MONGODB_URI,
                maxPoolSize=50,
                minPoolSize=5,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=30000,
            )
            self._db = self._client[DATABASE_NAME]
        return self._client

    @property
    def db(self):
        if self._db is None:
            _ = self.client
        return self._db

    def get_collection(self, name: str):
        return self.db[name]

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
            self._db = None


# Lazy singleton instance
mongo_client = MongoDBClient()