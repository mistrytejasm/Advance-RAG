from app.database.mongodb_client import MongoDBClient

class ChunkRepository:
    def __init__(self):
        self.collection = MongoDBClient().get_collection(
            "chunks"
        )

    def insert_chunk(self, chunk_data):
        self.collection.insert_one(
            chunk_data
        )

    def get_chunks_by_document_id(self, document_id: str):
        # Retrieve chunks and omit the MongoDB internal _id object
        cursor = self.collection.find({"document_id": document_id}, {"_id": 0})
        return list(cursor)