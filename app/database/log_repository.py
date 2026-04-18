from app.database.mongodb_client import MongoDBClient

class LogRepository:

    def __init__(self):
        self.collection = MongoDBClient().get_collection(
            "logs"
        )

    def insert_log(self, log_data):
        self.collection.insert_one(
            log_data
        )