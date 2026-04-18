import os
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = "data/uploads"
PROCESSED_DIR = "data/processed"

ALLOWED_EXTENSION = [
    ".pdf",
    ".docx",
    ".png",
    ".jpg"
]

MAX_FILE_SIZE = 10 * 1024 * 1024 # 10MB