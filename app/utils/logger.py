import logging
import os
from logging.handlers import TimedRotatingFileHandler

# Define log directory and file path
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Create a custom production-grade logger
logger = logging.getLogger("rag_logger")
logger.setLevel(logging.INFO)

# Avoid duplicating handlers if the logger is instantiated multiple times
if not logger.handlers:
    # 1. Console Handler (for standard terminal output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 2. File Handler (Rotates daily at midnight, keeps exactly 5 backup days)
    file_handler = TimedRotatingFileHandler(
        filename=LOG_FILE,
        when="midnight",  # Rotate the log exactly at midnight
        interval=1,       # 1 day interval
        backupCount=5,    # Retain the last 5 days of log files, automatically delete older
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.suffix = "%Y-%m-%d" # Suffix format for rotated files: app.log.2026-04-18

    # Define a rich production-grade log formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Attach the formatter to both output handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add both handlers to the centralized logger instance
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)