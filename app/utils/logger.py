import logging
import os

def setup_logger():
    """Set up logger with file and console handlers."""
    logger = logging.getLogger("voice_ai")
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
    
    # Create logs directory if it doesn't exist
    os.makedirs("app/logs", exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler("app/logs/voice_ai.log", encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setStream(open(console_handler.stream.fileno(), mode='w', encoding='utf-8', buffering=1))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()