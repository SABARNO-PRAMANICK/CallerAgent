import logging
from pathlib import Path

def setup_logger():
    """Configure the logger with hardcoded settings."""
    log_level = logging.INFO
    log_file = "/app/logs/voice_ai.log"
    
    log_path = Path(log_file).parent
    log_path.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("calleragent")

logger = setup_logger()