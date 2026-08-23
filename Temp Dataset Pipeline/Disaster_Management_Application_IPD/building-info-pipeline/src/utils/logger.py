import logging
import sys

def setup_logger(name: str = "building-info-pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    
    # Only configure if it doesn't already have handlers to prevent duplicates
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

# Create a default instance for easy import
logger = setup_logger()
