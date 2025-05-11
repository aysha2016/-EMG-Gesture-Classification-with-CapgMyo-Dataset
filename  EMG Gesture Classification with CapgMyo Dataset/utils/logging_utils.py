import logging
import sys
from typing import Optional

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get logger with consistent formatting"""
    logger = logging.getLogger(name)
    
    if level is None:
        level = logging.INFO
    
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
    
    return logger
