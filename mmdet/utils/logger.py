import logging

def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger."""
    logger = logging.getLogger('mmdet')
    logger.setLevel(log_level)
    
    # Create a handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    # Create a handler for file output
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    return logger
