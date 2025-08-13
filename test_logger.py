from src.utils.logger import default_logger as logger


try:
    logger.info("Success Running")
except:
    logger.info("Failed Running")