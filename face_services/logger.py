import logging

logger = logging.getLogger(__name__)
logger.setLevel("INFO")
handler = logging.StreamHandler()
log_format = "%(asctime)s - Face Services API - %(message)s"
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)
logger.addHandler(handler)
