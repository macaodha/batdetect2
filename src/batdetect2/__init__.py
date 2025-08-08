import logging

from loguru import logger

logger.disable("batdetect2")


numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

__version__ = "1.1.1"
