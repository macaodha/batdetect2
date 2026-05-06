import logging
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from batdetect2.api_v2 import BatDetect2API

logger.disable("batdetect2")


numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

__all__ = ["BatDetect2API", "__version__"]
__version__ = "1.1.1"


def __getattr__(name: str):
    if name == "BatDetect2API":
        from batdetect2.api_v2 import BatDetect2API

        return BatDetect2API

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
