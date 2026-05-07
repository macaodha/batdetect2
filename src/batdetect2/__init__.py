import logging
import warnings
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from batdetect2.api_v2 import BatDetect2API

__all__ = ["BatDetect2API", "__version__"]
__version__ = "2.0.0b1"

logger.disable("batdetect2")

# Silences the irrelevant warning
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
warnings.filterwarnings("ignore", message=".*isinstance(treespec, LeafSpec).*")

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


def __getattr__(name: str):
    if name == "BatDetect2API":
        from batdetect2.api_v2 import BatDetect2API

        return BatDetect2API

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
