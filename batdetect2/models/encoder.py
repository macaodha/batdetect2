import sys
from typing import Iterable, List, Literal, Sequence

import torch
from torch import nn

from batdetect2.models.blocks import ConvBlockDownCoordF, ConvBlockDownStandard

if sys.version_info >= (3, 10):
    from itertools import pairwise
else:

    def pairwise(iterable: Sequence) -> Iterable:
        for x, y in zip(iterable[:-1], iterable[1:]):
            yield x, y
