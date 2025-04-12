from pathlib import Path
from typing import Literal, Union

from batdetect2.configs import BaseConfig

__all__ = [
    "AOEFAnnotationFile",
    "AnnotationFormats",
    "BatDetect2AnnotationFile",
    "BatDetect2AnnotationFiles",
]


class BatDetect2AnnotationFiles(BaseConfig):
    format: Literal["batdetect2"] = "batdetect2"
    path: Path


class BatDetect2AnnotationFile(BaseConfig):
    format: Literal["batdetect2_file"] = "batdetect2_file"
    path: Path


class AOEFAnnotationFile(BaseConfig):
    format: Literal["aoef"] = "aoef"
    path: Path


AnnotationFormats = Union[
    BatDetect2AnnotationFiles,
    BatDetect2AnnotationFile,
    AOEFAnnotationFile,
]
