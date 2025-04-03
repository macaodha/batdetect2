from pathlib import Path
from typing import Literal, Optional

from soundevent import data, io

from batdetect2.data.annotations.types import AnnotatedDataset

__all__ = [
    "AOEFAnnotations",
    "load_aoef_annotated_dataset",
]


class AOEFAnnotations(AnnotatedDataset):
    format: Literal["aoef"] = "aoef"
    annotations_path: Path


def load_aoef_annotated_dataset(
    dataset: AOEFAnnotations,
    base_dir: Optional[Path] = None,
) -> data.AnnotationSet:
    audio_dir = dataset.audio_dir
    path = dataset.annotations_path

    if base_dir:
        audio_dir = base_dir / audio_dir
        path = base_dir / path

    loaded = io.load(path, audio_dir=audio_dir)

    if not isinstance(loaded, (data.AnnotationSet, data.AnnotationProject)):
        raise ValueError(
            f"The AOEF file at {path} does not contain a set of annotations"
        )

    return loaded
