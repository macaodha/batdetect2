from pathlib import Path
from typing import Optional, Union

from soundevent import data

from batdetect2.data.annotations.aeof import (
    AOEFAnnotations,
    load_aoef_annotated_dataset,
)
from batdetect2.data.annotations.batdetect2_files import (
    BatDetect2FilesAnnotations,
    load_batdetect2_files_annotated_dataset,
)
from batdetect2.data.annotations.batdetect2_merged import (
    BatDetect2MergedAnnotations,
    load_batdetect2_merged_annotated_dataset,
)
from batdetect2.data.annotations.types import AnnotatedDataset

__all__ = [
    "load_annotated_dataset",
    "AnnotatedDataset",
    "AOEFAnnotations",
    "BatDetect2FilesAnnotations",
    "BatDetect2MergedAnnotations",
    "AnnotationFormats",
]


AnnotationFormats = Union[
    BatDetect2MergedAnnotations,
    BatDetect2FilesAnnotations,
    AOEFAnnotations,
]


def load_annotated_dataset(
    dataset: AnnotatedDataset,
    base_dir: Optional[Path] = None,
) -> data.AnnotationSet:
    if isinstance(dataset, AOEFAnnotations):
        return load_aoef_annotated_dataset(dataset, base_dir=base_dir)

    if isinstance(dataset, BatDetect2MergedAnnotations):
        return load_batdetect2_merged_annotated_dataset(
            dataset, base_dir=base_dir
        )

    if isinstance(dataset, BatDetect2FilesAnnotations):
        return load_batdetect2_files_annotated_dataset(
            dataset,
            base_dir=base_dir,
        )

    raise NotImplementedError(f"Unknown annotation format: {dataset.name}")
