from pathlib import Path
from typing import Optional

from soundevent import data

from batdetect2.configs import load_config
from batdetect2.data.annotations import load_annotated_dataset
from batdetect2.data.types import Dataset

__all__ = [
    "load_dataset",
    "load_dataset_from_config",
]


def load_dataset(
    dataset: Dataset,
    base_dir: Optional[Path] = None,
) -> data.AnnotationSet:
    clip_annotations = []
    for source in dataset.sources:
        annotated_source = load_annotated_dataset(source, base_dir=base_dir)
        clip_annotations.extend(annotated_source.clip_annotations)
    return data.AnnotationSet(clip_annotations=clip_annotations)


def load_dataset_from_config(
    path: data.PathLike,
    field: Optional[str] = None,
    base_dir: Optional[Path] = None,
):
    config = load_config(
        path=path,
        schema=Dataset,
        field=field,
    )
    return load_dataset(config, base_dir=base_dir)
