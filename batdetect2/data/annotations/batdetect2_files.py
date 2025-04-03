import os
from pathlib import Path
from typing import Literal, Optional, Union

from soundevent import data

from batdetect2.data.annotations.legacy import (
    file_annotation_to_annotation_task,
    file_annotation_to_clip,
    file_annotation_to_clip_annotation,
    list_file_annotations,
    load_file_annotation,
)
from batdetect2.data.annotations.types import AnnotatedDataset

PathLike = Union[Path, str, os.PathLike]


__all__ = [
    "load_batdetect2_files_annotated_dataset",
    "BatDetect2FilesAnnotations",
]


class BatDetect2FilesAnnotations(AnnotatedDataset):
    format: Literal["batdetect2"] = "batdetect2"
    annotations_dir: Path


def load_batdetect2_files_annotated_dataset(
    dataset: BatDetect2FilesAnnotations,
    base_dir: Optional[PathLike] = None,
) -> data.AnnotationProject:
    """Convert annotations to annotation project."""
    audio_dir = dataset.audio_dir
    path = dataset.annotations_dir

    if base_dir:
        audio_dir = base_dir / audio_dir
        path = base_dir / path

    paths = list_file_annotations(path)

    annotations = []
    tasks = []

    for p in paths:
        try:
            file_annotation = load_file_annotation(p)
        except FileNotFoundError:
            continue

        try:
            clip = file_annotation_to_clip(
                file_annotation,
                audio_dir=audio_dir,
            )
        except FileNotFoundError:
            continue

        annotations.append(
            file_annotation_to_clip_annotation(
                file_annotation,
                clip,
            )
        )

        tasks.append(
            file_annotation_to_annotation_task(
                file_annotation,
                clip,
            )
        )

    return data.AnnotationProject(
        name=dataset.name,
        description=dataset.description,
        clip_annotations=annotations,
        tasks=tasks,
    )
