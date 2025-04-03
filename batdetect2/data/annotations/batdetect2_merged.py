import json
import os
from pathlib import Path
from typing import Literal, Optional, Union

from soundevent import data

from batdetect2.data.annotations.legacy import (
    FileAnnotation,
    file_annotation_to_annotation_task,
    file_annotation_to_clip,
    file_annotation_to_clip_annotation,
)
from batdetect2.data.annotations.types import AnnotatedDataset

PathLike = Union[Path, str, os.PathLike]

__all__ = [
    "BatDetect2MergedAnnotations",
    "load_batdetect2_merged_annotated_dataset",
]


class BatDetect2MergedAnnotations(AnnotatedDataset):
    format: Literal["batdetect2_file"] = "batdetect2_file"
    annotations_path: Path


def load_batdetect2_merged_annotated_dataset(
    dataset: BatDetect2MergedAnnotations,
    base_dir: Optional[PathLike] = None,
) -> data.AnnotationProject:
    audio_dir = dataset.audio_dir
    path = dataset.annotations_path

    if base_dir:
        audio_dir = base_dir / audio_dir
        path = base_dir / path

    content = json.loads(Path(path).read_text())

    annotations = []
    tasks = []

    for ann in content:
        try:
            ann = FileAnnotation.model_validate(ann)
        except ValueError:
            continue

        try:
            clip = file_annotation_to_clip(ann, audio_dir=audio_dir)
        except FileNotFoundError:
            continue

        annotations.append(file_annotation_to_clip_annotation(ann, clip))
        tasks.append(file_annotation_to_annotation_task(ann, clip))

    return data.AnnotationProject(
        name=dataset.name,
        description=dataset.description,
        clip_annotations=annotations,
        tasks=tasks,
    )
