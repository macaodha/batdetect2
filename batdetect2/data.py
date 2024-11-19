from pathlib import Path
from typing import List, Literal, Tuple, Union

from pydantic import Field
from soundevent import data, io

from batdetect2.compat.data import (
    load_annotation_project_from_dir,
    load_annotation_project_from_file,
)
from batdetect2.configs import BaseConfig


class BatDetect2AnnotationFiles(BaseConfig):
    format: Literal["batdetect2"] = "batdetect2"
    path: Path


class BatDetect2AnnotationFile(BaseConfig):
    format: Literal["batdetect2_file"] = "batdetect2_file"
    path: Path


class AOEFAnnotationFile(BaseConfig):
    format: Literal["aoef"] = "aoef"
    annotations_file: Path


AnnotationFormats = Union[
    BatDetect2AnnotationFiles,
    BatDetect2AnnotationFile,
    AOEFAnnotationFile,
]


class DatasetInfo(BaseConfig):
    name: str
    audio_dir: Path
    annotations: AnnotationFormats = Field(discriminator="format")


class DatasetsConfig(BaseConfig):
    train: List[DatasetInfo] = Field(default_factory=list)
    test: List[DatasetInfo] = Field(default_factory=list)


def load_dataset(info: DatasetInfo) -> data.AnnotationProject:
    if info.annotations.format == "batdetect2":
        return load_annotation_project_from_dir(
            info.annotations.path,
            name=info.name,
            audio_dir=info.audio_dir,
        )

    if info.annotations.format == "batdetect2_file":
        return load_annotation_project_from_file(
            info.annotations.path,
            name=info.name,
            audio_dir=info.audio_dir,
        )

    if info.annotations.format == "aoef":
        return io.load(  # type: ignore
            info.annotations.annotations_file,
            audio_dir=info.audio_dir,
        )

    raise NotImplementedError(
        f"Unknown annotation format: {info.annotations.name}"
    )


def load_datasets(
    config: DatasetsConfig,
) -> Tuple[List[data.ClipAnnotation], List[data.ClipAnnotation]]:
    test_annotations = []
    train_annotations = []

    for dataset in config.train:
        project = load_dataset(dataset)
        train_annotations.extend(project.clip_annotations)

    for dataset in config.test:
        project = load_dataset(dataset)
        test_annotations.extend(project.clip_annotations)

    return train_annotations, test_annotations
