from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from pydantic import Field
from soundevent import data, io

from batdetect2.compat.data import (
    load_annotation_project_from_dir,
    load_annotation_project_from_file,
)
from batdetect2.configs import BaseConfig, load_config

__all__ = [
    "load_datasets_from_config",
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


class DatasetInfo(BaseConfig):
    name: str
    audio_dir: Path
    annotations: AnnotationFormats = Field(discriminator="format")


class DatasetsConfig(BaseConfig):
    train: List[DatasetInfo] = Field(default_factory=list)
    test: List[DatasetInfo] = Field(default_factory=list)


def load_dataset(
    info: DatasetInfo,
    audio_dir: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> data.AnnotationProject:
    audio_dir = (
        info.audio_dir if base_dir is None else base_dir / info.audio_dir
    )

    path = (
        info.annotations.path
        if base_dir is None
        else base_dir / info.annotations.path
    )

    if info.annotations.format == "batdetect2":
        return load_annotation_project_from_dir(
            path,
            name=info.name,
            audio_dir=audio_dir,
        )

    if info.annotations.format == "batdetect2_file":
        return load_annotation_project_from_file(
            path,
            name=info.name,
            audio_dir=audio_dir,
        )

    if info.annotations.format == "aoef":
        return io.load(  # type: ignore
            info.annotations.path,
            audio_dir=audio_dir,
        )

    raise NotImplementedError(
        f"Unknown annotation format: {info.annotations.name}"
    )


def load_datasets(
    config: DatasetsConfig,
    base_dir: Optional[Path] = None,
) -> Tuple[List[data.ClipAnnotation], List[data.ClipAnnotation]]:
    test_annotations = []
    train_annotations = []

    for dataset in config.train:
        project = load_dataset(dataset, base_dir=base_dir)
        train_annotations.extend(project.clip_annotations)

    for dataset in config.test:
        project = load_dataset(dataset, base_dir=base_dir)
        test_annotations.extend(project.clip_annotations)

    return train_annotations, test_annotations


def load_datasets_from_config(
    path: data.PathLike,
    field: Optional[str] = None,
    base_dir: Optional[Path] = None,
):
    config = load_config(
        path=path,
        schema=DatasetsConfig,
        field=field,
    )
    return load_datasets(config, base_dir=base_dir)
