"""Defines the overall dataset structure and provides loading/saving utilities.

This module focuses on defining what constitutes a BatDetect2 dataset,
potentially composed of multiple distinct data sources with varying annotation
formats. It provides mechanisms to load the annotation metadata from these
sources into a unified representation.

The core components are:
- `DatasetConfig`: A configuration class (typically loaded from YAML) that
  describes the dataset's name, description, and constituent sources.
- `Dataset`: A type alias representing the loaded dataset as a list of
  `soundevent.data.ClipAnnotation` objects. Note that this implies all
  annotation metadata is loaded into memory.
- Loading functions (`load_dataset`, `load_dataset_from_config`): To parse
  a `DatasetConfig` and load the corresponding annotation metadata.
- Saving function (`save_dataset`): To save a loaded list of annotations
  into a standard `soundevent` format.

"""

from pathlib import Path
from typing import List, Optional, Sequence

from loguru import logger
from pydantic import Field
from soundevent import data, io

from batdetect2.core.configs import BaseConfig, load_config
from batdetect2.data.annotations import (
    AnnotatedDataset,
    AnnotationFormats,
    load_annotated_dataset,
)
from batdetect2.data.conditions import (
    SoundEventConditionConfig,
    build_sound_event_condition,
    filter_clip_annotation,
)
from batdetect2.data.transforms import (
    ApplyAll,
    SoundEventTransformConfig,
    build_sound_event_transform,
    transform_clip_annotation,
)
from batdetect2.targets.terms import data_source

__all__ = [
    "load_dataset",
    "load_dataset_from_config",
    "save_dataset",
    "Dataset",
    "DatasetConfig",
]


Dataset = Sequence[data.ClipAnnotation]
"""Type alias for a loaded dataset representation.

Represents an entire dataset *after loading* as a flat Python list containing
all `soundevent.data.ClipAnnotation` objects gathered from all configured data
sources.
"""


class DatasetConfig(BaseConfig):
    """Configuration model defining the structure of a BatDetect2 dataset."""

    name: str
    description: str
    sources: List[AnnotationFormats]

    sound_event_filter: SoundEventConditionConfig | None = None
    sound_event_transforms: List[SoundEventTransformConfig] = Field(
        default_factory=list
    )


def load_dataset(
    config: DatasetConfig,
    base_dir: data.PathLike | None = None,
) -> Dataset:
    """Load all clip annotations from the sources defined in a DatasetConfig."""
    clip_annotations = []

    condition = (
        build_sound_event_condition(config.sound_event_filter)
        if config.sound_event_filter is not None
        else None
    )

    transform = (
        ApplyAll(
            [
                build_sound_event_transform(step)
                for step in config.sound_event_transforms
            ]
        )
        if config.sound_event_transforms
        else None
    )

    for source in config.sources:
        annotated_source = load_annotated_dataset(source, base_dir=base_dir)

        logger.debug(
            "Loaded {num_examples} from dataset source '{source_name}'",
            num_examples=len(annotated_source.clip_annotations),
            source_name=source.name,
        )

        for clip_annotation in annotated_source.clip_annotations:
            clip_annotation = insert_source_tag(clip_annotation, source)

            if condition is not None:
                clip_annotation = filter_clip_annotation(
                    clip_annotation,
                    condition,
                )

            if transform is not None:
                clip_annotation = transform_clip_annotation(
                    clip_annotation,
                    transform,
                )

            clip_annotations.append(clip_annotation)

    return clip_annotations


def insert_source_tag(
    clip_annotation: data.ClipAnnotation,
    source: AnnotatedDataset,
) -> data.ClipAnnotation:
    """Insert the source tag into a ClipAnnotation.

    This function adds a tag to the `ClipAnnotation` object, indicating the
    source from which it was loaded. The source information is derived from
    the `recording` attribute of the `ClipAnnotation`.

    Parameters
    ----------
    clip_annotation : data.ClipAnnotation
        The `ClipAnnotation` object to which the source tag will be added.

    Returns
    -------
    data.ClipAnnotation
        The modified `ClipAnnotation` object with the source tag added.
    """
    return clip_annotation.model_copy(
        update=dict(
            tags=[
                *clip_annotation.tags,
                data.Tag(
                    term=data_source,
                    value=source.name,
                ),
            ]
        ),
    )


def load_dataset_config(path: data.PathLike, field: str | None = None):
    return load_config(path=path, schema=DatasetConfig, field=field)


def load_dataset_from_config(
    path: data.PathLike,
    field: str | None = None,
    base_dir: data.PathLike | None = None,
) -> Dataset:
    """Load dataset annotation metadata from a configuration file.

    This is a convenience function that first loads the `DatasetConfig` from
    the specified file path and optional nested field, and then calls
    `load_dataset` to load all corresponding `ClipAnnotation` objects.

    Parameters
    ----------
    path : data.PathLike
        Path to the configuration file (e.g., YAML).
    field : str, optional
        Dot-separated path to a nested section within the file containing the
        dataset configuration (e.g., "data.training_set"). If None, the
        entire file content is assumed to be the `DatasetConfig`.
    base_dir : Path, optional
        An optional base directory path to resolve relative paths within the
        configuration sources. Passed to `load_dataset`. Defaults to None.

    Returns
    -------
    Dataset (List[data.ClipAnnotation])
        A flat list containing all loaded `ClipAnnotation` metadata objects.

    Raises
    ------
    FileNotFoundError
        If the config file `path` does not exist.
    yaml.YAMLError, pydantic.ValidationError, KeyError, TypeError
        If the configuration file is invalid, cannot be parsed, or does not
        match the `DatasetConfig` schema.
    Exception
        Can raise exceptions from `load_dataset` if loading data from sources
        fails.
    """
    config = load_config(
        path=path,
        schema=DatasetConfig,
        field=field,
    )
    return load_dataset(config, base_dir=base_dir)


def save_dataset(
    dataset: Dataset,
    path: data.PathLike,
    name: str | None = None,
    description: str | None = None,
    audio_dir: Path | None = None,
) -> None:
    """Save a loaded dataset (list of ClipAnnotations) to a file.

    Wraps the provided list of `ClipAnnotation` objects into a
    `soundevent.data.AnnotationSet` and saves it using `soundevent.io.save`.
    This saves the aggregated annotation metadata in the standard soundevent
    format.

    Note: This function saves the *loaded annotation data*, not the original
    `DatasetConfig` structure that defined how the data was assembled from
    various sources.

    Parameters
    ----------
    dataset : Dataset (List[data.ClipAnnotation])
        The list of clip annotations to save (typically the result of
        `load_dataset` or a split thereof).
    path : data.PathLike
        The output file path (e.g., 'train_annotations.json',
        'val_annotations.json'). The format is determined by `soundevent.io`.
    name : str, optional
        An optional name to assign to the saved `AnnotationSet`.
    description : str, optional
        An optional description to assign to the saved `AnnotationSet`.
    audio_dir : Path, optional
        Passed to `soundevent.io.save`. May be used to relativize audio file
        paths within the saved annotations if applicable to the save format.
    """

    annotation_set = data.AnnotationSet(
        name=name,
        description=description,
        clip_annotations=list(dataset),
    )
    io.save(annotation_set, path, audio_dir=audio_dir)
