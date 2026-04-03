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
from typing import Sequence

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
    ClipAnnotationConditionConfig,
    SoundEventConditionConfig,
    build_clip_annotation_condition,
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
    sources: list[AnnotationFormats]

    clip_filter: ClipAnnotationConditionConfig | None = None
    sound_event_filter: SoundEventConditionConfig | None = None
    sound_event_transforms: list[SoundEventTransformConfig] = Field(
        default_factory=list
    )


def load_dataset(
    config: DatasetConfig,
    base_dir: data.PathLike | None = None,
    add_source_tag: bool = True,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
    apply_transforms: bool = True,
    apply_filters: bool = True,
) -> Dataset:
    """Load and merge clip annotations from configured dataset sources.

    Loads each source listed in ``config.sources`` and returns a flat
    collection of ``soundevent.data.ClipAnnotation`` objects. Source tags,
    dataset-level filters, and dataset-level transforms can be enabled or
    disabled with flags.

    Parameters
    ----------
    config : DatasetConfig
        Dataset definition containing source configurations, optional
        clip-level filter, sound-event filter, and optional sound-event
        transform pipeline.
    base_dir : data.PathLike, optional
        Base directory used to resolve relative paths in source
        configurations.
    add_source_tag : bool, default=True
        If True, append a ``data_source`` tag to each clip annotation with
        the source name.
    include_sources : list[str], optional
        Source names to include. If None, all sources are eligible.
    exclude_sources : list[str], optional
        Source names to skip after include filtering. If a source appears in
        both include and exclude lists, it is skipped.
    apply_transforms : bool, default=True
        If True, apply transforms defined in
        ``config.sound_event_transforms``.
    apply_filters : bool, default=True
        If True, apply filters defined in ``config.clip_filter`` and
        ``config.sound_event_filter``.

    Returns
    -------
    Dataset
        Flat collection of clip annotations loaded from the selected sources.
    """
    clip_annotations = []

    clip_condition = (
        build_clip_annotation_condition(
            config.clip_filter,
            base_dir=base_dir,
        )
        if config.clip_filter is not None
        else None
    )

    sound_event_condition = (
        build_sound_event_condition(
            config.sound_event_filter,
            base_dir=base_dir,
        )
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

        if include_sources and source.name not in include_sources:
            continue

        if exclude_sources and source.name in exclude_sources:
            continue

        logger.debug(
            "Loaded {num_examples} from dataset source '{source_name}'",
            num_examples=len(annotated_source.clip_annotations),
            source_name=source.name,
        )

        for clip_annotation in annotated_source.clip_annotations:
            if add_source_tag:
                clip_annotation = insert_source_tag(clip_annotation, source)

            if (
                clip_condition is not None
                and apply_filters
                and not clip_condition(clip_annotation)
            ):
                continue

            if sound_event_condition is not None and apply_filters:
                clip_annotation = filter_clip_annotation(
                    clip_annotation,
                    sound_event_condition,
                )

            if transform is not None and apply_transforms:
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
    add_source_tag: bool = True,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
    apply_transforms: bool = True,
    apply_filters: bool = True,
) -> Dataset:
    """Load a dataset by reading a ``DatasetConfig`` from disk.

    This convenience wrapper first loads a ``DatasetConfig`` from ``path``
    and optional ``field``, then delegates to :func:`load_dataset`.

    Parameters
    ----------
    path : data.PathLike
        Path to a configuration file containing a ``DatasetConfig``.
    field : str, optional
        Dot-separated field path to a nested config section. If None, the
        full file is parsed as ``DatasetConfig``.
    base_dir : data.PathLike, optional
        Base directory used to resolve relative paths in source
        configurations.
    add_source_tag : bool, default=True
        If True, append a ``data_source`` tag to each clip annotation.
    include_sources : list[str], optional
        Source names to include. If None, all sources are eligible.
    exclude_sources : list[str], optional
        Source names to skip after include filtering.
    apply_transforms : bool, default=True
        If True, apply transforms defined in the loaded config.
    apply_filters : bool, default=True
        If True, apply clip and sound-event filters defined in the loaded
        config.

    Returns
    -------
    Dataset
        Flat collection of clip annotations loaded from the selected sources.
    """
    config = load_config(
        path=path,
        schema=DatasetConfig,
        field=field,
    )
    return load_dataset(
        config,
        base_dir=base_dir,
        add_source_tag=add_source_tag,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
        apply_transforms=apply_transforms,
        apply_filters=apply_filters,
    )


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
