"""Loads annotation data from legacy BatDetect2 JSON formats.

This module provides backward compatibility for loading annotation data stored
in two related formats used by older BatDetect2 tools:

1.  **`batdetect2` format** (Directory-based): Annotations are stored in
    individual JSON files (one per audio recording) within a specified
    directory.
    Each JSON file contains a `FileAnnotation` structure. Loaded via
    `load_batdetect2_files_annotated_dataset` defined by
    `BatDetect2FilesAnnotations`.
2.  **`batdetect2_file` format** (Single-file): Annotations for multiple
    recordings are merged into a single JSON file, containing a list of
    `FileAnnotation` objects. Loaded via
    `load_batdetect2_merged_annotated_dataset` defined by
    `BatDetect2MergedAnnotations`.

Both formats use the same internal structure for annotations per file and
support filtering based on `annotated` and `issues` flags within that
structure.

The loading functions convert data from these legacy formats into the modern
`soundevent` data model (primarily `ClipAnnotation`) and return the results
aggregated into a `soundevent.data.AnnotationSet`.
"""

import json
import os
from pathlib import Path
from typing import Literal, Optional, Union

from loguru import logger
from pydantic import Field, ValidationError
from soundevent import data

from batdetect2.configs import BaseConfig
from batdetect2.data.annotations.legacy import (
    FileAnnotation,
    file_annotation_to_clip,
    file_annotation_to_clip_annotation,
    list_file_annotations,
    load_file_annotation,
)
from batdetect2.data.annotations.types import AnnotatedDataset

PathLike = Union[Path, str, os.PathLike]


__all__ = [
    "load_batdetect2_files_annotated_dataset",
    "load_batdetect2_merged_annotated_dataset",
    "BatDetect2FilesAnnotations",
    "BatDetect2MergedAnnotations",
    "AnnotationFilter",
]


class AnnotationFilter(BaseConfig):
    """Configuration for filtering legacy FileAnnotations based on flags.

    Specifies criteria based on boolean flags (`annotated` and `issues`)
    present within the legacy `FileAnnotation` JSON structure to select which
    entries (either files or records within a merged file) should be loaded and
    converted.

    Attributes
    ----------
    only_annotated : bool, default=True
        If True, only process entries where the `annotated` flag in the JSON
        is set to `True`.
    exclude_issues : bool, default=True
        If True, skip processing entries where the `issues` flag in the JSON
        is set to `True`.
    """

    only_annotated: bool = True
    exclude_issues: bool = True


class BatDetect2FilesAnnotations(AnnotatedDataset):
    """Configuration for the legacy 'batdetect2' format (directory-based).

    Defines a data source where annotations are stored as individual JSON files
    (one per recording, containing a `FileAnnotation` structure) within the
    `annotations_dir`. Requires a corresponding `audio_dir`. Assumes a naming
    convention links audio files to JSON files
    (e.g., `rec.wav` -> `rec.wav.json`).

    Attributes
    ----------
    format : Literal["batdetect2"]
        The fixed format identifier for this configuration type.
    annotations_dir : Path
        Path to the directory containing the individual JSON annotation files.
    filter : AnnotationFilter, optional
        Configuration for filtering which files to process based on their
        `annotated` and `issues` flags. Defaults to requiring `annotated=True`
        and `issues=False`. Set explicitly to `None` in config (e.g.,
        `filter: null`) to disable filtering.
    """

    format: Literal["batdetect2"] = "batdetect2"
    annotations_dir: Path

    filter: Optional[AnnotationFilter] = Field(
        default_factory=AnnotationFilter,
    )


class BatDetect2MergedAnnotations(AnnotatedDataset):
    """Configuration for the legacy 'batdetect2_file' format (merged file).

    Defines a data source where annotations for multiple recordings (each as a
    `FileAnnotation` structure) are stored within a single JSON file specified
    by `annotations_path`. Audio files are expected in `audio_dir`.

    Inherits `name`, `description`, and `audio_dir` from `AnnotatedDataset`.

    Attributes
    ----------
    format : Literal["batdetect2_file"]
        The fixed format identifier for this configuration type.
    annotations_path : Path
        Path to the single JSON file containing a list of `FileAnnotation`
        objects.
    filter : AnnotationFilter, optional
        Configuration for filtering which `FileAnnotation` entries within the
        merged file to process based on their `annotated` and `issues` flags.
        Defaults to requiring `annotated=True` and `issues=False`. Set to `None`
        in config (e.g., `filter: null`) to disable filtering.
    """

    format: Literal["batdetect2_file"] = "batdetect2_file"
    annotations_path: Path

    filter: Optional[AnnotationFilter] = Field(
        default_factory=AnnotationFilter,
    )


def load_batdetect2_files_annotated_dataset(
    dataset: BatDetect2FilesAnnotations,
    base_dir: Optional[PathLike] = None,
) -> data.AnnotationSet:
    """Load and convert 'batdetect2_file' annotations into an AnnotationSet.

    Scans the specified `annotations_dir` for individual JSON annotation files.
    For each file: loads the legacy `FileAnnotation`, applies filtering based
    on `dataset.filter` (`annotated`/`issues` flags), attempts to find the
    corresponding audio file, converts valid entries to `ClipAnnotation`, and
    collects them into a single `soundevent.data.AnnotationSet`.

    Parameters
    ----------
    dataset : BatDetect2FilesAnnotations
        Configuration describing the 'batdetect2' (directory) data source.
    base_dir : PathLike, optional
        Optional base directory to resolve relative paths in `dataset.audio_dir`
        and `dataset.annotations_dir`. Defaults to None.

    Returns
    -------
    soundevent.data.AnnotationSet
        An AnnotationSet containing all successfully loaded, filtered, and
        converted `ClipAnnotation` objects.

    Raises
    ------
    FileNotFoundError
        If the `annotations_dir` or `audio_dir` does not exist. Errors finding
        individual JSON or audio files during iteration are logged and skipped.
    """
    audio_dir = dataset.audio_dir
    path = dataset.annotations_dir

    if base_dir:
        audio_dir = base_dir / audio_dir
        path = base_dir / path

    paths = list_file_annotations(path)
    logger.debug(
        "Found {num_files} files in the annotations directory {path}",
        num_files=len(paths),
        path=path,
    )

    annotations = []

    for p in paths:
        try:
            file_annotation = load_file_annotation(p)
        except (FileNotFoundError, ValidationError):
            logger.warning("Could not load annotations in file {path}", path=p)
            continue

        if (
            dataset.filter
            and dataset.filter.only_annotated
            and not file_annotation.annotated
        ):
            logger.debug(
                "Annotation in file {path} omited: not annotated",
                path=p,
            )
            continue

        if (
            dataset.filter
            and dataset.filter.exclude_issues
            and file_annotation.issues
        ):
            logger.debug(
                "Annotation in file {path} omited: has issues",
                path=p,
            )
            continue

        try:
            clip = file_annotation_to_clip(
                file_annotation,
                audio_dir=audio_dir,
            )
        except FileNotFoundError as err:
            logger.warning(
                "Did not find the audio related to the annotation file {path}. Error: {err}",
                path=p,
                err=err,
            )
            continue

        annotations.append(
            file_annotation_to_clip_annotation(
                file_annotation,
                clip,
            )
        )

    return data.AnnotationSet(
        name=dataset.name,
        description=dataset.description,
        clip_annotations=annotations,
    )


def load_batdetect2_merged_annotated_dataset(
    dataset: BatDetect2MergedAnnotations,
    base_dir: Optional[PathLike] = None,
) -> data.AnnotationSet:
    """Load and convert 'batdetect2_merged' annotations into an AnnotationSet.

    Loads a single JSON file containing a list of legacy `FileAnnotation`
    objects. For each entry in the list: applies filtering based on
    `dataset.filter` (`annotated`/`issues` flags), attempts to find the
    corresponding audio file, converts valid entries to `ClipAnnotation`, and
    collects them into a single `soundevent.data.AnnotationSet`.

    Parameters
    ----------
    dataset : BatDetect2MergedAnnotations
        Configuration describing the 'batdetect2_file' (merged) data source.
    base_dir : PathLike, optional
        Optional base directory to resolve relative paths in `dataset.audio_dir`
        and `dataset.annotations_path`. Defaults to None.

    Returns
    -------
    soundevent.data.AnnotationSet
        An AnnotationSet containing all successfully loaded, filtered, and
        converted `ClipAnnotation` objects from the merged file.

    Raises
    ------
    FileNotFoundError
        If the `annotations_path` or `audio_dir` does not exist. Errors
        finding individual audio files referenced within the JSON are logged
        and skipped.
    json.JSONDecodeError
        If the annotations file is not valid JSON.
    TypeError
        If the root JSON structure is not a list.
    pydantic.ValidationError
        If entries within the JSON list do not conform to the legacy
        `FileAnnotation` structure.
    """
    audio_dir = dataset.audio_dir
    path = dataset.annotations_path

    if base_dir:
        audio_dir = base_dir / audio_dir
        path = base_dir / path

    content = json.loads(Path(path).read_text())

    if not isinstance(content, list):
        raise TypeError(
            f"Expected a list of FileAnnotations, but got {type(content)}",
        )

    annotations = []

    for ann in content:
        try:
            ann = FileAnnotation.model_validate(ann)
        except ValueError:
            continue

        if (
            dataset.filter
            and dataset.filter.only_annotated
            and not ann.annotated
        ):
            continue

        if dataset.filter and dataset.filter.exclude_issues and ann.issues:
            continue

        try:
            clip = file_annotation_to_clip(ann, audio_dir=audio_dir)
        except FileNotFoundError:
            continue

        annotations.append(file_annotation_to_clip_annotation(ann, clip))

    return data.AnnotationSet(
        name=dataset.name,
        description=dataset.description,
        clip_annotations=annotations,
    )
