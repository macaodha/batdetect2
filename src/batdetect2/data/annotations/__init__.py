"""Handles loading of annotation data from various formats.

This module serves as the central dispatcher for parsing annotation data
associated with BatDetect2 datasets. Datasets can be composed of multiple
sources, each potentially using a different annotation format (e.g., the
standard AOEF/soundevent format, or legacy BatDetect2 formats).

This module defines the `AnnotationFormats` type, which represents the union
of possible configuration models for these different formats (each identified
by a unique `format` field). The primary function, `load_annotated_dataset`,
inspects the configuration for a single data source and calls the appropriate
format-specific loading function to retrieve the annotations as a standard
`soundevent.data.AnnotationSet`.
"""

from typing import Annotated

from pydantic import Field
from soundevent import data

from batdetect2.data.annotations.aoef import AOEFAnnotations
from batdetect2.data.annotations.batdetect2 import (
    AnnotationFilter,
    BatDetect2FilesAnnotations,
    BatDetect2MergedAnnotations,
)
from batdetect2.data.annotations.registry import annotation_format_registry
from batdetect2.data.annotations.types import AnnotatedDataset

__all__ = [
    "AOEFAnnotations",
    "AnnotatedDataset",
    "AnnotationFilter",
    "AnnotationFormats",
    "BatDetect2FilesAnnotations",
    "BatDetect2MergedAnnotations",
    "load_annotated_dataset",
]


AnnotationFormats = Annotated[
    BatDetect2MergedAnnotations | BatDetect2FilesAnnotations | AOEFAnnotations,
    Field(discriminator="format"),
]
"""Type Alias representing all supported data source configurations.

Each specific configuration model within this union (e.g., `AOEFAnnotations`,
`BatDetect2FilesAnnotations`) corresponds to a different annotation format
or storage structure. These models are typically discriminated by a `format`
field (e.g., `format="aoef"`, `format="batdetect2_files"`), allowing Pydantic
and functions like `load_annotated_dataset` to determine which format a given
source configuration represents.
"""


def load_annotated_dataset(
    dataset: AnnotatedDataset,
    base_dir: data.PathLike | None = None,
) -> data.AnnotationSet:
    """Load annotations for a single data source based on its configuration.

    This function acts as a dispatcher. It inspects the format of the input
    `dataset` object and delegates to the appropriate format-specific loader
    registered in the `annotation_format_registry` (e.g.,
    `AOEFLoader` for `AOEFAnnotations`).

    Parameters
    ----------
    dataset : AnnotatedDataset
        The configuration object for the data source, specifying its format
        and necessary details (like paths). Must be an instance of one of the
        types included in the `AnnotationFormats` union.
    base_dir : Path, optional
        An optional base directory path. If provided, relative paths within
        the `dataset` will be resolved relative to this directory by
        the underlying loading functions. Defaults to None.

    Returns
    -------
    soundevent.data.AnnotationSet
        An AnnotationSet containing the `ClipAnnotation` objects loaded and
        parsed from the specified data source.

    Raises
    ------
    NotImplementedError
        If the `format` field of `dataset` does not match any registered
        annotation format loader.
    """
    loader = annotation_format_registry.build(dataset)
    return loader.load(base_dir=base_dir)
