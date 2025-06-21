"""Loads annotation data specifically from the AOEF / soundevent format.

This module provides the necessary configuration model and loading function
to handle data sources where annotations are stored in the standard format
used by the `soundevent` library (often as `.json` or `.aoef` files),
which includes outputs from annotation tools like Whombat.

It supports loading both simple `AnnotationSet` files and more complex
`AnnotationProject` files. For `AnnotationProject` files, it offers optional
filtering capabilities to select only annotations associated with tasks
that meet specific status criteria (e.g., completed, verified, without issues).
"""

from pathlib import Path
from typing import Literal, Optional
from uuid import uuid5

from pydantic import Field
from soundevent import data, io

from batdetect2.configs import BaseConfig
from batdetect2.data.annotations.types import AnnotatedDataset

__all__ = [
    "AOEFAnnotations",
    "load_aoef_annotated_dataset",
    "AnnotationTaskFilter",
]


class AnnotationTaskFilter(BaseConfig):
    """Configuration for filtering Annotation Tasks within an AnnotationProject.

    Specifies criteria based on task status badges to select relevant
    annotations, typically used when loading data from annotation projects
    that might contain work-in-progress.

    Attributes
    ----------
    only_completed : bool, default=True
        If True, only include annotations from tasks marked as 'completed'.
    only_verified : bool, default=False
        If True, only include annotations from tasks marked as 'verified'.
    exclude_issues : bool, default=True
        If True, exclude annotations from tasks marked as 'rejected' (indicating
        issues).
    """

    only_completed: bool = True
    only_verified: bool = False
    exclude_issues: bool = True


class AOEFAnnotations(AnnotatedDataset):
    """Configuration defining a data source stored in AOEF format.

    This model specifies how to load annotations from an AOEF (JSON file) file
    compatible with the `soundevent` library. It inherits `name`,
    `description`, and `audio_dir` from `AnnotatedDataset`.

    Attributes
    ----------
    format : Literal["aoef"]
        The fixed format identifier for this configuration type.
    annotations_path : Path
        The file system path to the `.aoef` or `.json` file containing the
        `AnnotationSet` or `AnnotationProject`.
    filter : AnnotationTaskFilter, optional
        Configuration for filtering tasks if the `annotations_path` points to
        an `AnnotationProject`. If omitted, default filtering
        (only completed, exclude issues, verification not required) is applied
        to projects. Set explicitly to `None` in config (e.g., `filter: null`)
        to disable filtering for projects entirely.
    """

    format: Literal["aoef"] = "aoef"

    annotations_path: Path

    filter: Optional[AnnotationTaskFilter] = Field(
        default_factory=AnnotationTaskFilter
    )


def load_aoef_annotated_dataset(
    dataset: AOEFAnnotations,
    base_dir: Optional[Path] = None,
) -> data.AnnotationSet:
    """Load annotations from an AnnotationSet or AnnotationProject file.

    Reads the file specified in the `dataset` configuration using
    `soundevent.io.load`. If the loaded file contains an `AnnotationProject`
    and filtering is enabled via `dataset.filter`, it applies the filter
    criteria based on task status and returns a new `AnnotationSet` containing
    only the selected annotations. If the file contains an `AnnotationSet`,
    or if it's a project and filtering is disabled, the all annotations are
    returned.

    Parameters
    ----------
    dataset : AOEFAnnotations
        The configuration object describing the AOEF data source, including
        the path to the annotation file and optional filtering settings.
    base_dir : Path, optional
        An optional base directory. If provided, `dataset.annotations_path`
        and `dataset.audio_dir` will be resolved relative to this
        directory. Defaults to None.

    Returns
    -------
    soundevent.data.AnnotationSet
        An AnnotationSet containing the loaded (and potentially filtered)
        `ClipAnnotation` objects.

    Raises
    ------
    FileNotFoundError
        If the specified `annotations_path` (after resolving `base_dir`)
        does not exist.
    ValueError
        If the loaded file does not contain a valid `AnnotationSet` or
        `AnnotationProject`.
    Exception
        May re-raise errors from `soundevent.io.load` related to parsing
        or file format issues.

    Notes
    -----
    - The `soundevent` library handles parsing of `.json` or `.aoef` formats.
    - If an `AnnotationProject` is loaded and `dataset.filter` is *not* None,
      a *new* `AnnotationSet` instance is created containing only the filtered
      clip annotations.
    """
    audio_dir = dataset.audio_dir
    path = dataset.annotations_path

    if base_dir:
        audio_dir = base_dir / audio_dir
        path = base_dir / path

    loaded = io.load(path, audio_dir=audio_dir)

    if not isinstance(loaded, (data.AnnotationSet, data.AnnotationProject)):
        raise ValueError(
            f"The file at {path} loaded successfully but does not "
            "contain a soundevent AnnotationSet or AnnotationProject "
            f"(loaded type: {type(loaded).__name__})."
        )

    if isinstance(loaded, data.AnnotationProject) and dataset.filter:
        loaded = filter_ready_clips(
            loaded,
            only_completed=dataset.filter.only_completed,
            only_verified=dataset.filter.only_verified,
            exclude_issues=dataset.filter.exclude_issues,
        )

    return loaded


def select_task(
    annotation_task: data.AnnotationTask,
    only_completed: bool = True,
    only_verified: bool = False,
    exclude_issues: bool = True,
) -> bool:
    """Check if an AnnotationTask meets specified status criteria.

    Evaluates the `status_badges` of the task against the filter flags.

    Parameters
    ----------
    annotation_task : data.AnnotationTask
        The annotation task to check.
    only_completed : bool, default=True
        Task must be marked 'completed' to pass.
    only_verified : bool, default=False
        Task must be marked 'verified' to pass.
    exclude_issues : bool, default=True
        Task must *not* be marked 'rejected' (have issues) to pass.

    Returns
    -------
    bool
        True if the task meets all active filter criteria, False otherwise.
    """
    has_issues = False
    is_completed = False
    is_verified = False

    for badge in annotation_task.status_badges:
        if badge.state == data.AnnotationState.completed:
            is_completed = True
            continue

        if badge.state == data.AnnotationState.rejected:
            has_issues = True
            continue

        if badge.state == data.AnnotationState.verified:
            is_verified = True

    if exclude_issues and has_issues:
        return False

    if only_verified and not is_verified:
        return False

    if only_completed and not is_completed:
        return False

    return True


def filter_ready_clips(
    annotation_project: data.AnnotationProject,
    only_completed: bool = True,
    only_verified: bool = False,
    exclude_issues: bool = True,
) -> data.AnnotationSet:
    """Filter AnnotationProject to create an AnnotationSet of 'ready' clips.

    Iterates through tasks in the project, selects tasks meeting the status
    criteria using `select_task`, and creates a new `AnnotationSet` containing
    only the `ClipAnnotation` objects associated with those selected tasks.

    Parameters
    ----------
    annotation_project : data.AnnotationProject
        The input annotation project.
    only_completed : bool, default=True
        Filter flag passed to `select_task`.
    only_verified : bool, default=False
        Filter flag passed to `select_task`.
    exclude_issues : bool, default=True
        Filter flag passed to `select_task`.

    Returns
    -------
    data.AnnotationSet
        A new annotation set containing only the clip annotations linked to
        tasks that satisfied the filtering criteria. The returned set has a
        deterministic UUID based on the project UUID and filter settings.
    """
    ready_clip_uuids = set()

    for annotation_task in annotation_project.tasks:
        if not select_task(
            annotation_task,
            only_completed=only_completed,
            only_verified=only_verified,
            exclude_issues=exclude_issues,
        ):
            continue

        ready_clip_uuids.add(annotation_task.clip.uuid)

    return data.AnnotationSet(
        uuid=uuid5(
            annotation_project.uuid,
            f"{only_completed}_{only_verified}_{exclude_issues}",
        ),
        name=annotation_project.name,
        description=annotation_project.description,
        clip_annotations=[
            annotation
            for annotation in annotation_project.clip_annotations
            if annotation.clip.uuid in ready_clip_uuids
        ],
    )
