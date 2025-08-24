from collections.abc import Generator
from typing import Optional, Tuple

from soundevent import data

from batdetect2.data.datasets import Dataset
from batdetect2.typing.targets import TargetProtocol


def iterate_over_sound_events(
    dataset: Dataset,
    targets: TargetProtocol,
    apply_filter: bool = True,
    apply_transform: bool = True,
    exclude_generic: bool = True,
) -> Generator[Tuple[Optional[str], data.SoundEventAnnotation], None, None]:
    """Iterate over sound events in a dataset, applying filtering and
    transformations.

    This generator function processes sound event annotations from a given
    dataset, allowing for optional filtering, transformation, and exclusion of
    unclassifiable (generic) events based on the provided target definitions.

    Parameters
    ----------
    dataset : Dataset
        The dataset containing clip annotations, each of which may contain
        multiple sound event annotations.
    targets : TargetProtocol
        An object implementing the `TargetProtocol`, which provides methods
        for filtering, transforming, and encoding sound events.
    apply_filter : bool, optional
        If True, sound events will be filtered using `targets.filter()`.
        Only events for which `targets.filter()` returns True will be yielded.
        Defaults to True.
    apply_transform : bool, optional
        If True, sound events will be transformed using `targets.transform()`
        before being yielded. Defaults to True.
    exclude_generic : bool, optional
        If True, sound events that result in a `None` class name after
        `targets.encode()` will be excluded. This is typically used to
        filter out events that cannot be mapped to a specific target class.
        Defaults to True.

    Yields
    ------
    Tuple[Optional[str], data.SoundEventAnnotation]
        A tuple containing:
        - The encoded class name (str) for the sound event, or None if it
          cannot be encoded to a specific class.
        - The sound event annotation itself, after passing all specified
          filtering and transformation steps.

    Notes
    -----
    The processing order for each sound event is:
    1. Filtering (if `apply_filter` is True). Events failing the filter are
       skipped.
    2. Transformation (if `apply_transform` is True).
    3. Encoding to determine class name and check for genericity (if
       `exclude_generic` is True). Events with a `None` class name are skipped
        if `exclude_generic` is True.
    """
    for clip_annotation in dataset:
        for sound_event_annotation in clip_annotation.sound_events:
            if apply_filter:
                if not targets.filter(sound_event_annotation):
                    continue

            if apply_transform:
                sound_event_annotation = targets.transform(
                    sound_event_annotation
                )

            class_name = targets.encode_class(sound_event_annotation)
            if class_name is None and exclude_generic:
                continue

            yield class_name, sound_event_annotation
