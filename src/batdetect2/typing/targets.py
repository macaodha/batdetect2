"""Defines the core interface (Protocol) for the target definition pipeline.

This module specifies the standard structure, attributes, and methods expected
from an object that encapsulates the complete configured logic for processing
sound event annotations within the `batdetect2.targets` system.

The main component defined here is the `TargetProtocol`. This protocol acts as
a contract for the entire target definition process, covering semantic aspects
(filtering, tag transformation, class encoding/decoding) as well as geometric
aspects (mapping regions of interest to target positions and sizes). It ensures
that components responsible for these tasks can be interacted with consistently
throughout BatDetect2.
"""

from collections.abc import Callable
from typing import List, Optional, Protocol

import numpy as np
from soundevent import data

__all__ = [
    "TargetProtocol",
    "SoundEventEncoder",
    "SoundEventDecoder",
    "SoundEventFilter",
    "Position",
    "Size",
]

SoundEventEncoder = Callable[[data.SoundEventAnnotation], Optional[str]]
"""Type alias for a sound event class encoder function.

An encoder function takes a sound event annotation and returns the string name
of the target class it belongs to, based on a predefined set of rules.
If the annotation does not match any defined target class according to the
rules, the function returns None.
"""


SoundEventDecoder = Callable[[str], List[data.Tag]]
"""Type alias for a sound event class decoder function.

A decoder function takes a class name string (as predicted by the model or
assigned during encoding) and returns a list of `soundevent.data.Tag` objects
that represent that class according to the configuration. This is used to
translate model outputs back into meaningful annotations.
"""

SoundEventFilter = Callable[[data.SoundEventAnnotation], bool]
"""Type alias for a filter function.

A filter function accepts a soundevent.data.SoundEventAnnotation object
and returns True if the annotation should be kept based on the filter's
criteria, or False if it should be discarded.
"""

Position = tuple[float, float]
"""A tuple representing (time, frequency) coordinates."""

Size = np.ndarray
"""A NumPy array representing the size dimensions of a target."""


class TargetProtocol(Protocol):
    """Protocol defining the interface for the target definition pipeline.

    This protocol outlines the standard attributes and methods for an object
    that encapsulates the complete, configured process for handling sound event
    annotations (both tags and geometry). It defines how to:
    - Select relevant annotations.
    - Encode an annotation into a specific target class name.
    - Decode a class name back into representative tags.
    - Extract a target reference position from an annotation's geometry (ROI).
    - Calculate target size dimensions from an annotation's geometry.
    - Recover an approximate geometry (ROI) from a position and size
      dimensions.

    Implementations of this protocol bundle all configured logic for these
    steps.

    Attributes
    ----------
    class_names : List[str]
        An ordered list of the unique names of the specific target classes
        defined by the configuration.
    generic_class_tags : List[data.Tag]
        A list of `soundevent.data.Tag` objects representing the configured
        generic class category (e.g., used when no specific class matches).
    dimension_names : List[str]
        A list containing the names of the size dimensions returned by
        `get_size` and expected by `recover_roi` (e.g., ['width', 'height']).
    """

    class_names: List[str]
    """Ordered list of unique names for the specific target classes."""

    generic_class_tags: List[data.Tag]
    """List of tags representing the generic (unclassified) category."""

    dimension_names: List[str]
    """Names of the size dimensions (e.g., ['width', 'height'])."""

    def filter(self, sound_event: data.SoundEventAnnotation) -> bool:
        """Apply the filter to a sound event annotation.

        Determines if the annotation should be included in further processing
        and training based on the configured filtering rules.

        Parameters
        ----------
        sound_event : data.SoundEventAnnotation
            The annotation to filter.

        Returns
        -------
        bool
            True if the annotation should be kept (passes the filter),
            False otherwise. Implementations should return True if no
            filtering is configured.
        """
        ...

    def encode_class(
        self,
        sound_event: data.SoundEventAnnotation,
    ) -> Optional[str]:
        """Encode a sound event annotation to its target class name.

        Parameters
        ----------
        sound_event : data.SoundEventAnnotation
            The (potentially filtered and transformed) annotation to encode.

        Returns
        -------
        str or None
            The string name of the matched target class if the annotation
            matches a specific class definition. Returns None if the annotation
            does not match any specific class rule (indicating it may belong
            to a generic category or should be handled differently downstream).
        """
        ...

    def decode_class(self, class_label: str) -> List[data.Tag]:
        """Decode a predicted class name back into representative tags.

        Parameters
        ----------
        class_label : str
            The class name string (e.g., predicted by a model) to decode.

        Returns
        -------
        List[data.Tag]
            The list of tags corresponding to the input class name according
            to the configuration. May return an empty list or raise an error
            for unmapped labels, depending on the implementation's configuration
            (e.g., `raise_on_unmapped` flag during building).

        Raises
        ------
        ValueError, KeyError
            Implementations might raise an error if the `class_label` is not
            found in the configured mapping and error raising is enabled.
        """
        ...

    def encode_roi(
        self, sound_event: data.SoundEventAnnotation
    ) -> tuple[Position, Size]:
        """Extract the target reference position from the annotation's geometry.

        Calculates the `(time, frequency)` coordinate representing the primary
        location of the sound event.

        Parameters
        ----------
        sound_event : data.SoundEventAnnotation
            The annotation containing the geometry (ROI) to process.

        Returns
        -------
        Tuple[float, float]
            The calculated reference position `(time, frequency)`.

        Raises
        ------
        ValueError
            If the annotation lacks geometry or if the position cannot be
            calculated for the geometry type or configured reference point.
        """
        ...

    # TODO: Update docstrings
    def decode_roi(
        self,
        position: Position,
        size: Size,
        class_name: Optional[str] = None,
    ) -> data.Geometry:
        """Recover the ROI geometry from a position and dimensions.

        Performs the inverse mapping of `get_position` and `get_size`. It takes
        a reference position `(time, frequency)` and an array of size
        dimensions and reconstructs an approximate geometric representation.

        Parameters
        ----------
        pos : Tuple[float, float]
            The reference position `(time, frequency)`.
        dims : np.ndarray
            The NumPy array containing the dimensions (e.g., predicted
            by the model), corresponding to the order in `dimension_names`.
        class_name: str
            class

        Returns
        -------
        soundevent.data.Geometry
            The reconstructed geometry.

        Raises
        ------
        ValueError
            If the number of provided `dims` does not match `dimension_names`,
            if dimensions are invalid (e.g., negative after unscaling), or
            if reconstruction fails based on the configured position type.
        """
        ...


class ROITargetMapper(Protocol):
    """Protocol defining the interface for ROI-to-target mapping.

    Specifies the `encode` and `decode` methods required for converting a
    `soundevent.data.SoundEvent` into a target representation (a reference
    position and a size vector) and for recovering an approximate ROI from that
    representation.

    Attributes
    ----------
    dimension_names : List[str]
        A list containing the names of the dimensions in the `Size` array
        returned by `encode` and expected by `decode`.
    """

    dimension_names: List[str]

    def encode(self, sound_event: data.SoundEvent) -> tuple[Position, Size]:
        """Encode a SoundEvent's geometry into a position and size.

        Parameters
        ----------
        sound_event : data.SoundEvent
            The input sound event, which must have a geometry attribute.

        Returns
        -------
        Tuple[Position, Size]
            A tuple containing:
            - The reference position as (time, frequency) coordinates.
            - A NumPy array with the calculated size dimensions.

        Raises
        ------
        ValueError
            If the sound event does not have a geometry.
        """
        ...

    def decode(self, position: Position, size: Size) -> data.Geometry:
        """Decode a position and size back into a geometric ROI.

        Performs the inverse mapping: takes a reference position and size
        dimensions and reconstructs a geometric representation.

        Parameters
        ----------
        position : Position
            The reference position (time, frequency).
        size : Size
            NumPy array containing the size dimensions, matching the order
            and meaning specified by `dimension_names`.

        Returns
        -------
        soundevent.data.Geometry
            The reconstructed geometry, typically a `BoundingBox`.

        Raises
        ------
        ValueError
            If the `size` array has an unexpected shape or if reconstruction
            fails.
        """
        ...
