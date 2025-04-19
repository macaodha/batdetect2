"""Defines the core interface (Protocol) for the target definition pipeline.

This module specifies the standard structure and methods expected from an object
that encapsulates the configured logic for processing sound event annotations
within the `batdetect2.targets` system.

The main component defined here is the `TargetEncoder` protocol. This protocol
acts as a contract, ensuring that components responsible for applying
filtering, transformations, encoding annotations to class names, and decoding
class names back to tags can be interacted with in a consistent manner
throughout BatDetect2. It also defines essential metadata attributes expected
from implementations.
"""

from typing import List, Optional, Protocol

from soundevent import data

__all__ = [
    "TargetProtocol",
]


class TargetProtocol(Protocol):
    """Protocol defining the interface for the target definition pipeline.

    This protocol outlines the standard attributes and methods for an object
    that encapsulates the complete, configured process for handling sound event
    annotations to determine their target class for model training, and for
    interpreting model predictions back into annotation tags.

    Attributes
    ----------
    class_names : List[str]
        An ordered list of the unique names of the specific target classes
        defined by the configuration represented by this object.
    generic_class_tags : List[data.Tag]
        A list of `soundevent.data.Tag` objects representing the
        generic class category (e.g., the default 'Bat' class tags used when
        no specific class matches).
    """

    class_names: List[str]
    """Ordered list of unique names for the specific target classes."""

    generic_class_tags: List[data.Tag]
    """List of tags representing the generic (unclassified) category."""

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

    def transform(
        self,
        sound_event: data.SoundEventAnnotation,
    ) -> data.SoundEventAnnotation:
        """Apply tag transformations to an annotation.

        Parameters
        ----------
        sound_event : data.SoundEventAnnotation
            The annotation whose tags should be transformed.

        Returns
        -------
        data.SoundEventAnnotation
            A new annotation object with the transformed tags. Implementations
            should return the original annotation object if no transformations
            were configured.
        """
        ...

    def encode(
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
            The string name of the matched target class if the annotation matches
            a specific class definition. Returns None if the annotation does not
            match any specific class rule (indicating it may belong to a generic
            category or should be handled differently downstream).
        """
        ...

    def decode(self, class_label: str) -> List[data.Tag]:
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
