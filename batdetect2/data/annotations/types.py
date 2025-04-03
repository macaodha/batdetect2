from pathlib import Path
from typing import Literal, Union

from batdetect2.configs import BaseConfig

__all__ = [
    "AnnotatedDataset",
    "BatDetect2MergedAnnotations",
]


class AnnotatedDataset(BaseConfig):
    """Represents a single, cohesive source of audio recordings and annotations.

    A source typically groups recordings originating from a specific context,
    such as a single project, site, deployment, or recordist. All audio files
    belonging to a source should be located within a single directory,
    specified by `audio_dir`.

    Annotations associated with these recordings are defined by the
    `annotations` field, which supports various formats (e.g., AOEF files,
                                                         specific CSV
                                                         structures).
    Crucially, file paths referenced within the annotation data *must* be
    relative to the `audio_dir`. This ensures that the dataset definition
    remains portable across different systems and base directories.

    Attributes:
        name: A unique identifier for this data source.
        description: Detailed information about the source, including recording
            methods, annotation procedures, equipment used, potential biases,
            or any important caveats for users.
        audio_dir: The file system path to the directory containing the audio
            recordings for this source.
    """

    name: str
    audio_dir: Path
    description: str = ""


