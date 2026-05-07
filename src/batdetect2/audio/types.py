from typing import Protocol

import numpy as np
from soundevent import data

__all__ = [
    "AudioLoader",
    "ClipperProtocol",
]


class AudioLoader(Protocol):
    samplerate: int

    def load_file(
        self,
        path: data.PathLike,
        audio_dir: data.PathLike | None = None,
    ) -> np.ndarray: ...

    def load_recording(
        self,
        recording: data.Recording,
        audio_dir: data.PathLike | None = None,
    ) -> np.ndarray: ...

    def load_clip(
        self,
        clip: data.Clip,
        audio_dir: data.PathLike | None = None,
    ) -> np.ndarray: ...


class ClipperProtocol(Protocol):
    def __call__(
        self,
        clip_annotation: data.ClipAnnotation,
    ) -> data.ClipAnnotation: ...

    def get_subclip(self, clip: data.Clip) -> data.Clip: ...
