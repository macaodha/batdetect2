"""Defines common interfaces (Protocols) for preprocessing components.

This module centralizes the Protocol definitions used throughout the
`batdetect2.preprocess` package. Protocols define expected methods and
signatures, allowing for flexible and interchangeable implementations of
components like audio loaders and spectrogram builders.

Using these protocols ensures that different parts of the preprocessing
pipeline can interact consistently, regardless of the specific underlying
implementation (e.g., different libraries or custom configurations).
"""

from typing import Protocol

import numpy as np
import torch
from soundevent import data

__all__ = [
    "AudioLoader",
    "SpectrogramBuilder",
    "PreprocessorProtocol",
]


class AudioLoader(Protocol):
    """Defines the interface for an audio loading and processing component.

    An AudioLoader is responsible for retrieving audio data corresponding to
    different soundevent objects (files, Recordings, Clips) and applying a
    configured set of initial preprocessing steps. Adhering to this protocol
    allows for different loading strategies or implementations.
    """

    samplerate: int

    def load_file(
        self,
        path: data.PathLike,
        audio_dir: data.PathLike | None = None,
    ) -> np.ndarray:
        """Load and preprocess audio directly from a file path.

        Parameters
        ----------
        path : PathLike
            Path to the audio file.
        audio_dir : PathLike, optional
            A directory prefix to prepend to the path if `path` is relative.

        Raises
        ------
        FileNotFoundError
            If the audio file cannot be found.
        Exception
            If the audio file cannot be loaded or processed.
        """
        ...

    def load_recording(
        self,
        recording: data.Recording,
        audio_dir: data.PathLike | None = None,
    ) -> np.ndarray:
        """Load and preprocess the entire audio for a Recording object.

        Parameters
        ----------
        recording : data.Recording
            The Recording object containing metadata about the audio file.
        audio_dir : PathLike, optional
            A directory where the audio file associated with the recording
            can be found, especially if the path in the recording is relative.

        Returns
        -------
        np.ndarray
            The loaded and preprocessed audio waveform as a 1-D NumPy
            array. Typically loads only the first channel.

        Raises
        ------
        FileNotFoundError
            If the audio file associated with the recording cannot be found.
        Exception
            If the audio file cannot be loaded or processed.
        """
        ...

    def load_clip(
        self,
        clip: data.Clip,
        audio_dir: data.PathLike | None = None,
    ) -> np.ndarray:
        """Load and preprocess the audio segment defined by a Clip object.

        Parameters
        ----------
        clip : data.Clip
            The Clip object specifying the recording and the start/end times
            of the segment to load.
        audio_dir : PathLike, optional
            A directory where the audio file associated with the clip's
            recording can be found.

        Returns
        -------
        np.ndarray
            The loaded and preprocessed audio waveform for the specified
            clip duration as a 1-D NumPy array. Typically loads only the
            first channel.

        Raises
        ------
        FileNotFoundError
            If the audio file associated with the clip cannot be found.
        Exception
            If the audio file cannot be loaded or processed.
        """
        ...


class SpectrogramBuilder(Protocol):
    """Defines the interface for a spectrogram generation component."""

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """Generate a spectrogram from an audio waveform."""
        ...


class PreprocessorProtocol(Protocol):
    """Defines a high-level interface for the complete preprocessing pipeline."""

    max_freq: float

    min_freq: float

    input_samplerate: int

    output_samplerate: float

    def __call__(self, wav: torch.Tensor) -> torch.Tensor: ...

    def generate_spectrogram(self, wav: torch.Tensor) -> torch.Tensor: ...

    def process_audio(self, wav: torch.Tensor) -> torch.Tensor: ...

    def process_spectrogram(self, spec: torch.Tensor) -> torch.Tensor: ...

    def process_numpy(self, wav: np.ndarray) -> np.ndarray:
        """Run the full preprocessing pipeline on a NumPy waveform.

        This default implementation converts the array to a
        ``torch.Tensor``, calls :meth:`__call__`, and converts the
        result back to a NumPy array. Concrete implementations may
        override this for efficiency.

        Parameters
        ----------
        wav : np.ndarray
            Input waveform as a 1-D NumPy array.

        Returns
        -------
        np.ndarray
            Preprocessed spectrogram as a NumPy array.
        """
        return self(torch.tensor(wav)).numpy()
