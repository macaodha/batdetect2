"""Defines common interfaces (Protocols) for preprocessing components.

This module centralizes the Protocol definitions used throughout the
`batdetect2.preprocess` package. Protocols define expected methods and
signatures, allowing for flexible and interchangeable implementations of
components like audio loaders and spectrogram builders.

Using these protocols ensures that different parts of the preprocessing
pipeline can interact consistently, regardless of the specific underlying
implementation (e.g., different libraries or custom configurations).
"""

from typing import Optional, Protocol, Union

import numpy as np
import xarray as xr
from soundevent import data


class AudioLoader(Protocol):
    """Defines the interface for an audio loading and processing component.

    An AudioLoader is responsible for retrieving audio data corresponding to
    different soundevent objects (files, Recordings, Clips) and applying a
    configured set of initial preprocessing steps. Adhering to this protocol
    allows for different loading strategies or implementations.
    """

    def load_file(
        self,
        path: data.PathLike,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        """Load and preprocess audio directly from a file path.

        Parameters
        ----------
        path : PathLike
            Path to the audio file.
        audio_dir : PathLike, optional
            A directory prefix to prepend to the path if `path` is relative.

        Returns
        -------
        xr.DataArray
            The loaded and preprocessed audio waveform as an xarray DataArray
            with time coordinates. Typically loads only the first channel.

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
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
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
        xr.DataArray
            The loaded and preprocessed audio waveform. Typically loads only
            the first channel.

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
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
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
        xr.DataArray
            The loaded and preprocessed audio waveform for the specified clip
            duration. Typically loads only the first channel.

        Raises
        ------
        FileNotFoundError
            If the audio file associated with the clip cannot be found.
        Exception
            If the audio file cannot be loaded or processed.
        """
        ...


class SpectrogramBuilder(Protocol):
    """Defines the interface for a spectrogram generation component.

    A SpectrogramBuilder takes a waveform (as numpy array or xarray DataArray)
    and produces a spectrogram (as an xarray DataArray) based on its internal
    configuration or implementation.
    """

    def __call__(
        self,
        wav: Union[np.ndarray, xr.DataArray],
        samplerate: Optional[int] = None,
    ) -> xr.DataArray:
        """Generate a spectrogram from an audio waveform.

        Parameters
        ----------
        wav : Union[np.ndarray, xr.DataArray]
            The input audio waveform. If a numpy array, `samplerate` must
            also be provided. If an xarray DataArray, it must have a 'time'
            coordinate from which the sample rate can be inferred.
        samplerate : int, optional
            The sample rate of the audio in Hz. Required if `wav` is a
            numpy array. If `wav` is an xarray DataArray, this parameter is
            ignored as the sample rate is derived from the coordinates.

        Returns
        -------
        xr.DataArray
            The computed spectrogram as an xarray DataArray with 'time' and
            'frequency' coordinates.

        Raises
        ------
        ValueError
            If `wav` is a numpy array and `samplerate` is not provided, or
            if `wav` is an xarray DataArray without a valid 'time' coordinate.
        """
        ...


class PreprocessorProtocol(Protocol):
    """Defines a high-level interface for the complete preprocessing pipeline.

    A Preprocessor combines audio loading and spectrogram generation steps.
    It provides methods to go directly from source descriptions (file paths,
    Recording objects, Clip objects) to the final spectrogram representation
    needed by the model. It may also expose intermediate steps like audio
    loading or spectrogram computation from a waveform.
    """

    def preprocess_file(
        self,
        path: data.PathLike,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        """Load audio from a file and compute the final processed spectrogram.

        Performs the full pipeline:

            Load -> Preprocess Audio -> Compute Spectrogram.

        Parameters
        ----------
        path : PathLike
            Path to the audio file.
        audio_dir : PathLike, optional
            A directory prefix if `path` is relative.

        Returns
        -------
        xr.DataArray
            The final processed spectrogram.

        Raises
        ------
        FileNotFoundError
            If the audio file cannot be found.
        Exception
            If any step in the loading or preprocessing fails.
        """
        ...

    def preprocess_recording(
        self,
        recording: data.Recording,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        """Load audio for a Recording and compute the processed spectrogram.

        Performs the full pipeline for the entire duration of the recording.

        Parameters
        ----------
        recording : data.Recording
            The Recording object.
        audio_dir : PathLike, optional
            Directory containing the audio file.

        Returns
        -------
        xr.DataArray
            The final processed spectrogram.

        Raises
        ------
        FileNotFoundError
            If the audio file cannot be found.
        Exception
            If any step in the loading or preprocessing fails.
        """
        ...

    def preprocess_clip(
        self,
        clip: data.Clip,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        """Load audio for a Clip and compute the final processed spectrogram.

        Performs the full pipeline for the specified clip segment.

        Parameters
        ----------
        clip : data.Clip
            The Clip object defining the audio segment.
        audio_dir : PathLike, optional
            Directory containing the audio file.

        Returns
        -------
        xr.DataArray
            The final processed spectrogram.

        Raises
        ------
        FileNotFoundError
            If the audio file cannot be found.
        Exception
            If any step in the loading or preprocessing fails.
        """
        ...

    def load_file_audio(
        self,
        path: data.PathLike,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        """Load and preprocess *only* the audio waveform from a file path.

        Performs the initial audio loading and waveform processing steps
        (like resampling, scaling), but stops *before* spectrogram generation.

        Parameters
        ----------
        path : PathLike
            Path to the audio file.
        audio_dir : PathLike, optional
            A directory prefix if `path` is relative.

        Returns
        -------
        xr.DataArray
            The loaded and preprocessed audio waveform.

        Raises
        ------
        FileNotFoundError, Exception
            If audio loading/preprocessing fails.
        """
        ...

    def load_recording_audio(
        self,
        recording: data.Recording,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        """Load and preprocess *only* the audio waveform for a Recording.

        Performs the initial audio loading and waveform processing steps
        for the entire recording duration.

        Parameters
        ----------
        recording : data.Recording
            The Recording object.
        audio_dir : PathLike, optional
            Directory containing the audio file.

        Returns
        -------
        xr.DataArray
            The loaded and preprocessed audio waveform.

        Raises
        ------
        FileNotFoundError, Exception
            If audio loading/preprocessing fails.
        """
        ...

    def load_clip_audio(
        self,
        clip: data.Clip,
        audio_dir: Optional[data.PathLike] = None,
    ) -> xr.DataArray:
        """Load and preprocess *only* the audio waveform for a Clip.

        Performs the initial audio loading and waveform processing steps
        for the specified clip segment.

        Parameters
        ----------
        clip : data.Clip
            The Clip object defining the segment.
        audio_dir : PathLike, optional
            Directory containing the audio file.

        Returns
        -------
        xr.DataArray
            The loaded and preprocessed audio waveform segment.

        Raises
        ------
        FileNotFoundError, Exception
            If audio loading/preprocessing fails.
        """
        ...

    def compute_spectrogram(
        self,
        wav: Union[xr.DataArray, np.ndarray],
    ) -> xr.DataArray:
        """Compute the spectrogram from a pre-loaded audio waveform.

        Applies the spectrogram generation steps (STFT, scaling, etc.) defined
        by the `SpectrogramBuilder` component of the preprocessor to an
        already loaded (and potentially preprocessed) waveform.

        Parameters
        ----------
        wav : Union[xr.DataArray, np.ndarray]
            The input audio waveform. If numpy array, `samplerate` is required.
        samplerate : int, optional
            Sample rate in Hz (required if `wav` is np.ndarray).

        Returns
        -------
        xr.DataArray
            The computed spectrogram.

        Raises
        ------
        ValueError, Exception
            If waveform input is invalid or spectrogram computation fails.
        """
        ...
