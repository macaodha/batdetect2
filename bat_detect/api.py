from typing import List, Optional, Tuple

import numpy as np
import torch

import bat_detect.utils.audio_utils as au
import bat_detect.utils.detector_utils as du
from bat_detect.detector.parameters import (
    DEFAULT_PROCESSING_CONFIGURATIONS,
    DEFAULT_SPECTROGRAM_PARAMETERS,
    TARGET_SAMPLERATE_HZ,
)
from bat_detect.types import (
    Annotation,
    DetectionModel,
    ProcessingConfiguration,
    SpectrogramParameters,
)
from bat_detect.utils.detector_utils import list_audio_files, load_model

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = [
    "load_model",
    "load_audio",
    "list_audio_files",
    "generate_spectrogram",
    "get_config",
    "process_file",
    "process_spectrogram",
    "process_audio",
]


def get_config(**kwargs) -> ProcessingConfiguration:
    """Get default processing configuration.

    Can be used to override default parameters by passing keyword arguments.
    """
    return {**DEFAULT_PROCESSING_CONFIGURATIONS, **kwargs}


def load_audio(
    path: str,
    time_exp_fact: float = 1,
    target_samp_rate: int = TARGET_SAMPLERATE_HZ,
    scale: bool = False,
    max_duration: Optional[float] = None,
) -> Tuple[int, np.ndarray]:
    """Load audio from file.

    Parameters
    ----------
    path : str
        Path to audio file.
    time_exp_fact : float, optional
        Time expansion factor, by default 1
    target_samp_rate : int, optional
        Target sample rate, by default 256000
    scale : bool, optional
        Scale audio to [-1, 1], by default False
    max_duration : Optional[float], optional
        Maximum duration of audio in seconds, by default None

    Returns
    -------
    np.ndarray
        Audio data.
    int
        Sample rate.
    """
    return au.load_audio(
        path,
        time_exp_fact,
        target_samp_rate,
        scale,
        max_duration,
    )


def generate_spectrogram(
    audio: np.ndarray,
    samp_rate: int,
    config: Optional[SpectrogramParameters] = None,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """Generate spectrogram from audio array.

    Parameters
    ----------
    audio : np.ndarray
        Audio data.
    samp_rate : int
        Sample rate.
    config : Optional[SpectrogramParameters], optional
        Spectrogram parameters, by default None (uses default parameters).

    Returns
    -------
    torch.Tensor
        Spectrogram.
    """
    if config is None:
        config = DEFAULT_SPECTROGRAM_PARAMETERS

    _, spec, _ = du.compute_spectrogram(
        audio,
        samp_rate,
        config,
        return_np=False,
        device=device,
    )

    return spec


def process_file(
    audio_file: str,
    model: DetectionModel,
    config: Optional[ProcessingConfiguration] = None,
    device: torch.device = DEVICE,
) -> du.RunResults:
    """Process audio file with model.

    Parameters
    ----------
    audio_file : str
        Path to audio file.
    model : DetectionModel
        Detection model.
    config : Optional[ProcessingConfiguration], optional
        Processing configuration, by default None (uses default parameters).
    device : torch.device, optional
        Device to use, by default tries to use GPU if available.
    """
    if config is None:
        config = DEFAULT_PROCESSING_CONFIGURATIONS

    return du.process_file(
        audio_file,
        model,
        config,
        device,
    )


def process_spectrogram(
    spec: torch.Tensor,
    samp_rate: int,
    model: DetectionModel,
    config: Optional[ProcessingConfiguration] = None,
) -> Tuple[List[Annotation], List[np.ndarray]]:
    """Process spectrogram with model.

    Parameters
    ----------
    spec : torch.Tensor
        Spectrogram.
    samp_rate : int
        Sample rate of the audio from which the spectrogram was generated.
    model : DetectionModel
        Detection model.
    config : Optional[ProcessingConfiguration], optional
        Processing configuration, by default None (uses default parameters).

    Returns
    -------
    DetectionResult
    """
    if config is None:
        config = DEFAULT_PROCESSING_CONFIGURATIONS

    return du.process_spectrogram(
        spec,
        samp_rate,
        model,
        config,
    )


def process_audio(
    audio: np.ndarray,
    samp_rate: int,
    model: DetectionModel,
    config: Optional[ProcessingConfiguration] = None,
    device: torch.device = DEVICE,
) -> Tuple[List[Annotation], List[np.ndarray], torch.Tensor]:
    """Process audio array with model.

    Parameters
    ----------
    audio : np.ndarray
        Audio data.
    samp_rate : int
        Sample rate.
    model : DetectionModel
        Detection model.
    config : Optional[ProcessingConfiguration], optional
        Processing configuration, by default None (uses default parameters).
    device : torch.device, optional
        Device to use, by default tries to use GPU if available.

    Returns
    -------
    annotations : List[Annotation]
        List of predicted annotations.

    features: List[np.ndarray]
        List of extracted features for each annotation.

    spec : torch.Tensor
        Spectrogram of the audio used for prediction.
    """
    if config is None:
        config = DEFAULT_PROCESSING_CONFIGURATIONS

    return du.process_audio_array(
        audio,
        samp_rate,
        model,
        config,
        device,
    )
