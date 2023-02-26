import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch

import bat_detect.utils.audio_utils as au
import bat_detect.utils.detector_utils as du
from bat_detect.detector.parameters import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSING_CONFIGURATIONS,
    DEFAULT_SPECTROGRAM_PARAMETERS,
    TARGET_SAMPLERATE_HZ,
)
from bat_detect.types import (
    Annotation,
    DetectionModel,
    ModelOutput,
    ProcessingConfiguration,
    SpectrogramParameters,
)
from bat_detect.utils.detector_utils import list_audio_files, load_model

# Remove warnings from torch
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

__all__ = [
    "config",
    "generate_spectrogram",
    "get_config",
    "list_audio_files",
    "load_audio",
    "load_model",
    "model",
    "postprocess",
    "process_audio",
    "process_file",
    "process_spectrogram",
]


# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Default model
MODEL, PARAMS = load_model(DEFAULT_MODEL_PATH, device=DEVICE)


def get_config(**kwargs) -> ProcessingConfiguration:
    """Get default processing configuration.

    Can be used to override default parameters by passing keyword arguments.
    """
    return {**DEFAULT_PROCESSING_CONFIGURATIONS, **kwargs}  # type: ignore


# Default processing configuration
CONFIG = get_config(**PARAMS)


def load_audio(
    path: str,
    time_exp_fact: float = 1,
    target_samp_rate: int = TARGET_SAMPLERATE_HZ,
    scale: bool = False,
    max_duration: Optional[float] = None,
) -> np.ndarray:
    """Load audio from file.

    All audio will be resampled to the target sample rate. If the audio is
    longer than max_duration, it will be truncated to max_duration.

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
    """
    _, audio = au.load_audio(
        path,
        time_exp_fact,
        target_samp_rate,
        scale,
        max_duration,
    )
    return audio


def generate_spectrogram(
    audio: np.ndarray,
    samp_rate: int = TARGET_SAMPLERATE_HZ,
    config: Optional[SpectrogramParameters] = None,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """Generate spectrogram from audio array.

    Parameters
    ----------
    audio : np.ndarray
        Audio data.
    samp_rate : int, optional
        Sample rate. Defaults to 256000 which is the target sample rate of
        the default model. Only change if you loaded the audio with a
        different sample rate.
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
    model: DetectionModel = MODEL,
    config: Optional[ProcessingConfiguration] = None,
    device: torch.device = DEVICE,
) -> du.RunResults:
    """Process audio file with model.

    Parameters
    ----------
    audio_file : str
        Path to audio file.
    model : DetectionModel, optional
        Detection model. Uses default model if not specified.
    config : Optional[ProcessingConfiguration], optional
        Processing configuration, by default None (uses default parameters).
    device : torch.device, optional
        Device to use, by default tries to use GPU if available.
    """
    if config is None:
        config = CONFIG

    return du.process_file(
        audio_file,
        model,
        config,
        device,
    )


def process_spectrogram(
    spec: torch.Tensor,
    samp_rate: int = TARGET_SAMPLERATE_HZ,
    model: DetectionModel = MODEL,
    config: Optional[ProcessingConfiguration] = None,
) -> Tuple[List[Annotation], List[np.ndarray]]:
    """Process spectrogram with model.

    Parameters
    ----------
    spec : torch.Tensor
        Spectrogram.
    samp_rate : int, optional
        Sample rate of the audio from which the spectrogram was generated.
        Defaults to 256000 which is the target sample rate of the default
        model. Only change if you generated the spectrogram with a different
        sample rate.
    model : DetectionModel, optional
        Detection model. Uses default model if not specified.
    config : Optional[ProcessingConfiguration], optional
        Processing configuration, by default None (uses default parameters).

    Returns
    -------
    DetectionResult
    """
    if config is None:
        config = CONFIG

    return du.process_spectrogram(
        spec,
        samp_rate,
        model,
        config,
    )


def process_audio(
    audio: np.ndarray,
    samp_rate: int = TARGET_SAMPLERATE_HZ,
    model: DetectionModel = MODEL,
    config: Optional[ProcessingConfiguration] = None,
    device: torch.device = DEVICE,
) -> Tuple[List[Annotation], List[np.ndarray], torch.Tensor]:
    """Process audio array with model.

    Parameters
    ----------
    audio : np.ndarray
        Audio data.
    samp_rate : int, optional
        Sample rate, by default 256000. Only change if you loaded the audio
        with a different sample rate.
    model : DetectionModel, optional
        Detection model. Uses default model if not specified.
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
        config = CONFIG

    return du.process_audio_array(
        audio,
        samp_rate,
        model,
        config,
        device,
    )


def postprocess(
    outputs: ModelOutput,
    samp_rate: int = TARGET_SAMPLERATE_HZ,
    config: Optional[ProcessingConfiguration] = None,
) -> Tuple[List[Annotation], np.ndarray]:
    """Postprocess model outputs.

    Convert model tensor outputs to predicted bounding boxes and
    extracted features.

    Will run non-maximum suppression and remove overlapping annotations.

    Parameters
    ----------
    outputs : ModelOutput
        Model raw outputs.
    samp_rate : int, Optional
        Sample rate of the audio from which the spectrogram was generated.
        Defaults to 256000 which is the target sample rate of the default
        model. Only change if you generated outputs from a spectrogram with
        sample rate.
    config : Optional[ProcessingConfiguration], Optional
        Processing configuration, by default None (uses default parameters).

    Returns
    -------
    annotations : List[Annotation]
        List of predicted annotations.
    features: np.ndarray
        An array of extracted features for each annotation. The shape of the
        array is (n_annotations, n_features).
    """
    if config is None:
        config = CONFIG

    return du.postprocess_model_outputs(
        outputs,
        samp_rate,
        config,
    )


model: DetectionModel = MODEL
"""Base detection model."""

config: ProcessingConfiguration = CONFIG
"""Default processing configuration."""
