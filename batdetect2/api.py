"""Python API for batdetect2.

This module provides a Python API for batdetect2. It can be used to
process audio files or spectrograms with the default model or a custom
model.

Example
-------
You can use the default model to process audio files. To process a single
file, use the `process_file` function.
>>> import batdetect2.api as api
>>> # Process audio file
>>> results = api.process_file("audio_file.wav")

To process multiple files, use the `list_audio_files` function to get a list
of audio files in a directory. Then use the `process_file` function to
process each file.

>>> import batdetect2.api as api
>>> # Get list of audio files
>>> audio_files = api.list_audio_files("audio_directory")
>>> # Process audio files
>>> results = [api.process_file(f) for f in audio_files]

The `process_file` function will slice the recording into 3 second chunks
and process each chunk separately, in case the recording is longer. The
results will be combined into a dictionary with the following keys:

    - `pred_dict`: All the predictions from the model in the format
    expected by the annotation tool.
    - `cnn_feats`: Optional. A list of `numpy` arrays containing the CNN features
    for each detection. The CNN features are the output of the CNN before
    the final classification layer. You can use these features to train
    your own classifier, or to do other processing on the detections.
    They are in the same order as the detections in 
    `results['pred_dict']['annotation']`. Will only be returned if the
    `cnn_feats` parameter in the config is set to `True`.
    - `spec_slices`: Optional. A list of `numpy` arrays containing the spectrogram
    for each of the processed chunks. Will only be returned if the
    `spec_slices` parameter in the config is set to `True`.

Alternatively, you can use the `process_audio` function to process an audio
array directly, or `process_spectrogram` to process spectrograms. This
allows you to do other preprocessing steps before running the model for
predictions.

>>> import batdetect2.api as api
>>> # Load audio
>>> audio = api.load_audio("audio_file.wav")
>>> # Process the audio array
>>> detections, features, spec = api.process_audio(audio)
>>> # Or compute and process the spectrogram
>>> spec = api.generate_spectrogram(audio)
>>> detections, features = api.process_spectrogram(spec)

Here `detections` is the list of detected calls, `features` is the list of
CNN features for each detection, and `spec` is the spectrogram of the
processed audio. Each detection is a dictionary similary to the
following:

    {
        'start_time': 0.0,
        'end_time': 0.1,
        'low_freq': 10000,
        'high_freq': 20000,
        'class': 'Myotis myotis',
        'class_prob': 0.9,
        'det_prob': 0.9,
        'individual': 0,
        'event': 'Echolocation'
    }

If you wish to interact directly with the model, you can use the `model`
attribute to get the default model.

>>> import batdetect2.api as api
>>> # Get the default model
>>> model = api.model
>>> # Process the spectrogram
>>> outputs = model(spec)

However, you will need to do the postprocessing yourself. The
model outputs are a collection of raw tensors. The `postprocess`
function can be used to convert the model outputs into a list of
detections and a list of CNN features.

>>> import batdetect2.api as api
>>> # Get the default model
>>> model = api.model
>>> # Process the spectrogram
>>> outputs = model(spec)
>>> # Postprocess the outputs
>>> detections, features = api.postprocess(outputs)

If you wish to use a custom model or change the default parameters, please
consult the API documentation in the code.

"""
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch

import batdetect2.utils.audio_utils as au
import batdetect2.utils.detector_utils as du
from batdetect2.detector.parameters import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSING_CONFIGURATIONS,
    DEFAULT_SPECTROGRAM_PARAMETERS,
    TARGET_SAMPLERATE_HZ,
)
from batdetect2.types import (
    Annotation,
    DetectionModel,
    ModelOutput,
    ProcessingConfiguration,
    RunResults,
    SpectrogramParameters,
)
from batdetect2.utils.detector_utils import list_audio_files, load_model

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
    "print_summary",
]


# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Default model
MODEL, PARAMS = load_model(DEFAULT_MODEL_PATH, device=DEVICE)


def get_config(**kwargs) -> ProcessingConfiguration:
    """Get default processing configuration.

    Can be used to override default parameters by passing keyword arguments.
    """
    return {**DEFAULT_PROCESSING_CONFIGURATIONS, **PARAMS, **kwargs}  # type: ignore


# Default processing configuration
CONFIG = get_config()


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
    max_duration : float, optional
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
    config : SpectrogramParameters, optional
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
) -> Tuple[List[Annotation], np.ndarray]:
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
    detections : List[Annotation]
        List of detections.
    features: np.ndarray
        An array of features. The array has shape (n_detections, n_features)
        where each row is a feature vector for a detection.
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
) -> Tuple[List[Annotation], np.ndarray, torch.Tensor]:
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
    features: np.ndarray
        An array of features. The array has shape (n_detections, n_features)
        where each row is a feature vector for a detection.
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


def print_summary(results: RunResults) -> None:
    """Print summary of results.

    Parameters
    ----------
    results : DetectionResult
        Detection result.
    """
    print("Results for " + results["pred_dict"]["id"])
    print("{} calls detected\n".format(len(results["pred_dict"]["annotation"])))

    print("time\tprob\tlfreq\tspecies_name")
    for ann in results["pred_dict"]["annotation"]:
        print(
            "{}\t{}\t{}\t{}".format(
                ann["start_time"],
                ann["class_prob"],
                ann["low_freq"],
                ann["class"],
            )
        )
