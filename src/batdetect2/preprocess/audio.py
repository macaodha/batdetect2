"""Audio-level transforms applied to waveforms before spectrogram computation.

This module defines ``torch.nn.Module`` transforms that operate on raw
audio tensors and the Pydantic configuration classes that control them.
Each transform is registered in the ``audio_transforms`` registry so that
the pipeline can be assembled from a configuration object.

The supported transforms are:

* ``CenterAudio`` — subtract the DC offset (mean) from the waveform.
* ``ScaleAudio`` — peak-normalise the waveform to the range ``[-1, 1]``.
* ``FixDuration`` — truncate or zero-pad the waveform to a fixed length.
"""

from typing import Annotated, Literal

import torch
from pydantic import Field

from batdetect2.audio import TARGET_SAMPLERATE_HZ
from batdetect2.core import BaseConfig, Registry
from batdetect2.preprocess.common import center_tensor, peak_normalize

__all__ = [
    "CenterAudioConfig",
    "ScaleAudioConfig",
    "FixDurationConfig",
    "build_audio_transform",
]


audio_transforms: Registry[torch.nn.Module, [int]] = Registry(
    "audio_transform"
)
"""Registry mapping audio transform config classes to their builder methods."""


class CenterAudioConfig(BaseConfig):
    """Configuration for the DC-offset removal transform.

    Attributes
    ----------
    name : str
        Fixed identifier; always ``"center_audio"``.
    """

    name: Literal["center_audio"] = "center_audio"


class CenterAudio(torch.nn.Module):
    """Remove the DC offset from an audio waveform.

    Subtracts the global mean of the waveform from every sample,
    centring the signal around zero. This is useful when an analogue
    recording chain introduces a constant voltage bias.
    """

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Subtract the mean from the waveform.

        Parameters
        ----------
        wav : torch.Tensor
            Input waveform tensor of shape ``(samples,)`` or
            ``(channels, samples)``.

        Returns
        -------
        torch.Tensor
            Zero-centred waveform with the same shape as the input.
        """
        return center_tensor(wav)

    @audio_transforms.register(CenterAudioConfig)
    @staticmethod
    def from_config(config: CenterAudioConfig, samplerate: int):
        return CenterAudio()


class ScaleAudioConfig(BaseConfig):
    """Configuration for the peak-normalisation transform.

    Attributes
    ----------
    name : str
        Fixed identifier; always ``"scale_audio"``.
    """

    name: Literal["scale_audio"] = "scale_audio"


class ScaleAudio(torch.nn.Module):
    """Peak-normalise an audio waveform to the range ``[-1, 1]``.

    Divides the waveform by its largest absolute sample value. If the
    waveform is identically zero it is returned unchanged.
    """

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Peak-normalise the waveform.

        Parameters
        ----------
        wav : torch.Tensor
            Input waveform tensor of any shape.

        Returns
        -------
        torch.Tensor
            Normalised waveform with the same shape as the input and
            values in the range ``[-1, 1]``.
        """
        return peak_normalize(wav)

    @audio_transforms.register(ScaleAudioConfig)
    @staticmethod
    def from_config(config: ScaleAudioConfig, samplerate: int):
        return ScaleAudio()


class FixDurationConfig(BaseConfig):
    """Configuration for the fixed-duration transform.

    Attributes
    ----------
    name : str
        Fixed identifier; always ``"fix_duration"``.
    duration : float, default=0.5
        Target duration in seconds. The waveform will be truncated or
        zero-padded to match this length.
    """

    name: Literal["fix_duration"] = "fix_duration"
    duration: float = 0.5


class FixDuration(torch.nn.Module):
    """Ensure a waveform has exactly a specified number of samples.

    If the input is longer than the target length it is truncated from
    the end. If it is shorter, it is zero-padded at the end.

    Parameters
    ----------
    samplerate : int
        Sample rate of the audio in Hz. Used with ``duration`` to
        compute the target number of samples.
    duration : float
        Target duration in seconds.
    """

    def __init__(self, samplerate: int, duration: float):
        super().__init__()
        self.samplerate = samplerate
        self.duration = duration
        self.length = int(samplerate * duration)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Truncate or pad the waveform to the target length.

        Parameters
        ----------
        wav : torch.Tensor
            Input waveform tensor of shape ``(samples,)`` or
            ``(channels, samples)``. The last dimension is adjusted.

        Returns
        -------
        torch.Tensor
            Waveform with exactly ``self.length`` samples along the last
            dimension.
        """
        length = wav.shape[-1]

        if length == self.length:
            return wav

        if length > self.length:
            return wav[: self.length]

        return torch.nn.functional.pad(wav, (0, self.length - length))

    @audio_transforms.register(FixDurationConfig)
    @staticmethod
    def from_config(config: FixDurationConfig, samplerate: int):
        return FixDuration(samplerate=samplerate, duration=config.duration)


AudioTransform = Annotated[
    FixDurationConfig | ScaleAudioConfig | CenterAudioConfig,
    Field(discriminator="name"),
]
"""Discriminated union of all audio transform configuration types.

Use this type when a field should accept any of the supported audio
transforms. Pydantic will select the correct config class based on the
``name`` field.
"""


def build_audio_transform(
    config: AudioTransform,
    samplerate: int = TARGET_SAMPLERATE_HZ,
) -> torch.nn.Module:
    """Build an audio transform module from a configuration object.

    Parameters
    ----------
    config : AudioTransform
        A configuration object for one of the supported audio transforms
        (``CenterAudioConfig``, ``ScaleAudioConfig``, or
        ``FixDurationConfig``).
    samplerate : int, default=256000
        Sample rate of the audio in Hz. Passed to the transform builder;
        some transforms (e.g. ``FixDuration``) use it to convert seconds
        to samples.

    Returns
    -------
    torch.nn.Module
        The constructed audio transform module.
    """
    return audio_transforms.build(config, samplerate)
