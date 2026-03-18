"""Assembles the full batdetect2 preprocessing pipeline.

This module defines :class:`Preprocessor`, the concrete implementation of
:class:`~batdetect2.preprocess.types.PreprocessorProtocol`, and the
:func:`build_preprocessor` factory function that constructs it from a
:class:`~batdetect2.preprocess.config.PreprocessingConfig`.

The preprocessing pipeline converts a raw audio waveform (as a
``torch.Tensor``) into a normalised, cropped, and resized spectrogram ready
for the detection model. The stages are applied in this order:

1. **Audio transforms** — optional waveform-level operations such as DC
   removal, peak normalisation, or duration fixing.
2. **STFT** — Short-Time Fourier Transform to produce an amplitude
   spectrogram.
3. **Frequency crop** — retain only the frequency band of interest.
4. **Spectrogram transforms** — normalisation operations such as PCEN and
   spectral mean subtraction.
5. **Resize** — scale the spectrogram to the model's expected height and
   reduce the time resolution.
"""

import torch
from loguru import logger

from batdetect2.audio import TARGET_SAMPLERATE_HZ
from batdetect2.preprocess.audio import build_audio_transform
from batdetect2.preprocess.config import PreprocessingConfig
from batdetect2.preprocess.spectrogram import (
    _spec_params_from_config,
    build_spectrogram_builder,
    build_spectrogram_crop,
    build_spectrogram_resizer,
    build_spectrogram_transform,
)
from batdetect2.preprocess.types import PreprocessorProtocol

__all__ = [
    "Preprocessor",
    "build_preprocessor",
]


class Preprocessor(torch.nn.Module, PreprocessorProtocol):
    """Standard implementation of the :class:`~batdetect2.preprocess.types.PreprocessorProtocol`.

    Wraps all preprocessing stages as ``torch.nn.Module`` submodules so
    that parameters (e.g. PCEN filter coefficients) can be tracked and
    moved between devices.

    Parameters
    ----------
    config : PreprocessingConfig
        Full pipeline configuration.
    input_samplerate : int
        Sample rate of the audio that will be passed to this preprocessor,
        in Hz.

    Attributes
    ----------
    input_samplerate : int
        Sample rate of the input audio in Hz.
    output_samplerate : float
        Effective frame rate of the output spectrogram in frames per second.
        Computed from the STFT hop length and the time-axis resize factor.
    min_freq : float
        Lower bound of the retained frequency band in Hz.
    max_freq : float
        Upper bound of the retained frequency band in Hz.
    """

    input_samplerate: int
    output_samplerate: float

    max_freq: float
    min_freq: float

    def __init__(
        self,
        config: PreprocessingConfig,
        input_samplerate: int,
    ) -> None:
        super().__init__()

        self.audio_transforms = torch.nn.Sequential(
            *(
                build_audio_transform(step, samplerate=input_samplerate)
                for step in config.audio_transforms
            )
        )

        self.spectrogram_transforms = torch.nn.Sequential(
            *(
                build_spectrogram_transform(step, samplerate=input_samplerate)
                for step in config.spectrogram_transforms
            )
        )

        self.spectrogram_builder = build_spectrogram_builder(
            config.stft,
            samplerate=input_samplerate,
        )

        self.spectrogram_crop = build_spectrogram_crop(
            config.frequencies,
            stft=config.stft,
            samplerate=input_samplerate,
        )

        self.spectrogram_resizer = build_spectrogram_resizer(config.size)

        self.min_freq = config.frequencies.min_freq
        self.max_freq = config.frequencies.max_freq

        self.input_samplerate = input_samplerate
        self.output_samplerate = compute_output_samplerate(
            config,
            input_samplerate=input_samplerate,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Run the full preprocessing pipeline on a waveform.

        Applies audio transforms, then the STFT, then
        :meth:`process_spectrogram`.

        Parameters
        ----------
        wav : torch.Tensor
            Input waveform of shape ``(samples,)``.

        Returns
        -------
        torch.Tensor
            Preprocessed spectrogram of shape
            ``(freq_bins, time_frames)``.
        """
        wav = self.audio_transforms(wav)
        spec = self.spectrogram_builder(wav)
        return self.process_spectrogram(spec)

    def generate_spectrogram(self, wav: torch.Tensor) -> torch.Tensor:
        """Compute the raw STFT spectrogram without any further processing.

        Parameters
        ----------
        wav : torch.Tensor
            Input waveform of shape ``(samples,)``.

        Returns
        -------
        torch.Tensor
            Amplitude spectrogram of shape ``(n_fft//2 + 1, time_frames)``
            with no frequency cropping, normalisation, or resizing applied.
        """
        return self.spectrogram_builder(wav)

    def process_audio(self, wav: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`forward`.

        Parameters
        ----------
        wav : torch.Tensor
            Input waveform of shape ``(samples,)``.

        Returns
        -------
        torch.Tensor
            Preprocessed spectrogram (same as calling the object directly).
        """
        return self(wav)

    def process_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply the post-STFT processing stages to an existing spectrogram.

        Applies frequency cropping, spectrogram-level transforms (e.g.
        PCEN, spectral mean subtraction), and the final resize step.

        Parameters
        ----------
        spec : torch.Tensor
            Raw amplitude spectrogram of shape
            ``(..., n_fft//2 + 1, time_frames)``.

        Returns
        -------
        torch.Tensor
            Normalised and resized spectrogram of shape
            ``(..., height, scaled_time_frames)``.
        """
        spec = self.spectrogram_crop(spec)
        spec = self.spectrogram_transforms(spec)
        return self.spectrogram_resizer(spec)


def compute_output_samplerate(
    config: PreprocessingConfig,
    input_samplerate: int = TARGET_SAMPLERATE_HZ,
) -> float:
    """Compute the effective frame rate of the preprocessor's output.

    The output frame rate (in frames per second) depends on the STFT hop
    length and the time-axis resize factor applied by the final resize step.

    Parameters
    ----------
    config : PreprocessingConfig
        Pipeline configuration.
    input_samplerate : int, default=256000
        Sample rate of the input audio in Hz.

    Returns
    -------
    float
        Output frame rate in frames per second.
        For example, at the default settings (256 kHz, hop=128,
        resize_factor=0.5) this equals ``1000.0``.
    """
    _, hop_size = _spec_params_from_config(
        config.stft, samplerate=input_samplerate
    )
    factor = config.size.resize_factor
    return input_samplerate * factor / hop_size


def build_preprocessor(
    config: PreprocessingConfig | None = None,
    input_samplerate: int = TARGET_SAMPLERATE_HZ,
) -> PreprocessorProtocol:
    """Build the standard preprocessor from a configuration object.

    Parameters
    ----------
    config : PreprocessingConfig, optional
        Pipeline configuration. If ``None``, the default
        ``PreprocessingConfig()`` is used (PCEN + spectral mean
        subtraction, 256 kHz, standard STFT parameters).
    input_samplerate : int, default=256000
        Sample rate of the audio that will be fed to the preprocessor,
        in Hz.

    Returns
    -------
    PreprocessorProtocol
        A :class:`Preprocessor` instance ready to convert waveforms to
        spectrograms.
    """
    config = config or PreprocessingConfig()
    logger.opt(lazy=True).debug(
        "Building preprocessor with config: \n{}",
        lambda: config.to_yaml_string(),
    )
    return Preprocessor(config=config, input_samplerate=input_samplerate)
