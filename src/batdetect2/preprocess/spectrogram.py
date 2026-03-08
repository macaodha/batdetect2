"""Computes spectrograms from audio waveforms with configurable parameters.

This module defines the STFT-based spectrogram builder and a collection of
spectrogram-level transforms (PCEN, spectral mean subtraction, amplitude
scaling, peak normalisation, frequency cropping, and resizing) that form the
signal-processing stage of the batdetect2 preprocessing pipeline.

Each transform is paired with a Pydantic configuration class and registered
in the ``spectrogram_transforms`` registry so that the pipeline can be fully
specified via a YAML or Python configuration object.
"""

from typing import Annotated, Callable, Literal

import numpy as np
import torch
import torchaudio
from pydantic import Field

from batdetect2.audio import TARGET_SAMPLERATE_HZ
from batdetect2.core.configs import BaseConfig
from batdetect2.core.registries import Registry
from batdetect2.preprocess.common import peak_normalize

__all__ = [
    "STFTConfig",
    "build_spectrogram_transform",
    "build_spectrogram_builder",
]


MIN_FREQ = 10_000
"""Default minimum frequency (Hz) for spectrogram frequency cropping."""

MAX_FREQ = 120_000
"""Default maximum frequency (Hz) for spectrogram frequency cropping."""


class STFTConfig(BaseConfig):
    """Configuration for the Short-Time Fourier Transform (STFT).

    Attributes
    ----------
    window_duration : float, default=0.002
        Duration of the STFT analysis window in seconds (e.g. 0.002 for
        2 ms). Must be > 0. A longer window gives finer frequency resolution
        but coarser time resolution.
    window_overlap : float, default=0.75
        Fraction of overlap between consecutive windows (e.g. 0.75 for
        75 %). Must be >= 0 and < 1. Higher overlap gives finer time
        resolution at the cost of more computation.
    window_fn : str, default="hann"
        Name of the tapering window applied to each frame before the FFT.
        Supported values: ``"hann"``, ``"hamming"``, ``"kaiser"``,
        ``"blackman"``, ``"bartlett"``.

    Notes
    -----
    At the default sample rate of 256 kHz, ``window_duration=0.002`` and
    ``window_overlap=0.75`` give ``n_fft=512`` and ``hop_length=128``.
    """

    window_duration: float = Field(default=0.002, gt=0)
    window_overlap: float = Field(default=0.75, ge=0, lt=1)
    window_fn: str = "hann"


def build_spectrogram_builder(
    config: STFTConfig,
    samplerate: int = TARGET_SAMPLERATE_HZ,
) -> torch.nn.Module:
    """Build a torchaudio STFT spectrogram module from an ``STFTConfig``.

    Parameters
    ----------
    config : STFTConfig
        STFT parameters (window duration, overlap, and window function).
    samplerate : int, default=256000
        Sample rate of the input audio in Hz. Used to convert the
        window duration into a number of samples.

    Returns
    -------
    torch.nn.Module
        A ``torchaudio.transforms.Spectrogram`` module configured to
        produce an amplitude (``power=1``) spectrogram with centred
        frames.
    """
    n_fft, hop_length = _spec_params_from_config(config, samplerate=samplerate)
    return torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        window_fn=get_spectrogram_window(config.window_fn),
        center=True,
        power=1,
    )


def get_spectrogram_window(name: str) -> Callable[..., torch.Tensor]:
    """Return the PyTorch window function matching the given name.

    Parameters
    ----------
    name : str
        Name of the window function. One of ``"hann"``, ``"hamming"``,
        ``"kaiser"``, ``"blackman"``, or ``"bartlett"``.

    Returns
    -------
    Callable[..., torch.Tensor]
        A PyTorch window function that accepts a window length and returns
        a 1-D tensor of weights.

    Raises
    ------
    NotImplementedError
        If ``name`` does not match any supported window function.
    """
    if name == "hann":
        return torch.hann_window

    if name == "hamming":
        return torch.hamming_window

    if name == "kaiser":
        return torch.kaiser_window

    if name == "blackman":
        return torch.blackman_window

    if name == "bartlett":
        return torch.bartlett_window

    raise NotImplementedError(
        f"Spectrogram window function {name} not implemented"
    )


def _spec_params_from_config(
    config: STFTConfig,
    samplerate: int = TARGET_SAMPLERATE_HZ,
) -> tuple[int, int]:
    """Compute ``n_fft`` and ``hop_length`` from an ``STFTConfig``.

    Parameters
    ----------
    config : STFTConfig
        STFT parameters.
    samplerate : int, default=256000
        Sample rate of the input audio in Hz.

    Returns
    -------
    tuple[int, int]
        A pair ``(n_fft, hop_length)`` giving the FFT size and the step
        between consecutive frames in samples.
    """
    n_fft = int(samplerate * config.window_duration)
    hop_length = int(n_fft * (1 - config.window_overlap))
    return n_fft, hop_length


def _frequency_to_index(
    freq: float,
    n_fft: int,
    samplerate: int = TARGET_SAMPLERATE_HZ,
) -> int | None:
    """Convert a frequency in Hz to the nearest STFT frequency bin index.

    Parameters
    ----------
    freq : float
        Frequency in Hz to convert.
    n_fft : int
        FFT size used by the STFT.
    samplerate : int, default=256000
        Sample rate of the audio in Hz.

    Returns
    -------
    int or None
        The bin index corresponding to ``freq``, or ``None`` if the
        frequency is outside the valid range (i.e. <= 0 Hz or >= the
        Nyquist frequency).
    """
    alpha = freq * 2 / samplerate
    height = np.floor(n_fft / 2) + 1
    index = int(np.floor(alpha * height))

    if index <= 0:
        return None

    if index >= height:
        return None

    return index


class FrequencyConfig(BaseConfig):
    """Configuration for frequency axis parameters.

    Attributes
    ----------
    max_freq : int, default=120000
        Maximum frequency in Hz to retain after STFT. Frequency bins
        above this value are discarded. Must be >= 0.
    min_freq : int, default=10000
        Minimum frequency in Hz to retain after STFT. Frequency bins
        below this value are discarded. Must be >= 0.
    """

    max_freq: int = Field(default=MAX_FREQ, ge=0)
    min_freq: int = Field(default=MIN_FREQ, ge=0)


class FrequencyCrop(torch.nn.Module):
    """Crop a spectrogram to a specified frequency band.

    On construction the Hz boundaries are converted to STFT bin indices.
    During the forward pass the spectrogram is sliced along its
    frequency axis (second-to-last dimension) to retain only the bins
    that fall within ``[min_freq, max_freq)``.

    Parameters
    ----------
    samplerate : int
        Sample rate of the audio in Hz.
    n_fft : int
        FFT size used by the STFT.
    min_freq : int, optional
        Lower frequency bound in Hz. If ``None``, no lower crop is
        applied and the DC bin is retained.
    max_freq : int, optional
        Upper frequency bound in Hz. If ``None``, no upper crop is
        applied and all bins up to Nyquist are retained.
    """

    def __init__(
        self,
        samplerate: int,
        n_fft: int,
        min_freq: int | None = None,
        max_freq: int | None = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.samplerate = samplerate
        self.min_freq = min_freq
        self.max_freq = max_freq

        low_index = None
        if min_freq is not None:
            low_index = _frequency_to_index(
                min_freq,
                n_fft=self.n_fft,
                samplerate=self.samplerate,
            )
        self.low_index = low_index

        high_index = None
        if max_freq is not None:
            high_index = _frequency_to_index(
                max_freq,
                n_fft=self.n_fft,
                samplerate=self.samplerate,
            )
        self.high_index = high_index

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Crop the spectrogram to the configured frequency band.

        Parameters
        ----------
        spec : torch.Tensor
            Spectrogram tensor of shape ``(..., freq_bins, time_frames)``.

        Returns
        -------
        torch.Tensor
            Cropped spectrogram with shape
            ``(..., n_retained_bins, time_frames)``.
        """
        low_index = self.low_index
        if low_index is None:
            low_index = 0

        if self.high_index is None:
            length = spec.shape[-2] - low_index
        else:
            length = self.high_index - low_index

        return torch.narrow(
            spec,
            dim=-2,
            start=low_index,
            length=length,
        )


def build_spectrogram_crop(
    config: FrequencyConfig,
    stft: STFTConfig | None = None,
    samplerate: int = TARGET_SAMPLERATE_HZ,
) -> torch.nn.Module:
    """Build a ``FrequencyCrop`` module from configuration objects.

    Parameters
    ----------
    config : FrequencyConfig
        Frequency boundary configuration specifying ``min_freq`` and
        ``max_freq`` in Hz.
    stft : STFTConfig, optional
        STFT configuration used to derive ``n_fft``. Defaults to
        ``STFTConfig()`` if not provided.
    samplerate : int, default=256000
        Sample rate of the audio in Hz.

    Returns
    -------
    torch.nn.Module
        A ``FrequencyCrop`` module ready to crop spectrograms.
    """
    stft = stft or STFTConfig()
    n_fft, _ = _spec_params_from_config(stft, samplerate=samplerate)
    return FrequencyCrop(
        samplerate=samplerate,
        n_fft=n_fft,
        min_freq=config.min_freq,
        max_freq=config.max_freq,
    )


class ResizeConfig(BaseConfig):
    """Configuration for the final spectrogram resize step.

    Attributes
    ----------
    name : str
        Fixed identifier; always ``"resize_spec"``.
    height : int, default=128
        Target number of frequency bins in the output spectrogram.
        The spectrogram is resized to this height using bilinear
        interpolation.
    resize_factor : float, default=0.5
        Fraction by which the time axis is scaled. For example, ``0.5``
        halves the number of time frames, reducing computational cost
        downstream.
    """

    name: Literal["resize_spec"] = "resize_spec"
    height: int = 128
    resize_factor: float = 0.5


class ResizeSpec(torch.nn.Module):
    """Resize a spectrogram to a fixed height and scaled width.

    Uses bilinear interpolation so it handles arbitrary input shapes
    gracefully. Input tensors with fewer than four dimensions are
    temporarily unsqueezed to satisfy ``torch.nn.functional.interpolate``.

    Parameters
    ----------
    height : int
        Target number of frequency bins (output height).
    time_factor : float
        Multiplicative scaling applied to the time axis length.
    """

    def __init__(self, height: int, time_factor: float):
        super().__init__()
        self.height = height
        self.time_factor = time_factor

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Resize the spectrogram to the configured output dimensions.

        Parameters
        ----------
        spec : torch.Tensor
            Input spectrogram of shape ``(..., freq_bins, time_frames)``.

        Returns
        -------
        torch.Tensor
            Resized spectrogram with shape
            ``(..., height, int(time_factor * time_frames))``.
        """
        current_length = spec.shape[-1]
        target_length = int(self.time_factor * current_length)

        original_ndim = spec.ndim
        while spec.ndim < 4:
            spec = spec.unsqueeze(0)

        resized = torch.nn.functional.interpolate(
            spec,
            size=(self.height, target_length),
            mode="bilinear",
        )

        while resized.ndim != original_ndim:
            resized = resized.squeeze(0)

        return resized


def build_spectrogram_resizer(config: ResizeConfig) -> torch.nn.Module:
    """Build a ``ResizeSpec`` module from a ``ResizeConfig``.

    Parameters
    ----------
    config : ResizeConfig
        Resize configuration specifying ``height`` and ``resize_factor``.

    Returns
    -------
    torch.nn.Module
        A ``ResizeSpec`` module configured with the given parameters.
    """
    return ResizeSpec(height=config.height, time_factor=config.resize_factor)


spectrogram_transforms: Registry[torch.nn.Module, [int]] = Registry(
    "spectrogram_transform"
)


class PcenConfig(BaseConfig):
    """Configuration for Per-Channel Energy Normalisation (PCEN).

    PCEN is a frontend processing technique that replaces simple log
    compression. It applies a learnable automatic gain control followed
    by a stabilised root compression, making the representation more
    robust to variations in recording level.

    Attributes
    ----------
    name : str
        Fixed identifier; always ``"pcen"``.
    time_constant : float, default=0.4
        Time constant (in seconds) of the IIR smoothing filter used
        for the background estimate. Larger values produce a slower-
        adapting background.
    gain : float, default=0.98
        Exponent controlling how strongly the background estimate
        suppresses the signal.
    bias : float, default=2
        Stabilisation bias added inside the root-compression step to
        avoid division by zero.
    power : float, default=0.5
        Root-compression exponent. A value of 0.5 gives square-root
        compression, similar to log compression but differentiable at
        zero.
    """

    name: Literal["pcen"] = "pcen"
    time_constant: float = 0.4
    gain: float = 0.98
    bias: float = 2
    power: float = 0.5


class PCEN(torch.nn.Module):
    """Per-Channel Energy Normalisation (PCEN) transform.

    Applies automatic gain control and root compression to a spectrogram.
    The background estimate is computed with a first-order IIR filter
    applied along the time axis.

    Parameters
    ----------
    smoothing_constant : float
        IIR filter coefficient ``alpha``. Derived from the time constant
        and sample rate via ``_compute_smoothing_constant``.
    gain : float, default=0.98
        AGC gain exponent.
    bias : float, default=2.0
        Root-compression stabilisation bias.
    power : float, default=0.5
        Root-compression exponent.
    eps : float, default=1e-6
        Small constant for numerical stability.
    dtype : torch.dtype, default=torch.float32
        Floating-point precision used for internal computation.

    Notes
    -----
    The smoothing constant is computed to match the original batdetect2
    implementation for numerical compatibility. See
    ``_compute_smoothing_constant`` for details.
    """

    def __init__(
        self,
        smoothing_constant: float,
        gain: float = 0.98,
        bias: float = 2.0,
        power: float = 0.5,
        eps: float = 1e-6,
        dtype=torch.float32,
    ):
        super().__init__()
        self.smoothing_constant = smoothing_constant
        self.gain = torch.tensor(gain, dtype=dtype)
        self.bias = torch.tensor(bias, dtype=dtype)
        self.power = torch.tensor(power, dtype=dtype)
        self.eps = torch.tensor(eps, dtype=dtype)
        self.dtype = dtype

        self._b = torch.tensor([self.smoothing_constant, 0.0], dtype=dtype)
        self._a = torch.tensor(
            [1.0, self.smoothing_constant - 1.0], dtype=dtype
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply PCEN to a spectrogram.

        Parameters
        ----------
        spec : torch.Tensor
            Input amplitude spectrogram of shape
            ``(..., freq_bins, time_frames)``.

        Returns
        -------
        torch.Tensor
            PCEN-normalised spectrogram with the same shape and dtype as
            the input.
        """
        S = spec.to(self.dtype) * 2**31

        M = (
            torchaudio.functional.lfilter(
                S,
                self._a,
                self._b,
                clamp=False,
            )
        ).clamp(min=0)

        smooth = torch.exp(
            -self.gain * (torch.log(self.eps) + torch.log1p(M / self.eps))
        )

        return (
            (self.bias**self.power)
            * torch.expm1(self.power * torch.log1p(S * smooth / self.bias))
        ).to(spec.dtype)

    @spectrogram_transforms.register(PcenConfig)
    @staticmethod
    def from_config(config: PcenConfig, samplerate: int):
        smooth = _compute_smoothing_constant(samplerate, config.time_constant)
        return PCEN(
            smoothing_constant=smooth,
            gain=config.gain,
            bias=config.bias,
            power=config.power,
        )


def _compute_smoothing_constant(
    samplerate: int,
    time_constant: float,
) -> float:
    """Compute the IIR smoothing coefficient for PCEN.

    Parameters
    ----------
    samplerate : int
        Sample rate of the audio in Hz.
    time_constant : float
        Desired smoothing time constant in seconds.

    Returns
    -------
    float
        IIR filter coefficient ``alpha`` used by ``PCEN``.

    Notes
    -----
    The hop length (512) and the sample-rate divisor (10) are fixed to
    reproduce the numerical behaviour of the original batdetect2
    implementation, which used ``librosa.pcen`` with ``sr=samplerate/10``
    and the default ``hop_length=512``. These values do not reflect the
    actual STFT hop length used in the pipeline; they are retained
    solely for backward compatibility.
    """
    # NOTE: These parameters are fixed to match the original implementation.
    hop_length = 512
    sr = samplerate / 10
    t_frames = time_constant * sr / float(hop_length)
    return (np.sqrt(1 + 4 * t_frames**2) - 1) / (2 * t_frames**2)


class ScaleAmplitudeConfig(BaseConfig):
    """Configuration for amplitude scaling of a spectrogram.

    Attributes
    ----------
    name : str
        Fixed identifier; always ``"scale_amplitude"``.
    scale : str, default="db"
        Scaling mode. Either ``"db"`` (convert amplitude to decibels
        using ``torchaudio.transforms.AmplitudeToDB``) or ``"power"``
        (square the amplitude values).
    """

    name: Literal["scale_amplitude"] = "scale_amplitude"
    scale: Literal["power", "db"] = "db"


class ToPower(torch.nn.Module):
    """Square the values of a spectrogram (amplitude → power).

    Raises each element to the power of two, converting an amplitude
    spectrogram into a power spectrogram.
    """

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Square all elements of the spectrogram.

        Parameters
        ----------
        spec : torch.Tensor
            Input amplitude spectrogram.

        Returns
        -------
        torch.Tensor
            Power spectrogram (same shape as input).
        """
        return spec**2


_scalers = {
    "db": torchaudio.transforms.AmplitudeToDB,
    "power": ToPower,
}


class ScaleAmplitude(torch.nn.Module):
    """Convert spectrogram amplitude values to a different scale.

    Supports conversion to decibels (dB) or to power (squared amplitude).

    Parameters
    ----------
    scale : str
        Either ``"db"`` or ``"power"``.
    """

    def __init__(self, scale: Literal["power", "db"]):
        super().__init__()
        self.scale = scale
        self.scaler = _scalers[scale]()

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply the configured amplitude scaling.

        Parameters
        ----------
        spec : torch.Tensor
            Input spectrogram tensor.

        Returns
        -------
        torch.Tensor
            Scaled spectrogram with the same shape as the input.
        """
        return self.scaler(spec)

    @spectrogram_transforms.register(ScaleAmplitudeConfig)
    @staticmethod
    def from_config(config: ScaleAmplitudeConfig, samplerate: int):
        return ScaleAmplitude(scale=config.scale)


class SpectralMeanSubtractionConfig(BaseConfig):
    """Configuration for spectral mean subtraction.

    Attributes
    ----------
    name : str
        Fixed identifier; always ``"spectral_mean_subtraction"``.
    """

    name: Literal["spectral_mean_subtraction"] = "spectral_mean_subtraction"


class SpectralMeanSubtraction(torch.nn.Module):
    """Remove the time-averaged background noise from a spectrogram.

    For each frequency bin, the mean value across all time frames is
    computed and subtracted. The result is then clamped to zero so that
    no values fall below the baseline. This is a simple form of spectral
    denoising that suppresses stationary background noise.
    """

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Subtract the time-axis mean from each frequency bin.

        Parameters
        ----------
        spec : torch.Tensor
            Input spectrogram of shape ``(..., freq_bins, time_frames)``.

        Returns
        -------
        torch.Tensor
            Denoised spectrogram with the same shape as the input. All
            values are non-negative (clamped to 0).
        """
        mean = spec.mean(-1, keepdim=True)
        return (spec - mean).clamp(min=0)

    @spectrogram_transforms.register(SpectralMeanSubtractionConfig)
    @staticmethod
    def from_config(
        config: SpectralMeanSubtractionConfig,
        samplerate: int,
    ):
        return SpectralMeanSubtraction()


class PeakNormalizeConfig(BaseConfig):
    """Configuration for peak normalisation of a spectrogram.

    Attributes
    ----------
    name : str
        Fixed identifier; always ``"peak_normalize"``.
    """

    name: Literal["peak_normalize"] = "peak_normalize"


class PeakNormalize(torch.nn.Module):
    """Scale a spectrogram so that its largest absolute value equals one.

    Wraps :func:`batdetect2.preprocess.common.peak_normalize` as a
    ``torch.nn.Module`` for use inside a sequential transform pipeline.
    """

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Peak-normalise the spectrogram.

        Parameters
        ----------
        spec : torch.Tensor
            Input spectrogram tensor of any shape.

        Returns
        -------
        torch.Tensor
            Normalised spectrogram where the maximum absolute value is 1.
            If the input is identically zero, it is returned unchanged.
        """
        return peak_normalize(spec)

    @spectrogram_transforms.register(PeakNormalizeConfig)
    @staticmethod
    def from_config(config: PeakNormalizeConfig, samplerate: int):
        return PeakNormalize()


SpectrogramTransform = Annotated[
    PcenConfig
    | ScaleAmplitudeConfig
    | SpectralMeanSubtractionConfig
    | PeakNormalizeConfig,
    Field(discriminator="name"),
]
"""Discriminated union of all spectrogram transform configuration types.

Use this type when a field should accept any of the supported spectrogram
transforms. Pydantic will select the correct config class based on the
``name`` field.
"""


def build_spectrogram_transform(
    config: SpectrogramTransform,
    samplerate: int,
) -> torch.nn.Module:
    """Build a spectrogram transform module from a configuration object.

    Parameters
    ----------
    config : SpectrogramTransform
        A configuration object for one of the supported spectrogram
        transforms (PCEN, amplitude scaling, spectral mean subtraction,
        or peak normalisation).
    samplerate : int
        Sample rate of the audio in Hz. Some transforms (e.g. PCEN) use
        this to set internal parameters.

    Returns
    -------
    torch.nn.Module
        The constructed transform module.
    """
    return spectrogram_transforms.build(config, samplerate)
