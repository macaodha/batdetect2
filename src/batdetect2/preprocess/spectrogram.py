"""Computes spectrograms from audio waveforms with configurable parameters."""

from typing import Annotated, Callable, Literal, Optional, Union

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
        Duration of the STFT window in seconds (e.g., 0.002 for 2ms). Must be
        > 0. Determines frequency resolution (longer window = finer frequency
        resolution).
    window_overlap : float, default=0.75
        Fraction of overlap between consecutive STFT windows (e.g., 0.75
        for 75%). Must be >= 0 and < 1. Determines time resolution
        (higher overlap = finer time resolution).
    window_fn : str, default="hann"
        Name of the window function to apply before FFT calculation. Common
        options include "hann", "hamming", "blackman". See
        `scipy.signal.get_window`.
    """

    window_duration: float = Field(default=0.002, gt=0)
    window_overlap: float = Field(default=0.75, ge=0, lt=1)
    window_fn: str = "hann"


def build_spectrogram_builder(
    config: STFTConfig,
    samplerate: int = TARGET_SAMPLERATE_HZ,
) -> torch.nn.Module:
    n_fft, hop_length = _spec_params_from_config(config, samplerate=samplerate)
    return torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        window_fn=get_spectrogram_window(config.window_fn),
        center=True,
        power=1,
    )


def get_spectrogram_window(name: str) -> Callable[..., torch.Tensor]:
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
):
    n_fft = int(samplerate * config.window_duration)
    hop_length = int(n_fft * (1 - config.window_overlap))
    return n_fft, hop_length


def _frequency_to_index(
    freq: float,
    n_fft: int,
    samplerate: int = TARGET_SAMPLERATE_HZ,
) -> Optional[int]:
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
        Maximum frequency in Hz to retain in the spectrogram after STFT.
        Frequencies above this value will be cropped. Must be > 0.
    min_freq : int, default=10000
        Minimum frequency in Hz to retain in the spectrogram after STFT.
        Frequencies below this value will be cropped. Must be >= 0.
    """

    max_freq: int = Field(default=MAX_FREQ, ge=0)
    min_freq: int = Field(default=MIN_FREQ, ge=0)


class FrequencyCrop(torch.nn.Module):
    def __init__(
        self,
        samplerate: int,
        n_fft: int,
        min_freq: Optional[int] = None,
        max_freq: Optional[int] = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.samplerate = samplerate
        self.min_freq = min_freq
        self.max_freq = max_freq

        low_index = None
        if min_freq is not None:
            low_index = _frequency_to_index(
                min_freq, self.samplerate, self.n_fft
            )
        self.low_index = low_index

        high_index = None
        if max_freq is not None:
            high_index = _frequency_to_index(
                max_freq, self.samplerate, self.n_fft
            )
        self.high_index = high_index

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
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
    stft: Optional[STFTConfig] = None,
    samplerate: int = TARGET_SAMPLERATE_HZ,
) -> torch.nn.Module:
    stft = stft or STFTConfig()
    n_fft, _ = _spec_params_from_config(stft, samplerate=samplerate)
    return FrequencyCrop(
        samplerate=samplerate,
        n_fft=n_fft,
        min_freq=config.min_freq,
        max_freq=config.max_freq,
    )


class ResizeConfig(BaseConfig):
    name: Literal["resize_spec"] = "resize_spec"
    height: int = 128
    resize_factor: float = 0.5


class ResizeSpec(torch.nn.Module):
    def __init__(self, height: int, time_factor: float):
        super().__init__()
        self.height = height
        self.time_factor = time_factor

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
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
    return ResizeSpec(height=config.height, time_factor=config.resize_factor)


spectrogram_transforms: Registry[torch.nn.Module, [int]] = Registry(
    "spectrogram_transform"
)


class PcenConfig(BaseConfig):
    """Configuration for Per-Channel Energy Normalization (PCEN)."""

    name: Literal["pcen"] = "pcen"
    time_constant: float = 0.4
    gain: float = 0.98
    bias: float = 2
    power: float = 0.5


class PCEN(torch.nn.Module):
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

    @classmethod
    def from_config(cls, config: PcenConfig, samplerate: int):
        smooth = _compute_smoothing_constant(samplerate, config.time_constant)
        return cls(
            smoothing_constant=smooth,
            gain=config.gain,
            bias=config.bias,
            power=config.power,
        )


spectrogram_transforms.register(PcenConfig, PCEN)


def _compute_smoothing_constant(
    samplerate: int,
    time_constant: float,
) -> float:
    # NOTE: These were taken to match the original implementation
    hop_length = 512
    sr = samplerate / 10
    time_constant = time_constant
    t_frames = time_constant * sr / float(hop_length)
    return (np.sqrt(1 + 4 * t_frames**2) - 1) / (2 * t_frames**2)


class ScaleAmplitudeConfig(BaseConfig):
    name: Literal["scale_amplitude"] = "scale_amplitude"
    scale: Literal["power", "db"] = "db"


class ToPower(torch.nn.Module):
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return spec**2


_scalers = {
    "db": torchaudio.transforms.AmplitudeToDB,
    "power": ToPower,
}


class ScaleAmplitude(torch.nn.Module):
    def __init__(self, scale: Literal["power", "db"]):
        self.scale = scale
        self.scaler = _scalers[scale]()

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return self.scaler(spec)

    @classmethod
    def from_config(cls, config: ScaleAmplitudeConfig, samplerate: int):
        return cls(scale=config.scale)


spectrogram_transforms.register(ScaleAmplitudeConfig, ScaleAmplitude)


class SpectralMeanSubstractionConfig(BaseConfig):
    name: Literal["spectral_mean_substraction"] = "spectral_mean_substraction"


class SpectralMeanSubstraction(torch.nn.Module):
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        mean = spec.mean(-1, keepdim=True)
        return (spec - mean).clamp(min=0)

    @classmethod
    def from_config(
        cls,
        config: SpectralMeanSubstractionConfig,
        samplerate: int,
    ):
        return cls()


spectrogram_transforms.register(
    SpectralMeanSubstractionConfig,
    SpectralMeanSubstraction,
)


class PeakNormalizeConfig(BaseConfig):
    name: Literal["peak_normalize"] = "peak_normalize"


class PeakNormalize(torch.nn.Module):
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return peak_normalize(spec)

    @classmethod
    def from_config(cls, config: PeakNormalizeConfig, samplerate: int):
        return cls()


spectrogram_transforms.register(PeakNormalizeConfig, PeakNormalize)

SpectrogramTransform = Annotated[
    Union[
        PcenConfig,
        ScaleAmplitudeConfig,
        SpectralMeanSubstractionConfig,
        PeakNormalizeConfig,
    ],
    Field(discriminator="name"),
]


def build_spectrogram_transform(
    config: SpectrogramTransform,
    samplerate: int,
) -> torch.nn.Module:
    return spectrogram_transforms.build(config, samplerate)
