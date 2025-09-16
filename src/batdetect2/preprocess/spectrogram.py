"""Computes spectrograms from audio waveforms with configurable parameters."""

from typing import Callable, Optional

import numpy as np
import torch
import torchaudio

from batdetect2.preprocess.common import PeakNormalize
from batdetect2.preprocess.config import (
    ScaleAmplitudeConfig,
    SpectrogramConfig,
    SpectrogramTransform,
    STFTConfig,
)

__all__ = [
    "build_spectrogram_builder",
    "build_spectrogram_pipeline",
]


def build_spectrogram_builder(
    samplerate: int,
    conf: STFTConfig,
) -> torch.nn.Module:
    n_fft, hop_length = _spec_params_from_config(samplerate, conf)
    return torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        window_fn=get_spectrogram_window(conf.window_fn),
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


def _spec_params_from_config(samplerate: int, conf: STFTConfig):
    n_fft = int(samplerate * conf.window_duration)
    hop_length = int(n_fft * (1 - conf.window_overlap))
    return n_fft, hop_length


def _frequency_to_index(
    freq: float,
    samplerate: int,
    n_fft: int,
) -> Optional[int]:
    alpha = freq * 2 / samplerate
    height = np.floor(n_fft / 2) + 1
    index = int(np.floor(alpha * height))

    if index <= 0:
        return None

    if index >= height:
        return None

    return index


class FrequencyClip(torch.nn.Module):
    def __init__(
        self,
        low_index: Optional[int] = None,
        high_index: Optional[int] = None,
    ):
        super().__init__()
        self.low_index = low_index
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


class PCEN(torch.nn.Module):
    def __init__(
        self,
        smoothing_constant: float,
        gain: float = 0.98,
        bias: float = 2.0,
        power: float = 0.5,
        eps: float = 1e-6,
        dtype=torch.float64,
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


class ToPower(torch.nn.Module):
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return spec**2


def _build_amplitude_scaler(conf: ScaleAmplitudeConfig) -> torch.nn.Module:
    if conf.scale == "db":
        return torchaudio.transforms.AmplitudeToDB()

    if conf.scale == "power":
        return ToPower()

    raise NotImplementedError(
        f"Amplitude scaling {conf.scale} not implemented"
    )


class SpectralMeanSubstraction(torch.nn.Module):
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        mean = spec.mean(-1, keepdim=True)
        return (spec - mean).clamp(min=0)


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


def _build_spectrogram_transform_step(
    step: SpectrogramTransform,
    samplerate: int,
) -> torch.nn.Module:
    if step.name == "pcen":
        return PCEN(
            smoothing_constant=_compute_smoothing_constant(
                samplerate=samplerate,
                time_constant=step.time_constant,
            ),
            gain=step.gain,
            bias=step.bias,
            power=step.power,
        )

    if step.name == "scale_amplitude":
        return _build_amplitude_scaler(step)

    if step.name == "spectral_mean_substraction":
        return SpectralMeanSubstraction()

    if step.name == "peak_normalize":
        return PeakNormalize()

    raise NotImplementedError(
        f"Spectrogram preprocessing step {step.name} not implemented"
    )


def build_spectrogram_transform(
    samplerate: int,
    conf: SpectrogramConfig,
) -> torch.nn.Module:
    return torch.nn.Sequential(
        *[
            _build_spectrogram_transform_step(step, samplerate=samplerate)
            for step in conf.transforms
        ]
    )


class SpectrogramPipeline(torch.nn.Module):
    def __init__(
        self,
        spec_builder: torch.nn.Module,
        freq_cutter: torch.nn.Module,
        transforms: torch.nn.Module,
        resizer: torch.nn.Module,
    ):
        super().__init__()
        self.spec_builder = spec_builder
        self.freq_cutter = freq_cutter
        self.transforms = transforms
        self.resizer = resizer

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spec = self.spec_builder(wav)
        spec = self.freq_cutter(spec)
        spec = self.transforms(spec)
        return self.resizer(spec)

    def compute_spectrogram(self, wav: torch.Tensor) -> torch.Tensor:
        return self.spec_builder(wav)

    def select_frequencies(self, spec: torch.Tensor) -> torch.Tensor:
        return self.freq_cutter(spec)

    def transform_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        return self.transforms(spec)

    def resize_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        return self.resizer(spec)


def build_spectrogram_pipeline(
    samplerate: int,
    conf: SpectrogramConfig,
) -> SpectrogramPipeline:
    spec_builder = build_spectrogram_builder(samplerate, conf.stft)
    n_fft, _ = _spec_params_from_config(samplerate, conf.stft)
    cutter = FrequencyClip(
        low_index=_frequency_to_index(
            conf.frequencies.min_freq, samplerate, n_fft
        ),
        high_index=_frequency_to_index(
            conf.frequencies.max_freq, samplerate, n_fft
        ),
    )
    transforms = build_spectrogram_transform(samplerate, conf)
    resizer = ResizeSpec(
        height=conf.size.height,
        time_factor=conf.size.resize_factor,
    )
    return SpectrogramPipeline(
        spec_builder=spec_builder,
        freq_cutter=cutter,
        transforms=transforms,
        resizer=resizer,
    )
