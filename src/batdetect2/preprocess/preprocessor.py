from typing import Optional

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
from batdetect2.typing import PreprocessorProtocol

__all__ = [
    "Preprocessor",
    "build_preprocessor",
]


class Preprocessor(torch.nn.Module, PreprocessorProtocol):
    """Standard implementation of the `Preprocessor` protocol."""

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
            *[
                build_audio_transform(step, samplerate=input_samplerate)
                for step in config.audio_transforms
            ]
        )

        self.spectrogram_transforms = torch.nn.Sequential(
            *[
                build_spectrogram_transform(step, samplerate=input_samplerate)
                for step in config.spectrogram_transforms
            ]
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
        wav = self.audio_transforms(wav)
        spec = self.spectrogram_builder(wav)
        return self.process_spectrogram(spec)

    def generate_spectrogram(self, wav: torch.Tensor) -> torch.Tensor:
        return self.spectrogram_builder(wav)

    def process_audio(self, wav: torch.Tensor) -> torch.Tensor:
        return self(wav)

    def process_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        spec = self.spectrogram_crop(spec)
        spec = self.spectrogram_transforms(spec)
        return self.spectrogram_resizer(spec)


def compute_output_samplerate(
    config: PreprocessingConfig,
    input_samplerate: int = TARGET_SAMPLERATE_HZ,
) -> float:
    _, hop_size = _spec_params_from_config(
        config.stft, samplerate=input_samplerate
    )
    factor = config.size.resize_factor
    return input_samplerate * factor / hop_size


def build_preprocessor(
    config: PreprocessingConfig | None = None,
    input_samplerate: int = TARGET_SAMPLERATE_HZ,
) -> PreprocessorProtocol:
    """Factory function to build the standard preprocessor from configuration."""
    config = config or PreprocessingConfig()
    logger.opt(lazy=True).debug(
        "Building preprocessor with config: \n{}",
        lambda: config.to_yaml_string(),
    )
    return Preprocessor(config=config, input_samplerate=input_samplerate)
