from typing import Optional

import torch
from loguru import logger

from batdetect2.preprocess.audio import build_audio_pipeline
from batdetect2.preprocess.config import PreprocessingConfig
from batdetect2.preprocess.spectrogram import (
    _spec_params_from_config,
    build_spectrogram_pipeline,
)
from batdetect2.typing import PreprocessorProtocol, SpectrogramPipeline

__all__ = [
    "StandardPreprocessor",
    "build_preprocessor",
]


class StandardPreprocessor(torch.nn.Module, PreprocessorProtocol):
    """Standard implementation of the `Preprocessor` protocol."""

    input_samplerate: int
    output_samplerate: float

    max_freq: float
    min_freq: float

    def __init__(
        self,
        audio_pipeline: torch.nn.Module,
        spectrogram_pipeline: SpectrogramPipeline,
        input_samplerate: int,
        output_samplerate: float,
        max_freq: float,
        min_freq: float,
    ) -> None:
        super().__init__()
        self.audio_pipeline = audio_pipeline
        self.spectrogram_pipeline = spectrogram_pipeline

        self.max_freq = max_freq
        self.min_freq = min_freq

        self.input_samplerate = input_samplerate
        self.output_samplerate = output_samplerate

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        wav = self.audio_pipeline(wav)
        return self.spectrogram_pipeline(wav)


def compute_output_samplerate(config: PreprocessingConfig) -> float:
    samplerate = config.audio.samplerate
    _, hop_size = _spec_params_from_config(samplerate, config.spectrogram.stft)
    factor = config.spectrogram.size.resize_factor
    return samplerate * factor / hop_size


def build_preprocessor(
    config: Optional[PreprocessingConfig] = None,
) -> PreprocessorProtocol:
    """Factory function to build the standard preprocessor from configuration."""
    config = config or PreprocessingConfig()
    logger.opt(lazy=True).debug(
        "Building preprocessor with config: \n{}",
        lambda: config.to_yaml_string(),
    )

    samplerate = config.audio.samplerate

    min_freq = config.spectrogram.frequencies.min_freq
    max_freq = config.spectrogram.frequencies.max_freq

    output_samplerate = compute_output_samplerate(config)

    return StandardPreprocessor(
        audio_pipeline=build_audio_pipeline(config.audio),
        spectrogram_pipeline=build_spectrogram_pipeline(
            samplerate, config.spectrogram
        ),
        input_samplerate=samplerate,
        output_samplerate=output_samplerate,
        min_freq=min_freq,
        max_freq=max_freq,
    )
