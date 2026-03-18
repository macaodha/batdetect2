"""Main entry point for the BatDetect2 preprocessing subsystem."""

from batdetect2.audio import TARGET_SAMPLERATE_HZ
from batdetect2.preprocess.config import (
    PreprocessingConfig,
    load_preprocessing_config,
)
from batdetect2.preprocess.preprocessor import Preprocessor, build_preprocessor
from batdetect2.preprocess.spectrogram import MAX_FREQ, MIN_FREQ
from batdetect2.preprocess.types import PreprocessorProtocol

__all__ = [
    "PreprocessorProtocol",
    "MAX_FREQ",
    "MIN_FREQ",
    "PreprocessingConfig",
    "Preprocessor",
    "TARGET_SAMPLERATE_HZ",
    "build_preprocessor",
    "load_preprocessing_config",
]
