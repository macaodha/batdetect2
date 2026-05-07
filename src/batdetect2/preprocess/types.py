from typing import Protocol

import numpy as np
import torch

__all__ = [
    "PreprocessorProtocol",
    "SpectrogramBuilder",
]


class SpectrogramBuilder(Protocol):
    def __call__(self, wav: torch.Tensor) -> torch.Tensor: ...


class PreprocessorProtocol(Protocol):
    max_freq: float
    min_freq: float
    input_samplerate: int
    output_samplerate: float

    def __call__(self, wav: torch.Tensor) -> torch.Tensor: ...

    def generate_spectrogram(self, wav: torch.Tensor) -> torch.Tensor: ...

    def process_audio(self, wav: torch.Tensor) -> torch.Tensor: ...

    def process_spectrogram(self, spec: torch.Tensor) -> torch.Tensor: ...

    def process_numpy(self, wav: np.ndarray) -> np.ndarray:
        return self(torch.tensor(wav)).numpy()
