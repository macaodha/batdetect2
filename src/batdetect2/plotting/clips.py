from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from soundevent import data

from batdetect2.audio import build_audio_loader
from batdetect2.plotting.common import plot_spectrogram
from batdetect2.preprocess import build_preprocessor
from batdetect2.typing import AudioLoader, PreprocessorProtocol

__all__ = [
    "plot_clip",
]


def plot_clip(
    clip: data.Clip,
    audio_loader: Optional[AudioLoader] = None,
    preprocessor: Optional[PreprocessorProtocol] = None,
    figsize: Optional[Tuple[int, int]] = None,
    ax: Optional[Axes] = None,
    audio_dir: Optional[data.PathLike] = None,
    spec_cmap: str = "gray",
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if preprocessor is None:
        preprocessor = build_preprocessor()

    if audio_loader is None:
        audio_loader = build_audio_loader()

    wav = torch.tensor(audio_loader.load_clip(clip, audio_dir=audio_dir))
    spec = preprocessor(wav)

    plot_spectrogram(
        spec,
        start_time=clip.start_time,
        end_time=clip.end_time,
        min_freq=preprocessor.min_freq,
        max_freq=preprocessor.max_freq,
        ax=ax,
        cmap=spec_cmap,
    )

    return ax
