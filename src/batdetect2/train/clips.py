from typing import Optional, Tuple

import numpy as np
from loguru import logger

from batdetect2.configs import BaseConfig
from batdetect2.typing import ClipperProtocol
from batdetect2.typing.train import PreprocessedExample
from batdetect2.utils.arrays import adjust_width

DEFAULT_TRAIN_CLIP_DURATION = 0.512
DEFAULT_MAX_EMPTY_CLIP = 0.1


class ClipingConfig(BaseConfig):
    duration: float = DEFAULT_TRAIN_CLIP_DURATION
    random: bool = True
    max_empty: float = DEFAULT_MAX_EMPTY_CLIP


class Clipper(ClipperProtocol):
    def __init__(
        self,
        samplerate: int,
        duration: float = 0.5,
        max_empty: float = 0.2,
        random: bool = True,
    ):
        self.samplerate = samplerate
        self.duration = duration
        self.random = random
        self.max_empty = max_empty

    def extract_clip(
        self, example: PreprocessedExample
    ) -> Tuple[PreprocessedExample, float, float]:
        start_time = 0
        duration = example.audio.shape[-1] / self.samplerate

        if self.random:
            start_time = np.random.uniform(
                -self.max_empty,
                duration - self.duration + self.max_empty,
            )

        return (
            select_subclip(
                example,
                start=start_time,
                duration=self.duration,
                samplerate=self.samplerate,
            ),
            start_time,
            start_time + self.duration,
        )


def build_clipper(
    samplerate: int,
    config: Optional[ClipingConfig] = None,
    random: Optional[bool] = None,
) -> ClipperProtocol:
    config = config or ClipingConfig()
    logger.opt(lazy=True).debug(
        "Building clipper with config: \n{}",
        lambda: config.to_yaml_string(),
    )
    return Clipper(
        samplerate=samplerate,
        duration=config.duration,
        max_empty=config.max_empty,
        random=config.random if random else False,
    )


def select_subclip(
    example: PreprocessedExample,
    start: float,
    duration: float,
    samplerate: float,
    fill_value: float = 0,
) -> PreprocessedExample:
    audio_width = int(np.floor(duration * samplerate))
    audio_start = int(np.floor(start * samplerate))

    audio = adjust_width(
        example.audio[audio_start : audio_start + audio_width],
        audio_width,
        value=fill_value,
    )

    audio_duration = example.audio.shape[-1] / samplerate
    spec_sr = example.spectrogram.shape[-1] / audio_duration

    spec_start = int(np.floor(start * spec_sr))
    spec_width = int(np.floor(duration * spec_sr))

    return PreprocessedExample(
        audio=audio,
        spectrogram=adjust_width(
            example.spectrogram[:, spec_start : spec_start + spec_width],
            spec_width,
        ),
        class_heatmap=adjust_width(
            example.class_heatmap[:, :, spec_start : spec_start + spec_width],
            spec_width,
        ),
        detection_heatmap=adjust_width(
            example.detection_heatmap[:, spec_start : spec_start + spec_width],
            spec_width,
        ),
        size_heatmap=adjust_width(
            example.size_heatmap[:, :, spec_start : spec_start + spec_width],
            spec_width,
        ),
    )
