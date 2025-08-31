from typing import List, Optional

import numpy as np
from loguru import logger
from soundevent import data
from soundevent.geometry import compute_bounds, intervals_overlap

from batdetect2.configs import BaseConfig
from batdetect2.typing import ClipperProtocol

DEFAULT_TRAIN_CLIP_DURATION = 0.256
DEFAULT_MAX_EMPTY_CLIP = 0.1


class ClipingConfig(BaseConfig):
    duration: float = DEFAULT_TRAIN_CLIP_DURATION
    random: bool = True
    max_empty: float = DEFAULT_MAX_EMPTY_CLIP
    min_sound_event_overlap: float = 0


class Clipper:
    def __init__(
        self,
        duration: float = 0.5,
        max_empty: float = 0.2,
        random: bool = True,
        min_sound_event_overlap: float = 0,
    ):
        super().__init__()
        self.duration = duration
        self.random = random
        self.max_empty = max_empty
        self.min_sound_event_overlap = min_sound_event_overlap

    def __call__(
        self,
        clip_annotation: data.ClipAnnotation,
    ) -> data.ClipAnnotation:
        return get_subclip_annotation(
            clip_annotation,
            random=self.random,
            duration=self.duration,
            max_empty=self.max_empty,
            min_sound_event_overlap=self.min_sound_event_overlap,
        )


def get_subclip_annotation(
    clip_annotation: data.ClipAnnotation,
    random: bool = True,
    duration: float = 0.5,
    max_empty: float = 0.2,
    min_sound_event_overlap: float = 0,
) -> data.ClipAnnotation:
    clip = clip_annotation.clip

    subclip = select_subclip(
        clip,
        random=random,
        duration=duration,
        max_empty=max_empty,
    )

    sound_events = select_sound_event_annotations(
        clip_annotation,
        subclip,
        min_overlap=min_sound_event_overlap,
    )

    return clip_annotation.model_copy(
        update=dict(
            clip=subclip,
            sound_events=sound_events,
        )
    )


def select_subclip(
    clip: data.Clip,
    random: bool = True,
    duration: float = 0.5,
    max_empty: float = 0.2,
) -> data.Clip:
    start_time = clip.start_time
    end_time = clip.end_time

    if duration > clip.duration + max_empty or not random:
        return clip.model_copy(
            update=dict(
                start_time=start_time,
                end_time=start_time + duration,
            )
        )

    random_start_time = np.random.uniform(
        low=start_time,
        high=end_time + max_empty - duration,
    )

    return clip.model_copy(
        update=dict(
            start_time=random_start_time,
            end_time=random_start_time + duration,
        )
    )


def select_sound_event_annotations(
    clip_annotation: data.ClipAnnotation,
    subclip: data.Clip,
    min_overlap: float = 0,
) -> List[data.SoundEventAnnotation]:
    selected = []

    start_time = subclip.start_time
    end_time = subclip.end_time

    for sound_event_annotation in clip_annotation.sound_events:
        geometry = sound_event_annotation.sound_event.geometry

        if geometry is None:
            continue

        geom_start_time, _, geom_end_time, _ = compute_bounds(geometry)

        if not intervals_overlap(
            (start_time, end_time),
            (geom_start_time, geom_end_time),
            min_absolute_overlap=min_overlap,
        ):
            continue

        selected.append(sound_event_annotation)

    return selected


def build_clipper(
    config: Optional[ClipingConfig] = None,
    random: Optional[bool] = None,
) -> ClipperProtocol:
    config = config or ClipingConfig()
    logger.opt(lazy=True).debug(
        "Building clipper with config: \n{}",
        lambda: config.to_yaml_string(),
    )
    return Clipper(
        duration=config.duration,
        max_empty=config.max_empty,
        random=config.random if random else False,
    )
