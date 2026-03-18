from typing import Annotated, List, Literal

import numpy as np
from loguru import logger
from pydantic import Field
from soundevent import data
from soundevent.geometry import compute_bounds, intervals_overlap

from batdetect2.audio.types import ClipperProtocol
from batdetect2.core import (
    BaseConfig,
    ImportConfig,
    Registry,
    add_import_config,
)

DEFAULT_TRAIN_CLIP_DURATION = 0.256
DEFAULT_MAX_EMPTY_CLIP = 0.1


__all__ = [
    "build_clipper",
    "ClipConfig",
    "ClipperImportConfig",
]


clipper_registry: Registry[ClipperProtocol, []] = Registry("clipper")


@add_import_config(clipper_registry)
class ClipperImportConfig(ImportConfig):
    """Use any callable as a clipper.

    Set ``name="import"`` and provide a ``target`` pointing to any
    callable to use it instead of a built-in option.
    """

    name: Literal["import"] = "import"


class RandomClipConfig(BaseConfig):
    name: Literal["random_subclip"] = "random_subclip"
    duration: float = DEFAULT_TRAIN_CLIP_DURATION
    random: bool = True
    max_empty: float = DEFAULT_MAX_EMPTY_CLIP
    min_sound_event_overlap: float = 0


class RandomClip:
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
        subclip = self.get_subclip(clip_annotation.clip)
        sound_events = select_sound_event_annotations(
            clip_annotation,
            subclip,
            min_overlap=self.min_sound_event_overlap,
        )
        return clip_annotation.model_copy(
            update=dict(
                clip=subclip,
                sound_events=sound_events,
            )
        )

    def get_subclip(self, clip: data.Clip) -> data.Clip:
        return select_random_subclip(
            clip,
            random=self.random,
            duration=self.duration,
            max_empty=self.max_empty,
        )

    @clipper_registry.register(RandomClipConfig)
    @staticmethod
    def from_config(config: RandomClipConfig):
        return RandomClip(
            duration=config.duration,
            max_empty=config.max_empty,
            min_sound_event_overlap=config.min_sound_event_overlap,
        )


def get_subclip_annotation(
    clip_annotation: data.ClipAnnotation,
    random: bool = True,
    duration: float = 0.5,
    max_empty: float = 0.2,
    min_sound_event_overlap: float = 0,
) -> data.ClipAnnotation:
    clip = clip_annotation.clip

    subclip = select_random_subclip(
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


def select_random_subclip(
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


class PaddedClipConfig(BaseConfig):
    name: Literal["whole_audio_padded"] = "whole_audio_padded"
    chunk_size: float = DEFAULT_TRAIN_CLIP_DURATION


class PaddedClip:
    def __init__(self, chunk_size: float = DEFAULT_TRAIN_CLIP_DURATION):
        self.chunk_size = chunk_size

    def __call__(
        self,
        clip_annotation: data.ClipAnnotation,
    ) -> data.ClipAnnotation:
        clip = clip_annotation.clip
        clip = self.get_subclip(clip)
        return clip_annotation.model_copy(update=dict(clip=clip))

    def get_subclip(self, clip: data.Clip) -> data.Clip:
        duration = clip.duration

        target_duration = float(
            self.chunk_size * np.ceil(duration / self.chunk_size)
        )
        clip = clip.model_copy(
            update=dict(
                end_time=clip.start_time + target_duration,
            )
        )
        return clip

    @clipper_registry.register(PaddedClipConfig)
    @staticmethod
    def from_config(config: PaddedClipConfig):
        return PaddedClip(chunk_size=config.chunk_size)


class FixedDurationClipConfig(BaseConfig):
    name: Literal["fixed_duration"] = "fixed_duration"
    duration: float = DEFAULT_TRAIN_CLIP_DURATION


class FixedDurationClip:
    def __init__(self, duration: float = DEFAULT_TRAIN_CLIP_DURATION):
        self.duration = duration

    def __call__(
        self,
        clip_annotation: data.ClipAnnotation,
    ) -> data.ClipAnnotation:
        clip = self.get_subclip(clip_annotation.clip)
        sound_events = select_sound_event_annotations(
            clip_annotation,
            clip,
            min_overlap=0,
        )
        return clip_annotation.model_copy(
            update=dict(
                clip=clip,
                sound_events=sound_events,
            )
        )

    def get_subclip(self, clip: data.Clip) -> data.Clip:
        return clip.model_copy(
            update=dict(
                end_time=clip.start_time + self.duration,
            )
        )

    @clipper_registry.register(FixedDurationClipConfig)
    @staticmethod
    def from_config(config: FixedDurationClipConfig):
        return FixedDurationClip(duration=config.duration)


ClipConfig = Annotated[
    RandomClipConfig | PaddedClipConfig | FixedDurationClipConfig,
    Field(discriminator="name"),
]


def build_clipper(config: ClipConfig | None = None) -> ClipperProtocol:
    config = config or RandomClipConfig()

    logger.opt(lazy=True).debug(
        "Building clipper with config: \n{}",
        lambda: config.to_yaml_string(),
    )
    return clipper_registry.build(config)
