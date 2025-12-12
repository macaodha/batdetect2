from collections.abc import Callable
from typing import Annotated, Dict, List, Literal

from pydantic import Field
from soundevent import data

from batdetect2.core.configs import BaseConfig
from batdetect2.core.registries import Registry
from batdetect2.data.conditions import (
    SoundEventCondition,
    SoundEventConditionConfig,
    build_sound_event_condition,
)

SoundEventTransform = Callable[
    [data.SoundEventAnnotation],
    data.SoundEventAnnotation,
]

transforms: Registry[SoundEventTransform, []] = Registry("transform")


class SetFrequencyBoundConfig(BaseConfig):
    name: Literal["set_frequency"] = "set_frequency"
    boundary: Literal["low", "high"] = "low"
    hertz: float


class SetFrequencyBound:
    def __init__(self, hertz: float, boundary: Literal["low", "high"] = "low"):
        self.hertz = hertz
        self.boundary = boundary

    def __call__(
        self,
        sound_event_annotation: data.SoundEventAnnotation,
    ) -> data.SoundEventAnnotation:
        sound_event = sound_event_annotation.sound_event
        geometry = sound_event.geometry

        if geometry is None:
            return sound_event_annotation

        if not isinstance(geometry, data.BoundingBox):
            return sound_event_annotation

        start_time, low_freq, end_time, high_freq = geometry.coordinates

        if self.boundary == "low":
            low_freq = self.hertz
            high_freq = max(high_freq, low_freq)

        elif self.boundary == "high":
            high_freq = self.hertz
            low_freq = min(high_freq, low_freq)

        geometry = data.BoundingBox(
            coordinates=[start_time, low_freq, end_time, high_freq],
        )

        sound_event = sound_event.model_copy(update=dict(geometry=geometry))
        return sound_event_annotation.model_copy(
            update=dict(sound_event=sound_event)
        )

    @transforms.register(SetFrequencyBoundConfig)
    @staticmethod
    def from_config(config: SetFrequencyBoundConfig):
        return SetFrequencyBound(hertz=config.hertz, boundary=config.boundary)


class ApplyIfConfig(BaseConfig):
    name: Literal["apply_if"] = "apply_if"
    transform: "SoundEventTransformConfig"
    condition: SoundEventConditionConfig


class ApplyIf:
    def __init__(
        self,
        condition: SoundEventCondition,
        transform: SoundEventTransform,
    ):
        self.condition = condition
        self.transform = transform

    def __call__(
        self,
        sound_event_annotation: data.SoundEventAnnotation,
    ) -> data.SoundEventAnnotation:
        if not self.condition(sound_event_annotation):
            return sound_event_annotation

        return self.transform(sound_event_annotation)

    @transforms.register(ApplyIfConfig)
    @staticmethod
    def from_config(config: ApplyIfConfig):
        transform = build_sound_event_transform(config.transform)
        condition = build_sound_event_condition(config.condition)
        return ApplyIf(condition=condition, transform=transform)


class ReplaceTagConfig(BaseConfig):
    name: Literal["replace_tag"] = "replace_tag"
    original: data.Tag
    replacement: data.Tag


class ReplaceTag:
    def __init__(
        self,
        original: data.Tag,
        replacement: data.Tag,
    ):
        self.original = original
        self.replacement = replacement

    def __call__(
        self,
        sound_event_annotation: data.SoundEventAnnotation,
    ) -> data.SoundEventAnnotation:
        tags = []

        for tag in sound_event_annotation.tags:
            if tag == self.original:
                tags.append(self.replacement)
            else:
                tags.append(tag)

        return sound_event_annotation.model_copy(update=dict(tags=tags))

    @transforms.register(ReplaceTagConfig)
    @staticmethod
    def from_config(config: ReplaceTagConfig):
        return ReplaceTag(
            original=config.original, replacement=config.replacement
        )


class MapTagValueConfig(BaseConfig):
    name: Literal["map_tag_value"] = "map_tag_value"
    tag_key: str
    value_mapping: Dict[str, str]
    target_key: str | None = None


class MapTagValue:
    def __init__(
        self,
        tag_key: str,
        value_mapping: Dict[str, str],
        target_key: str | None = None,
    ):
        self.tag_key = tag_key
        self.value_mapping = value_mapping
        self.target_key = target_key

    def __call__(
        self,
        sound_event_annotation: data.SoundEventAnnotation,
    ) -> data.SoundEventAnnotation:
        tags = []

        for tag in sound_event_annotation.tags:
            if tag.key != self.tag_key:
                tags.append(tag)
                continue

            value = self.value_mapping.get(tag.value)

            if value is None:
                tags.append(tag)
                continue

            if self.target_key is None:
                tags.append(tag.model_copy(update=dict(value=value)))
            else:
                tags.append(
                    data.Tag(
                        key=self.target_key,  # type: ignore
                        value=value,
                    )
                )

        return sound_event_annotation.model_copy(update=dict(tags=tags))

    @transforms.register(MapTagValueConfig)
    @staticmethod
    def from_config(config: MapTagValueConfig):
        return MapTagValue(
            tag_key=config.tag_key,
            value_mapping=config.value_mapping,
            target_key=config.target_key,
        )


class ApplyAllConfig(BaseConfig):
    name: Literal["apply_all"] = "apply_all"
    steps: List["SoundEventTransformConfig"] = Field(default_factory=list)


class ApplyAll:
    def __init__(self, steps: List[SoundEventTransform]):
        self.steps = steps

    def __call__(
        self,
        sound_event_annotation: data.SoundEventAnnotation,
    ) -> data.SoundEventAnnotation:
        for step in self.steps:
            sound_event_annotation = step(sound_event_annotation)

        return sound_event_annotation

    @transforms.register(ApplyAllConfig)
    @staticmethod
    def from_config(config: ApplyAllConfig):
        steps = [build_sound_event_transform(step) for step in config.steps]
        return ApplyAll(steps)


SoundEventTransformConfig = Annotated[
    SetFrequencyBoundConfig
    | ReplaceTagConfig
    | MapTagValueConfig
    | ApplyIfConfig
    | ApplyAllConfig,
    Field(discriminator="name"),
]


def build_sound_event_transform(
    config: SoundEventTransformConfig,
) -> SoundEventTransform:
    return transforms.build(config)


def transform_clip_annotation(
    clip_annotation: data.ClipAnnotation,
    transform: SoundEventTransform,
) -> data.ClipAnnotation:
    return clip_annotation.model_copy(
        update=dict(
            sound_events=[
                transform(sound_event)
                for sound_event in clip_annotation.sound_events
            ]
        )
    )
