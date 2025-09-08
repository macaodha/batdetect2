from collections.abc import Callable
from typing import Annotated, List, Literal, Sequence, Union

from pydantic import Field
from soundevent import data
from soundevent.geometry import compute_bounds

from batdetect2.configs import BaseConfig
from batdetect2.data._core import Registry

SoundEventCondition = Callable[[data.SoundEventAnnotation], bool]

_conditions: Registry[SoundEventCondition] = Registry("condition")


class HasTagConfig(BaseConfig):
    name: Literal["has_tag"] = "has_tag"
    tag: data.Tag


@_conditions.register(HasTagConfig)
class HasTag:
    def __init__(self, tag: data.Tag):
        self.tag = tag

    def __call__(
        self, sound_event_annotation: data.SoundEventAnnotation
    ) -> bool:
        return self.tag in sound_event_annotation.tags

    @classmethod
    def from_config(cls, config: HasTagConfig):
        return cls(tag=config.tag)


class HasAllTagsConfig(BaseConfig):
    name: Literal["has_all_tags"] = "has_all_tags"
    tags: List[data.Tag]


@_conditions.register(HasAllTagsConfig)
class HasAllTags:
    def __init__(self, tags: List[data.Tag]):
        if not tags:
            raise ValueError("Need to specify at least one tag")

        self.tags = set(tags)

    def __call__(
        self, sound_event_annotation: data.SoundEventAnnotation
    ) -> bool:
        return self.tags.issubset(sound_event_annotation.tags)

    @classmethod
    def from_config(cls, config: HasAllTagsConfig):
        return cls(tags=config.tags)


class HasAnyTagConfig(BaseConfig):
    name: Literal["has_any_tag"] = "has_any_tag"
    tags: List[data.Tag]


@_conditions.register(HasAnyTagConfig)
class HasAnyTag:
    def __init__(self, tags: List[data.Tag]):
        if not tags:
            raise ValueError("Need to specify at least one tag")

        self.tags = set(tags)

    def __call__(
        self, sound_event_annotation: data.SoundEventAnnotation
    ) -> bool:
        return bool(self.tags.intersection(sound_event_annotation.tags))

    @classmethod
    def from_config(cls, config: HasAnyTagConfig):
        return cls(tags=config.tags)


Operator = Literal["gt", "gte", "lt", "lte", "eq"]


class DurationConfig(BaseConfig):
    name: Literal["duration"] = "duration"
    operator: Operator
    seconds: float


def _build_comparator(
    operator: Operator, value: float
) -> Callable[[float], bool]:
    if operator == "gt":
        return lambda x: x > value

    if operator == "gte":
        return lambda x: x >= value

    if operator == "lt":
        return lambda x: x < value

    if operator == "lte":
        return lambda x: x <= value

    if operator == "eq":
        return lambda x: x == value

    raise ValueError(f"Invalid operator {operator}")


@_conditions.register(DurationConfig)
class Duration:
    def __init__(self, operator: Operator, seconds: float):
        self.operator = operator
        self.seconds = seconds
        self._comparator = _build_comparator(self.operator, self.seconds)

    def __call__(
        self,
        sound_event_annotation: data.SoundEventAnnotation,
    ) -> bool:
        geometry = sound_event_annotation.sound_event.geometry

        if geometry is None:
            return False

        start_time, _, end_time, _ = compute_bounds(geometry)
        duration = end_time - start_time

        return self._comparator(duration)

    @classmethod
    def from_config(cls, config: DurationConfig):
        return cls(operator=config.operator, seconds=config.seconds)


class FrequencyConfig(BaseConfig):
    name: Literal["frequency"] = "frequency"
    boundary: Literal["low", "high"]
    operator: Operator
    hertz: float


@_conditions.register(FrequencyConfig)
class Frequency:
    def __init__(
        self,
        operator: Operator,
        boundary: Literal["low", "high"],
        hertz: float,
    ):
        self.operator = operator
        self.hertz = hertz
        self.boundary = boundary
        self._comparator = _build_comparator(self.operator, self.hertz)

    def __call__(
        self,
        sound_event_annotation: data.SoundEventAnnotation,
    ) -> bool:
        geometry = sound_event_annotation.sound_event.geometry

        if geometry is None:
            return False

        # Automatically false if geometry does not have a frequency range
        if isinstance(geometry, (data.TimeInterval, data.TimeStamp)):
            return False

        _, low_freq, _, high_freq = compute_bounds(geometry)

        if self.boundary == "low":
            return self._comparator(low_freq)

        return self._comparator(high_freq)

    @classmethod
    def from_config(cls, config: FrequencyConfig):
        return cls(
            operator=config.operator,
            boundary=config.boundary,
            hertz=config.hertz,
        )


class AllOfConfig(BaseConfig):
    name: Literal["all_of"] = "all_of"
    conditions: Sequence["SoundEventConditionConfig"]


@_conditions.register(AllOfConfig)
class AllOf:
    def __init__(self, conditions: List[SoundEventCondition]):
        self.conditions = conditions

    def __call__(
        self, sound_event_annotation: data.SoundEventAnnotation
    ) -> bool:
        return all(c(sound_event_annotation) for c in self.conditions)

    @classmethod
    def from_config(cls, config: AllOfConfig):
        conditions = [
            build_sound_event_condition(cond) for cond in config.conditions
        ]
        return cls(conditions)


class AnyOfConfig(BaseConfig):
    name: Literal["any_of"] = "any_of"
    conditions: List["SoundEventConditionConfig"]


@_conditions.register(AnyOfConfig)
class AnyOf:
    def __init__(self, conditions: List[SoundEventCondition]):
        self.conditions = conditions

    def __call__(
        self, sound_event_annotation: data.SoundEventAnnotation
    ) -> bool:
        return any(c(sound_event_annotation) for c in self.conditions)

    @classmethod
    def from_config(cls, config: AnyOfConfig):
        conditions = [
            build_sound_event_condition(cond) for cond in config.conditions
        ]
        return cls(conditions)


class NotConfig(BaseConfig):
    name: Literal["not"] = "not"
    condition: "SoundEventConditionConfig"


@_conditions.register(NotConfig)
class Not:
    def __init__(self, condition: SoundEventCondition):
        self.condition = condition

    def __call__(
        self, sound_event_annotation: data.SoundEventAnnotation
    ) -> bool:
        return not self.condition(sound_event_annotation)

    @classmethod
    def from_config(cls, config: NotConfig):
        condition = build_sound_event_condition(config.condition)
        return cls(condition)


SoundEventConditionConfig = Annotated[
    Union[
        HasTagConfig,
        HasAllTagsConfig,
        HasAnyTagConfig,
        DurationConfig,
        FrequencyConfig,
        AllOfConfig,
        AnyOfConfig,
        NotConfig,
    ],
    Field(discriminator="name"),
]


def build_sound_event_condition(
    config: SoundEventConditionConfig,
) -> SoundEventCondition:
    return _conditions.build(config)


def filter_clip_annotation(
    clip_annotation: data.ClipAnnotation,
    condition: SoundEventCondition,
) -> data.ClipAnnotation:
    return clip_annotation.model_copy(
        update=dict(
            sound_events=[
                sound_event
                for sound_event in clip_annotation.sound_events
                if condition(sound_event)
            ]
        )
    )
