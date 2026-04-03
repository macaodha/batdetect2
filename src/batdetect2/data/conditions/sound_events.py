from collections.abc import Callable, Sequence
from typing import Annotated, Literal

from pydantic import Field
from soundevent import data
from soundevent.geometry import compute_bounds

from batdetect2.core.configs import BaseConfig
from batdetect2.core.registries import (
    ImportConfig,
    Registry,
    add_import_config,
)
from batdetect2.data.conditions.common import (
    HasAllTagsConfig,
    HasAnyTagConfig,
    HasTagConfig,
    IdInListConfig,
    MultiConditionConfigBase,
    NotConditionConfigBase,
    register_all_of_condition,
    register_any_of_condition,
    register_has_all_tags_condition,
    register_has_any_tag_condition,
    register_has_tag_condition,
    register_id_in_list_condition,
    register_not_condition,
)

__all__ = [
    "AllOfConfig",
    "AnyOfConfig",
    "DurationConfig",
    "FrequencyConfig",
    "HasAllTagsConfig",
    "HasAnyTagConfig",
    "HasTagConfig",
    "NotConfig",
    "Operator",
    "SoundEventCondition",
    "SoundEventConditionConfig",
    "SoundEventConditionImportConfig",
    "build_sound_event_condition",
    "filter_clip_annotation",
]

SoundEventCondition = Callable[[data.SoundEventAnnotation], bool]

sound_event_conditions: Registry[
    SoundEventCondition,
    [data.PathLike | None],
] = Registry("sound_event_condition")


@add_import_config(sound_event_conditions, arg_names=["base_dir"])
class SoundEventConditionImportConfig(ImportConfig):
    """Use any callable as a sound event condition.

    Set ``name="import"`` and provide a ``target`` pointing to any
    callable to use it instead of a built-in option.
    """

    name: Literal["import"] = "import"


register_has_tag_condition(sound_event_conditions)(HasTagConfig)
register_has_all_tags_condition(sound_event_conditions)(HasAllTagsConfig)
register_has_any_tag_condition(sound_event_conditions)(HasAnyTagConfig)
register_id_in_list_condition(sound_event_conditions)(IdInListConfig)


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

    @sound_event_conditions.register(DurationConfig)
    @staticmethod
    def from_config(
        config: DurationConfig,
        base_dir: data.PathLike | None = None,
    ):
        _ = base_dir
        return Duration(operator=config.operator, seconds=config.seconds)


class FrequencyConfig(BaseConfig):
    name: Literal["frequency"] = "frequency"
    boundary: Literal["low", "high"]
    operator: Operator
    hertz: float


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

        if isinstance(geometry, (data.TimeInterval, data.TimeStamp)):
            return False

        _, low_freq, _, high_freq = compute_bounds(geometry)

        if self.boundary == "low":
            return self._comparator(low_freq)

        return self._comparator(high_freq)

    @sound_event_conditions.register(FrequencyConfig)
    @staticmethod
    def from_config(
        config: FrequencyConfig,
        base_dir: data.PathLike | None = None,
    ):
        _ = base_dir
        return Frequency(
            operator=config.operator,
            boundary=config.boundary,
            hertz=config.hertz,
        )


@register_all_of_condition(sound_event_conditions)
class AllOfConfig(MultiConditionConfigBase):
    name: Literal["all_of"] = "all_of"
    conditions: Sequence["SoundEventConditionConfig"]


@register_any_of_condition(sound_event_conditions)
class AnyOfConfig(MultiConditionConfigBase):
    name: Literal["any_of"] = "any_of"
    conditions: list["SoundEventConditionConfig"]


@register_not_condition(sound_event_conditions)
class NotConfig(NotConditionConfigBase):
    name: Literal["not"] = "not"
    condition: "SoundEventConditionConfig"


SoundEventConditionConfig = Annotated[
    IdInListConfig
    | HasTagConfig
    | HasAllTagsConfig
    | HasAnyTagConfig
    | DurationConfig
    | FrequencyConfig
    | AllOfConfig
    | AnyOfConfig
    | NotConfig
    | SoundEventConditionImportConfig,
    Field(discriminator="name"),
]


def build_sound_event_condition(
    config: SoundEventConditionConfig,
    base_dir: data.PathLike | None = None,
) -> SoundEventCondition:
    return sound_event_conditions.build(config, base_dir)


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
