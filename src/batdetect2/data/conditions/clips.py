from collections.abc import Callable, Sequence
from typing import Annotated, Literal

from pydantic import Field
from soundevent import data

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
from batdetect2.data.conditions.recordings import (
    RecordingCondition,
    RecordingConditionConfig,
    build_recording_condition,
)

__all__ = [
    "ClipAllOfConfig",
    "ClipAnnotationCondition",
    "ClipAnnotationConditionConfig",
    "ClipAnnotationConditionImportConfig",
    "ClipAnyOfConfig",
    "ClipNotConfig",
    "RecordingSatisfiesConfig",
    "build_clip_annotation_condition",
]

ClipAnnotationCondition = Callable[[data.ClipAnnotation], bool]

clip_annotation_conditions: Registry[
    ClipAnnotationCondition,
    [data.PathLike | None],
] = Registry("clip_condition")


@add_import_config(clip_annotation_conditions, arg_names=["base_dir"])
class ClipAnnotationConditionImportConfig(ImportConfig):
    """Use any callable as a clip annotation condition.

    Set ``name="import"`` and provide a ``target`` pointing to any callable
    to use it instead of a built-in option.
    """

    name: Literal["import"] = "import"


class RecordingSatisfiesConfig(BaseConfig):
    name: Literal["recording_satisfies"] = "recording_satisfies"
    condition: RecordingConditionConfig


class RecordingSatisfies:
    def __init__(self, condition: RecordingCondition):
        self.condition = condition

    def __call__(self, clip_annotation: data.ClipAnnotation) -> bool:
        recording = clip_annotation.clip.recording
        return self.condition(recording)

    @clip_annotation_conditions.register(RecordingSatisfiesConfig)
    @staticmethod
    def from_config(
        config: RecordingSatisfiesConfig,
        base_dir: data.PathLike | None = None,
    ) -> "RecordingSatisfies":
        condition = build_recording_condition(
            config.condition,
            base_dir=base_dir,
        )
        return RecordingSatisfies(condition)


register_has_tag_condition(clip_annotation_conditions)(HasTagConfig)
register_has_all_tags_condition(clip_annotation_conditions)(HasAllTagsConfig)
register_has_any_tag_condition(clip_annotation_conditions)(HasAnyTagConfig)
register_id_in_list_condition(clip_annotation_conditions)(IdInListConfig)


@register_all_of_condition(clip_annotation_conditions)
class ClipAllOfConfig(MultiConditionConfigBase):
    name: Literal["all_of"] = "all_of"
    conditions: Sequence["ClipAnnotationConditionConfig"]


@register_any_of_condition(clip_annotation_conditions)
class ClipAnyOfConfig(MultiConditionConfigBase):
    name: Literal["any_of"] = "any_of"
    conditions: Sequence["ClipAnnotationConditionConfig"]


@register_not_condition(clip_annotation_conditions)
class ClipNotConfig(NotConditionConfigBase):
    name: Literal["not"] = "not"
    condition: "ClipAnnotationConditionConfig"


ClipAnnotationConditionConfig = Annotated[
    RecordingSatisfiesConfig
    | IdInListConfig
    | HasTagConfig
    | HasAllTagsConfig
    | HasAnyTagConfig
    | ClipAllOfConfig
    | ClipAnyOfConfig
    | ClipNotConfig
    | ClipAnnotationConditionImportConfig,
    Field(discriminator="name"),
]


def build_clip_annotation_condition(
    config: ClipAnnotationConditionConfig,
    base_dir: data.PathLike | None = None,
) -> ClipAnnotationCondition:
    return clip_annotation_conditions.build(config, base_dir)
