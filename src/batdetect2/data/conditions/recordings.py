from collections.abc import Callable, Sequence
from typing import Annotated, Literal

from pydantic import Field
from soundevent import data

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
    "IdInListConfig",
    "RecordingAllOfConfig",
    "RecordingAnyOfConfig",
    "RecordingCondition",
    "RecordingConditionConfig",
    "RecordingConditionImportConfig",
    "RecordingNotConfig",
    "build_recording_condition",
]

RecordingCondition = Callable[[data.Recording], bool]

recording_conditions: Registry[RecordingCondition, [data.PathLike | None]] = (
    Registry("recording_condition")
)


@add_import_config(recording_conditions, arg_names=["base_dir"])
class RecordingConditionImportConfig(ImportConfig):
    """Use any callable as a recording condition.

    Set ``name="import"`` and provide a ``target`` pointing to any callable
    to use it instead of a built-in option.
    """

    name: Literal["import"] = "import"


register_id_in_list_condition(recording_conditions, IdInListConfig)
register_has_tag_condition(recording_conditions, HasTagConfig)
register_has_all_tags_condition(recording_conditions, HasAllTagsConfig)
register_has_any_tag_condition(recording_conditions, HasAnyTagConfig)


@register_all_of_condition(recording_conditions)
class RecordingAllOfConfig(MultiConditionConfigBase):
    name: Literal["all_of"] = "all_of"
    conditions: Sequence["RecordingConditionConfig"]


@register_any_of_condition(recording_conditions)
class RecordingAnyOfConfig(MultiConditionConfigBase):
    name: Literal["any_of"] = "any_of"
    conditions: Sequence["RecordingConditionConfig"]


@register_not_condition(recording_conditions)
class RecordingNotConfig(NotConditionConfigBase):
    name: Literal["not"] = "not"
    condition: "RecordingConditionConfig"


RecordingConditionConfig = Annotated[
    IdInListConfig
    | HasTagConfig
    | HasAllTagsConfig
    | HasAnyTagConfig
    | RecordingAllOfConfig
    | RecordingAnyOfConfig
    | RecordingNotConfig
    | RecordingConditionImportConfig,
    Field(discriminator="name"),
]


def build_recording_condition(
    config: RecordingConditionConfig,
    base_dir: data.PathLike | None = None,
) -> RecordingCondition:
    return recording_conditions.build(config, base_dir)
