from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Literal

from loguru import logger
from pydantic import Field, model_validator
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
    JsonList,
    ListFormatConfig,
    MultiConditionConfigBase,
    NotConditionConfigBase,
    build_list_loader,
    list_loaders,
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
    "PathInListConfig",
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


class PathInListConfig(BaseConfig):
    name: Literal["path_in_list"] = "path_in_list"
    path: Path
    format: ListFormatConfig = JsonList()
    base_dir: Path | None = None
    on_outside: Literal["allow", "warn", "error"] = "allow"

    @model_validator(mode="before")
    @classmethod
    def _normalize_format(cls, values):
        if not isinstance(values, dict):
            return values

        format_config = values.get("format")

        if isinstance(format_config, str):
            values = values.copy()
            config_class = list_loaders.get_config_type(format_config)
            values["format"] = config_class().model_dump()

        return values


class PathInList:
    def __init__(
        self,
        paths: set[Path],
        base_dir: Path | None,
        on_outside: Literal["allow", "warn", "error"],
    ):
        self.paths = paths
        self.base_dir = base_dir
        self.on_outside = on_outside

    def __call__(self, recording: data.Recording) -> bool:
        normalized_path = self._normalize_recording_path(recording.path)

        if normalized_path is None:
            return True

        return normalized_path in self.paths

    def _normalize_recording_path(self, path: data.PathLike) -> Path | None:
        recording_path = Path(path)

        if self.base_dir is None:
            return recording_path

        if not recording_path.is_absolute():
            return recording_path

        try:
            return recording_path.relative_to(self.base_dir)
        except ValueError as err:
            if self.on_outside == "allow":
                return None

            if self.on_outside == "warn":
                logger.warning(
                    "Recording path '{}' is outside '{}' in path_in_list; "
                    "allowing.",
                    recording_path,
                    self.base_dir,
                )
                return None

            raise ValueError(
                f"Recording path '{recording_path}' is outside "
                f"'{self.base_dir}' for 'path_in_list'."
            ) from err

    @recording_conditions.register(PathInListConfig)
    @staticmethod
    def from_config(
        config: PathInListConfig,
        base_dir: data.PathLike | None = None,
    ) -> "PathInList":
        list_path = config.path

        if base_dir is not None and not list_path.is_absolute():
            list_path = Path(base_dir) / list_path

        match_base_dir = config.base_dir
        if (
            match_base_dir is not None
            and base_dir is not None
            and not match_base_dir.is_absolute()
        ):
            match_base_dir = Path(base_dir) / match_base_dir

        loader = build_list_loader(config.format)

        paths = {
            Path(value).relative_to(match_base_dir)
            if (
                match_base_dir is not None
                and Path(value).is_absolute()
                and Path(value).is_relative_to(match_base_dir)
            )
            else Path(value)
            for value in loader(list_path)
        }

        return PathInList(
            paths=paths,
            base_dir=match_base_dir,
            on_outside=config.on_outside,
        )


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
    | PathInListConfig
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
