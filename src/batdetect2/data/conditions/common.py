import csv
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Generic, Literal, ParamSpec, Protocol, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field, model_validator
from soundevent import data

from batdetect2.core.configs import BaseConfig
from batdetect2.core.registries import Registry

__all__ = [
    "AllOf",
    "AnyOf",
    "Condition",
    "CsvList",
    "HasAllTags",
    "HasAllTagsConfig",
    "HasAnyTag",
    "HasAnyTagConfig",
    "HasTag",
    "HasTagConfig",
    "IdInList",
    "IdInListConfig",
    "JsonList",
    "ListLoader",
    "ListFormatConfig",
    "MultiConditionConfigBase",
    "Not",
    "NotConditionConfigBase",
    "ObjectWithTags",
    "ObjectWithUUID",
    "TxtList",
    "build_list_loader",
    "register_all_of_condition",
    "register_any_of_condition",
    "register_has_all_tags_condition",
    "register_has_any_tag_condition",
    "register_has_tag_condition",
    "register_id_in_list_condition",
    "register_not_condition",
]


class ObjectWithTags(Protocol):
    tags: list[data.Tag]


class ObjectWithUUID(Protocol):
    uuid: UUID


ConditionObject = TypeVar("ConditionObject")
TaggedObject = TypeVar("TaggedObject", bound="ObjectWithTags")
UUIDObject = TypeVar("UUIDObject", bound="ObjectWithUUID")
P = ParamSpec("P")
NotConfigType = TypeVar("NotConfigType", bound="NotConditionConfigBase")
MultiConfigType = TypeVar(
    "MultiConfigType",
    bound="MultiConditionConfigBase",
)
Condition = Callable[[ConditionObject], bool]


class NotConditionConfigBase(BaseConfig):
    condition: BaseModel


class MultiConditionConfigBase(BaseConfig):
    conditions: Sequence[BaseModel]


class Not(Generic[ConditionObject]):
    def __init__(self, condition: Condition[ConditionObject]):
        self.condition = condition

    def __call__(self, obj: ConditionObject) -> bool:
        return not self.condition(obj)


class AllOf(Generic[ConditionObject]):
    def __init__(self, conditions: Sequence[Condition[ConditionObject]]):
        self.conditions = list(conditions)

    def __call__(self, obj: ConditionObject) -> bool:
        return all(condition(obj) for condition in self.conditions)


class AnyOf(Generic[ConditionObject]):
    def __init__(self, conditions: Sequence[Condition[ConditionObject]]):
        self.conditions = list(conditions)

    def __call__(self, obj: ConditionObject) -> bool:
        return any(condition(obj) for condition in self.conditions)


class HasTag(Generic[TaggedObject]):
    def __init__(self, tag: data.Tag):
        self.tag_key = (tag.term.name, tag.value)

    def __call__(self, obj: TaggedObject) -> bool:
        return any(
            (tag.term.name, tag.value) == self.tag_key for tag in obj.tags
        )


class HasAllTags(Generic[TaggedObject]):
    def __init__(self, tags: list[data.Tag]):
        if not tags:
            raise ValueError("Need to specify at least one tag")

        self.required_keys = {(tag.term.name, tag.value) for tag in tags}

    def __call__(self, obj: TaggedObject) -> bool:
        tag_keys = {(tag.term.name, tag.value) for tag in obj.tags}
        return self.required_keys.issubset(tag_keys)


class HasAnyTag(Generic[TaggedObject]):
    def __init__(self, tags: list[data.Tag]):
        if not tags:
            raise ValueError("Need to specify at least one tag")

        self.required_keys = {(tag.term.name, tag.value) for tag in tags}

    def __call__(self, obj: TaggedObject) -> bool:
        tag_keys = {(tag.term.name, tag.value) for tag in obj.tags}
        return bool(self.required_keys.intersection(tag_keys))


class IdInList(Generic[UUIDObject]):
    def __init__(self, ids: set[UUID]):
        self.ids = ids

    def __call__(self, obj: UUIDObject) -> bool:
        return obj.uuid in self.ids


class HasTagConfig(BaseConfig):
    name: Literal["has_tag"] = "has_tag"
    tag: data.Tag


class HasAllTagsConfig(BaseConfig):
    name: Literal["has_all_tags"] = "has_all_tags"
    tags: list[data.Tag]


class HasAnyTagConfig(BaseConfig):
    name: Literal["has_any_tag"] = "has_any_tag"
    tags: list[data.Tag]


class JsonList(BaseConfig):
    name: Literal["json"] = "json"
    field: str | None = None


class TxtList(BaseConfig):
    name: Literal["txt"] = "txt"


class CsvList(BaseConfig):
    name: Literal["csv"] = "csv"
    column: str


ListFormatConfig = Annotated[
    JsonList | TxtList | CsvList,
    Field(discriminator="name"),
]


ListLoader = Callable[[Path], list[str]]

list_loaders: Registry[ListLoader, []] = Registry("list_loader")


class IdInListConfig(BaseConfig):
    name: Literal["id_in_list"] = "id_in_list"
    path: Path
    format: ListFormatConfig = JsonList()

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


class JsonListLoader:
    def __init__(self, field: str | None):
        self.field = field

    def __call__(self, path: Path) -> list[str]:
        content = json.loads(path.read_text())

        if self.field is not None:
            if not isinstance(content, dict):
                raise TypeError(
                    "Expected JSON object with field for 'id_in_list'."
                )

            if self.field not in content:
                raise KeyError(f"Field '{self.field}' not found in '{path}'.")

            content = content[self.field]

        if not isinstance(content, list):
            raise TypeError("Expected JSON list with IDs for 'id_in_list'.")

        return [str(value) for value in content]

    @list_loaders.register(JsonList)
    @staticmethod
    def from_config(config: JsonList) -> ListLoader:
        return JsonListLoader(field=config.field)


class TxtListLoader:
    def __call__(self, path: Path) -> list[str]:
        return [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    @list_loaders.register(TxtList)
    @staticmethod
    def from_config(config: TxtList) -> ListLoader:
        return TxtListLoader()


class CsvListLoader:
    def __init__(self, column: str):
        self.column = column

    def __call__(self, path: Path) -> list[str]:
        with path.open("r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)

            if reader.fieldnames is None:
                raise ValueError(
                    f"Expected CSV header row for 'id_in_list' in '{path}'."
                )

            if self.column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{self.column}' not found in '{path}'."
                )

            values = []
            for row in reader:
                value = row.get(self.column)

                if value is None:
                    continue

                value = value.strip()

                if not value:
                    continue

                values.append(value)

            return values

    @list_loaders.register(CsvList)
    @staticmethod
    def from_config(config: CsvList) -> ListLoader:
        return CsvListLoader(column=config.column)


def build_list_loader(config: ListFormatConfig) -> ListLoader:
    return list_loaders.build(config)


def register_id_in_list_condition(
    registry: Registry[Condition[UUIDObject], [data.PathLike | None]],
    config_cls: type[IdInListConfig],
) -> None:
    def builder(
        config: IdInListConfig,
        base_dir: data.PathLike | None = None,
    ) -> Condition[UUIDObject]:
        path = config.path

        if base_dir is not None and not path.is_absolute():
            path = Path(base_dir) / path

        ids = set()
        loader = build_list_loader(config.format)
        values = loader(path)
        for index, value in enumerate(values):
            try:
                ids.add(UUID(value))
            except ValueError as err:
                raise ValueError(
                    f"Invalid ID at index {index} in '{path}': {value!r}."
                ) from err

        return IdInList(ids)

    registry.register(config_cls)(builder)


def register_has_tag_condition(
    registry: Registry[Condition[TaggedObject], P],
    config_cls: type[HasTagConfig],
) -> None:
    def builder(
        config: HasTagConfig,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Condition[TaggedObject]:
        return HasTag(config.tag)

    registry.register(config_cls)(builder)


def register_has_all_tags_condition(
    registry: Registry[Condition[TaggedObject], P],
    config_cls: type[HasAllTagsConfig],
) -> None:
    def builder(
        config: HasAllTagsConfig,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Condition[TaggedObject]:
        return HasAllTags(config.tags)

    registry.register(config_cls)(builder)


def register_has_any_tag_condition(
    registry: Registry[Condition[TaggedObject], P],
    config_cls: type[HasAnyTagConfig],
) -> None:
    def builder(
        config: HasAnyTagConfig,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Condition[TaggedObject]:
        return HasAnyTag(config.tags)

    registry.register(config_cls)(builder)


def register_not_condition(
    registry: Registry[Condition[ConditionObject], P],
) -> Callable[[type[NotConfigType]], type[NotConfigType]]:
    def decorator(config_cls: type[NotConfigType]) -> type[NotConfigType]:
        @registry.register(config_cls)
        def builder(
            config: NotConfigType,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Condition[ConditionObject]:
            condition = registry.build(config.condition, *args, **kwargs)
            return Not(condition)

        return config_cls

    return decorator


def register_all_of_condition(
    registry: Registry[Condition[ConditionObject], P],
) -> Callable[[type[MultiConfigType]], type[MultiConfigType]]:
    def decorator(config_cls: type[MultiConfigType]) -> type[MultiConfigType]:
        @registry.register(config_cls)
        def builder(
            config: MultiConfigType,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Condition[ConditionObject]:
            conditions = [
                registry.build(condition, *args, **kwargs)
                for condition in config.conditions
            ]
            return AllOf(conditions)

        return config_cls

    return decorator


def register_any_of_condition(
    registry: Registry[Condition[ConditionObject], P],
) -> Callable[[type[MultiConfigType]], type[MultiConfigType]]:
    def decorator(config_cls: type[MultiConfigType]) -> type[MultiConfigType]:
        @registry.register(config_cls)
        def builder(
            config: MultiConfigType,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Condition[ConditionObject]:
            conditions = [
                registry.build(condition, *args, **kwargs)
                for condition in config.conditions
            ]
            return AnyOf(conditions)

        return config_cls

    return decorator
