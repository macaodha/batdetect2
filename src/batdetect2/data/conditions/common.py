import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Generic, Literal, ParamSpec, Protocol, TypeVar
from uuid import UUID

from pydantic import BaseModel
from soundevent import data

from batdetect2.core.configs import BaseConfig
from batdetect2.core.registries import Registry

__all__ = [
    "AllOf",
    "AnyOf",
    "Condition",
    "HasAllTags",
    "HasAllTagsConfig",
    "HasAnyTag",
    "HasAnyTagConfig",
    "HasTag",
    "HasTagConfig",
    "IdInList",
    "IdInListConfig",
    "MultiConditionConfigBase",
    "Not",
    "NotConditionConfigBase",
    "ObjectWithTags",
    "ObjectWithUUID",
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


class IdInListConfig(BaseConfig):
    name: Literal["id_in_list"] = "id_in_list"
    path: Path
    list_format: Literal["json", "txt"] = "json"


def _load_ids(
    path: Path,
    list_format: Literal["json", "txt"],
) -> list[str]:
    if list_format == "json":
        content = json.loads(path.read_text())

        if not isinstance(content, list):
            raise TypeError("Expected JSON list with IDs for 'id_in_list'.")

        return [str(value) for value in content]

    return [
        line.strip() for line in path.read_text().splitlines() if line.strip()
    ]


def register_id_in_list_condition(
    registry: Registry[Condition[UUIDObject], [data.PathLike | None]],
) -> Callable[[type[IdInListConfig]], type[IdInListConfig]]:
    def decorator(config_cls: type[IdInListConfig]) -> type[IdInListConfig]:
        @registry.register(config_cls)
        def builder(
            config: IdInListConfig,
            base_dir: data.PathLike | None = None,
        ) -> Condition[UUIDObject]:
            path = config.path

            if base_dir is not None and not path.is_absolute():
                path = Path(base_dir) / path

            ids = set()
            for index, value in enumerate(_load_ids(path, config.list_format)):
                try:
                    ids.add(UUID(value))
                except ValueError as err:
                    raise ValueError(
                        f"Invalid ID at index {index} in '{path}': {value!r}."
                    ) from err

            return IdInList(ids)

        return config_cls

    return decorator


def register_has_tag_condition(
    registry: Registry[Condition[TaggedObject], P],
) -> Callable[[type[HasTagConfig]], type[HasTagConfig]]:
    def decorator(config_cls: type[HasTagConfig]) -> type[HasTagConfig]:
        @registry.register(config_cls)
        def builder(
            config: HasTagConfig,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Condition[TaggedObject]:
            return HasTag(config.tag)

        return config_cls

    return decorator


def register_has_all_tags_condition(
    registry: Registry[Condition[TaggedObject], P],
) -> Callable[[type[HasAllTagsConfig]], type[HasAllTagsConfig]]:
    def decorator(
        config_cls: type[HasAllTagsConfig],
    ) -> type[HasAllTagsConfig]:
        @registry.register(config_cls)
        def builder(
            config: HasAllTagsConfig,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Condition[TaggedObject]:
            return HasAllTags(config.tags)

        return config_cls

    return decorator


def register_has_any_tag_condition(
    registry: Registry[Condition[TaggedObject], P],
) -> Callable[[type[HasAnyTagConfig]], type[HasAnyTagConfig]]:
    def decorator(
        config_cls: type[HasAnyTagConfig],
    ) -> type[HasAnyTagConfig]:
        @registry.register(config_cls)
        def builder(
            config: HasAnyTagConfig,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Condition[TaggedObject]:
            return HasAnyTag(config.tags)

        return config_cls

    return decorator


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
