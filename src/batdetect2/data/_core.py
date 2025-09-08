from typing import Generic, Protocol, Type, TypeVar

from pydantic import BaseModel

__all__ = [
    "Registry",
]

T_Config = TypeVar("T_Config", bound=BaseModel, contravariant=True)
T_Type = TypeVar("T_Type", covariant=True)


class LogicProtocol(Generic[T_Config, T_Type], Protocol):
    """A generic protocol for the logic classes (conditions or transforms)."""

    @classmethod
    def from_config(cls, config: T_Config) -> T_Type: ...


T_Proto = TypeVar("T_Proto", bound=LogicProtocol)


class Registry(Generic[T_Type]):
    """A generic class to create and manage a registry of items."""

    def __init__(self, name: str):
        self._name = name
        self._registry = {}

    def register(self, config_cls: Type[T_Config]):
        """A decorator factory to register a new item."""
        fields = config_cls.model_fields

        if "name" not in fields:
            raise ValueError("Configuration object must have a 'name' field.")

        name = fields["name"].default

        if not isinstance(name, str):
            raise ValueError("'name' field must be a string literal.")

        def decorator(logic_cls: Type[T_Proto]) -> Type[T_Proto]:
            self._registry[name] = logic_cls
            return logic_cls

        return decorator

    def build(self, config: BaseModel) -> T_Type:
        """Builds a logic instance from a config object."""

        name = getattr(config, "name")  # noqa: B009

        if name is None:
            raise ValueError("Config does not have a name field")

        if name not in self._registry:
            raise NotImplementedError(
                f"No {self._name} with name '{name}' is registered."
            )

        return self._registry[name].from_config(config)
