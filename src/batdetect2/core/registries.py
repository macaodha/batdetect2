import sys
from typing import Generic, Protocol, Type, TypeVar

from pydantic import BaseModel

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

__all__ = [
    "Registry",
]

T_Config = TypeVar("T_Config", bound=BaseModel, contravariant=True)
T_Type = TypeVar("T_Type", covariant=True)
P_Type = ParamSpec("P_Type")


class LogicProtocol(Generic[T_Config, T_Type, P_Type], Protocol):
    """A generic protocol for the logic classes."""

    @classmethod
    def from_config(
        cls,
        config: T_Config,
        *args: P_Type.args,
        **kwargs: P_Type.kwargs,
    ) -> T_Type: ...


T_Proto = TypeVar("T_Proto", bound=LogicProtocol)


class Registry(Generic[T_Type, P_Type]):
    """A generic class to create and manage a registry of items."""

    def __init__(self, name: str):
        self._name = name
        self._registry = {}

    def register(
        self,
        config_cls: Type[T_Config],
        logic_cls: LogicProtocol[T_Config, T_Type, P_Type],
    ) -> None:
        """A decorator factory to register a new item."""
        fields = config_cls.model_fields

        if "name" not in fields:
            raise ValueError("Configuration object must have a 'name' field.")

        name = fields["name"].default

        if not isinstance(name, str):
            raise ValueError("'name' field must be a string literal.")

        self._registry[name] = logic_cls

    def build(
        self,
        config: BaseModel,
        *args: P_Type.args,
        **kwargs: P_Type.kwargs,
    ) -> T_Type:
        """Builds a logic instance from a config object."""

        name = getattr(config, "name")  # noqa: B009

        if name is None:
            raise ValueError("Config does not have a name field")

        if name not in self._registry:
            raise NotImplementedError(
                f"No {self._name} with name '{name}' is registered."
            )

        return self._registry[name].from_config(config, *args, **kwargs)
