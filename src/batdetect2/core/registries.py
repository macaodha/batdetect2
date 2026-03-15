from typing import (
    Callable,
    Concatenate,
    Generic,
    ParamSpec,
    Type,
    TypeVar,
)

from pydantic import BaseModel

__all__ = [
    "Registry",
    "SimpleRegistry",
]

T_Config = TypeVar("T_Config", bound=BaseModel, contravariant=True)
T_Type = TypeVar("T_Type", covariant=True)
P_Type = ParamSpec("P_Type")


T = TypeVar("T")


class SimpleRegistry(Generic[T]):
    def __init__(self, name: str):
        self._name = name
        self._registry = {}

    def register(self, name: str):
        def decorator(obj: T) -> T:
            self._registry[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> T:
        return self._registry[name]

    def has(self, name: str) -> bool:
        return name in self._registry


class Registry(Generic[T_Type, P_Type]):
    """A generic class to create and manage a registry of items."""

    def __init__(self, name: str, discriminator: str = "name"):
        self._name = name
        self._registry: dict[
            str, Callable[Concatenate[..., P_Type], T_Type]
        ] = {}
        self._discriminator = discriminator
        self._config_types: dict[str, Type[BaseModel]] = {}

    def register(
        self,
        config_cls: Type[T_Config],
    ):
        fields = config_cls.model_fields

        if self._discriminator not in fields:
            raise ValueError(
                "Configuration object must have "
                f"a '{self._discriminator}' field."
            )

        name = fields[self._discriminator].default

        self._config_types[name] = config_cls

        if not isinstance(name, str):
            raise ValueError(
                f"'{self._discriminator}' field must be a string literal."
            )

        def decorator(
            func: Callable[Concatenate[T_Config, P_Type], T_Type],
        ):
            self._registry[name] = func
            return func

        return decorator

    def get_config_types(self) -> tuple[Type[BaseModel], ...]:
        return tuple(self._config_types.values())

    def get_config_type(self, name: str) -> Type[BaseModel]:
        try:
            return self._config_types[name]
        except KeyError as err:
            raise ValueError(
                f"No config type with name '{name}' is registered. "
                f"Existing config types: {list(self._config_types.keys())}"
            ) from err

    def build(
        self,
        config: BaseModel,
        *args: P_Type.args,
        **kwargs: P_Type.kwargs,
    ) -> T_Type:
        """Builds a logic instance from a config object."""

        name = getattr(config, self._discriminator)  # noqa: B009

        if name is None:
            raise ValueError(
                f"Config does not have a '{self._discriminator}' field"
            )

        if name not in self._registry:
            raise NotImplementedError(
                f"No {self._name} with name '{name}' is registered."
            )

        return self._registry[name](config, *args, **kwargs)
