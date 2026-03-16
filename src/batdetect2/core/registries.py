from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    ParamSpec,
    Type,
    TypeVar,
)

from hydra.utils import instantiate
from pydantic import BaseModel, Field

__all__ = [
    "add_import_config",
    "ImportConfig",
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


class ImportConfig(BaseModel):
    """Base config for dynamic instantiation via Hydra.

    Subclass this to create a registry-specific import escape hatch.
    The subclass must add a discriminator field whose name matches the
    registry's own discriminator key, with its value fixed to
    ``Literal["import"]``.

    Attributes
    ----------
    target : str
        Fully-qualified dotted path to the callable to instantiate,
        e.g. ``"mypackage.module.MyClass"``.
    arguments : dict[str, Any]
        Base keyword arguments forwarded to the callable. When the
        same key also appears in ``kwargs`` passed to ``build()``,
        the ``kwargs`` value takes priority.
    """

    target: str
    arguments: dict[str, Any] = Field(default_factory=dict)


T_Import = TypeVar("T_Import", bound=ImportConfig)


def add_import_config(
    registry: Registry[T_Type, P_Type],
) -> Callable[[Type[T_Import]], Type[T_Import]]:
    """Decorator that registers an ImportConfig subclass as an escape hatch.

    Wraps the decorated class in a builder that calls
    ``hydra.utils.instantiate`` using ``config.target`` and
    ``config.arguments``. The builder is registered on *registry*
    under the discriminator value ``"import"``.

    Parameters
    ----------
    registry : Registry
        The registry instance on which the config should be registered.

    Returns
    -------
    Callable[[type[ImportConfig]], type[ImportConfig]]
        A class decorator that registers the class and returns it
        unchanged.

    Examples
    --------
    Define a per-registry import escape hatch::

        @add_import_config(my_registry)
        class MyRegistryImportConfig(ImportConfig):
            name: Literal["import"] = "import"
    """

    def decorator(config_cls: Type[T_Import]) -> Type[T_Import]:
        def builder(
            config: T_Import,
            *args: P_Type.args,
            **kwargs: P_Type.kwargs,
        ) -> T_Type:
            if len(args) > 0:
                raise ValueError(
                    "Positional arguments are not supported "
                    "for import escape hatch."
                )

            hydra_cfg = {
                "_target_": config.target,
                **config.arguments,
                **kwargs,
            }
            return instantiate(hydra_cfg)

        registry.register(config_cls)(builder)
        return config_cls

    return decorator
