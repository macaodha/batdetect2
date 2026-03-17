"""Provides base classes and utilities for loading configurations in BatDetect2.

This module leverages Pydantic for robust configuration handling, ensuring
that configuration files (typically YAML) adhere to predefined schemas. It
defines a base configuration class (`BaseConfig`) that enforces strict schema
validation and a utility function (`load_config`) to load and validate
configuration data from files, with optional support for accessing nested
configuration sections.
"""

from typing import Any, Type, TypeVar, overload

import yaml
from deepmerge.merger import Merger
from pydantic import BaseModel, ConfigDict, TypeAdapter
from soundevent.data import PathLike

__all__ = [
    "BaseConfig",
    "load_config",
    "merge_configs",
]


class BaseConfig(BaseModel):
    """Base class for all configuration models in BatDetect2.

    Inherits from Pydantic's `BaseModel` to provide data validation, parsing,
    and serialization capabilities.
    """

    model_config = ConfigDict(extra="ignore")

    def to_yaml_string(
        self,
        exclude_none: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
    ) -> str:
        """Converts the Pydantic model instance to a YAML string.

        Parameters
        ----------
        exclude_none : bool, default=False
            Whether to exclude fields whose value is `None`.
        exclude_unset : bool, default=False
            Whether to exclude fields that were not explicitly set.
        exclude_defaults : bool, default=False
            Whether to exclude fields whose value is the default value.

        Returns
        -------
        str
            A YAML string representation of the model.
        """
        return yaml.dump(
            self.model_dump(
                mode="json",
                exclude_none=exclude_none,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
            )
        )

    @classmethod
    def from_yaml(cls, yaml_str: str):
        return cls.model_validate(yaml.safe_load(yaml_str))


T = TypeVar("T")
T_Model = TypeVar("T_Model", bound=BaseModel)
Schema = Type[T_Model] | TypeAdapter[T]


def get_object_field(obj: dict, current_key: str) -> Any:
    """Access a potentially nested field within a dictionary using dot notation.

    Parameters
    ----------
    obj : dict
        The dictionary (or nested dictionaries) to access.
    field : str
        The field name to retrieve. Nested fields are specified using dots
        (e.g., "parent_key.child_key.target_field").

    Returns
    -------
    Any
        The value found at the specified field path.

    Raises
    ------
    KeyError
        If any part of the field path does not exist in the dictionary
        structure.
    TypeError
        If an intermediate part of the path exists but is not a dictionary,
        preventing further nesting.

    Examples
    --------
    >>> data = {"a": {"b": {"c": 10}}}
    >>> get_object_field(data, "a.b.c")
    10
    >>> get_object_field(data, "a.b")
    {'c': 10}
    >>> get_object_field(data, "a")
    {'b': {'c': 10}}
    >>> get_object_field(data, "x")
    Traceback (most recent call last):
        ...
    KeyError: 'x'
    >>> get_object_field(data, "a.x.c")
    Traceback (most recent call last):
        ...
    KeyError: 'x'
    """
    if "." not in current_key:
        return obj.get(current_key, {})

    current_key, rest = current_key.split(".", 1)
    subobj = obj[current_key]

    if not isinstance(subobj, dict):
        raise TypeError(
            f"Intermediate key '{current_key}' in path '{current_key}' "
            f"does not lead to a dictionary (found type: {type(subobj)}). "
            "Cannot access further nested field."
        )

    return get_object_field(subobj, rest)


@overload
def load_config(
    path: PathLike,
    schema: Type[T_Model],
    field: str | None = None,
) -> T_Model: ...


@overload
def load_config(
    path: PathLike,
    schema: TypeAdapter[T],
    field: str | None = None,
) -> T: ...


def load_config(
    path: PathLike,
    schema: Type[T_Model] | TypeAdapter[T],
    field: str | None = None,
) -> T_Model | T:
    """Load and validate configuration data from a file against a schema.

    Reads a YAML file, optionally extracts a specific section using dot
    notation, and then validates the resulting data against the provided
    Pydantic schema.

    Parameters
    ----------
    path : PathLike
        The path to the configuration file (typically `.yaml`).
    schema : Type[T_Model] | TypeAdapter[T]
        Either a Pydantic `BaseModel` subclass or a `TypeAdapter` instance
        that defines the expected structure and types for the configuration
        data.
    field : str, optional
        A dot-separated string indicating a nested section within the YAML
        file to extract before validation. If None (default), the entire
        file content is validated against the schema.
        Example: `"training.optimizer"` would extract the `optimizer` section
        within the `training` section.

    Returns
    -------
    T_Model | T
        An instance of the schema type, populated and validated with
        data from the configuration file.

    Raises
    ------
    FileNotFoundError
        If the file specified by `path` does not exist.
    yaml.YAMLError
        If the file content is not valid YAML.
    pydantic.ValidationError
        If the loaded configuration data (after optionally extracting the
        `field`) does not conform to the provided `schema` (e.g., missing
        required fields, incorrect types, extra fields if using BaseConfig).
    KeyError
        If `field` is provided and specifies a path where intermediate keys
        do not exist in the loaded YAML data.
    TypeError
        If `field` is provided and specifies a path where an intermediate
        value is not a dictionary, preventing access to nested fields.
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    if field:
        config = get_object_field(config, field)

    if isinstance(schema, TypeAdapter):
        return schema.validate_python(config or {})

    return schema.model_validate(config or {})


default_merger = Merger(
    [],
    ["override"],
    ["override"],
)


def merge_configs(config1: T_Model, config2: T_Model) -> T_Model:
    """Merge two configuration objects."""
    model = type(config1)
    dict1 = config1.model_dump()
    dict2 = config2.model_dump()
    merged = default_merger.merge(dict1, dict2)
    return model.model_validate(merged)
