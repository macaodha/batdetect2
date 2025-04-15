"""Provides base classes and utilities for loading configurations in BatDetect2.

This module leverages Pydantic for robust configuration handling, ensuring
that configuration files (typically YAML) adhere to predefined schemas. It
defines a base configuration class (`BaseConfig`) that enforces strict schema
validation and a utility function (`load_config`) to load and validate
configuration data from files, with optional support for accessing nested
configuration sections.
"""

from typing import Any, Optional, Type, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict
from soundevent.data import PathLike

__all__ = [
    "BaseConfig",
    "load_config",
]


class BaseConfig(BaseModel):
    """Base class for all configuration models in BatDetect2.

    Inherits from Pydantic's `BaseModel` to provide data validation, parsing,
    and serialization capabilities.

    It sets `extra='forbid'` in its model configuration, meaning that any
    fields provided in a configuration file that are *not* explicitly defined
    in the specific configuration schema will raise a validation error. This
    helps catch typos and ensures configurations strictly adhere to the expected
    structure.

    Attributes
    ----------
    model_config : ConfigDict
        Pydantic model configuration dictionary. Set to forbid extra fields.
    """

    model_config = ConfigDict(extra="forbid")


T = TypeVar("T", bound=BaseModel)


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
        return obj[current_key]

    current_key, rest = current_key.split(".", 1)
    subobj = obj[current_key]

    if not isinstance(subobj, dict):
        raise TypeError(
            f"Intermediate key '{current_key}' in path '{current_key}' "
            f"does not lead to a dictionary (found type: {type(subobj)}). "
            "Cannot access further nested field."
        )

    return get_object_field(subobj, rest)


def load_config(
    path: PathLike,
    schema: Type[T],
    field: Optional[str] = None,
) -> T:
    """Load and validate configuration data from a file against a schema.

    Reads a YAML file, optionally extracts a specific section using dot
    notation, and then validates the resulting data against the provided
    Pydantic `schema`.

    Parameters
    ----------
    path : PathLike
        The path to the configuration file (typically `.yaml`).
    schema : Type[T]
        The Pydantic `BaseModel` subclass that defines the expected structure
        and types for the configuration data.
    field : str, optional
        A dot-separated string indicating a nested section within the YAML
        file to extract before validation. If None (default), the entire
        file content is validated against the schema.
        Example: `"training.optimizer"` would extract the `optimizer` section
        within the `training` section.

    Returns
    -------
    T
        An instance of the provided `schema`, populated and validated with
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

    return schema.model_validate(config)
