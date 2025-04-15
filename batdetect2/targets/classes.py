from collections import Counter
from functools import partial
from typing import Callable, List, Literal, Optional, Set

from pydantic import Field, field_validator
from soundevent import data

from batdetect2.configs import BaseConfig, load_config
from batdetect2.targets.terms import (
    TagInfo,
    TermRegistry,
    get_tag_from_info,
    term_registry,
)

__all__ = [
    "SoundEventEncoder",
    "TargetClass",
    "ClassesConfig",
    "load_classes_config",
    "build_encoder_from_config",
    "load_encoder_from_config",
    "get_class_names_from_config",
]

SoundEventEncoder = Callable[[data.SoundEventAnnotation], Optional[str]]
"""Type alias for a sound event class encoder function.

An encoder function takes a sound event annotation and returns the string name
of the target class it belongs to, based on a predefined set of rules.
If the annotation does not match any defined target class according to the
rules, the function returns None.
"""


class TargetClass(BaseConfig):
    """Defines the criteria for assigning an annotation to a specific class.

    Each instance represents one potential output class for the classification
    model. It specifies the class name and the tag conditions an annotation
    must meet to be assigned this class label.

    Attributes
    ----------
    name : str
        The unique name assigned to this target class (e.g., 'pippip',
        'myodau', 'noise'). This name will be used as the label during model
        training and output. Should be unique across all TargetClass
        definitions in a configuration.
    tag : List[TagInfo]
        A list of one or more tags (defined using `TagInfo`) that an annotation
        must possess to potentially match this class.
    match_type : Literal["all", "any"], default="all"
        Determines how the `tag` list is evaluated:
        - "all": The annotation must have *all* the tags listed in the `tag`
                 field to match this class definition.
        - "any": The annotation must have *at least one* of the tags listed
                 in the `tag` field to match this class definition.
    """

    name: str
    tags: List[TagInfo] = Field(default_factory=list, min_length=1)
    match_type: Literal["all", "any"] = Field(default="all")


class ClassesConfig(BaseConfig):
    """Configuration model holding the list of target class definitions.

    The order of `TargetClass` objects in the `classes` list defines the
    priority for classification. When encoding an annotation, the system checks
    against the class definitions in this sequence and assigns the name of the
    *first* matching class.

    Attributes
    ----------
    classes : List[TargetClass]
        An ordered list of target class definitions. The order determines
        matching priority (first match wins).

    Raises
    ------
    ValueError
        If validation fails (e.g., non-unique class names).

    Notes
    -----
    It is crucial that the `name` attribute of each `TargetClass` in the
    `classes` list is unique. This configuration includes a validator to
    enforce this uniqueness.
    """

    classes: List[TargetClass] = Field(default_factory=list)

    @field_validator("classes")
    def check_unique_class_names(cls, v: List[TargetClass]):
        """Ensure all defined class names are unique."""
        names = [c.name for c in v]

        if len(names) != len(set(names)):
            name_counts = Counter(names)
            duplicates = [
                name for name, count in name_counts.items() if count > 1
            ]
            raise ValueError(
                "Class names must be unique. Found duplicates: "
                f"{', '.join(duplicates)}"
            )
        return v


def load_classes_config(
    path: data.PathLike,
    field: Optional[str] = None,
) -> ClassesConfig:
    """Load the target classes configuration from a file.

    Parameters
    ----------
    path : data.PathLike
        Path to the configuration file (YAML).
    field : str, optional
        If the classes configuration is nested under a specific key in the
        file, specify the key here. Defaults to None.

    Returns
    -------
    ClassesConfig
        The loaded and validated classes configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file path does not exist.
    pydantic.ValidationError
        If the config file structure does not match the ClassesConfig schema
        or if class names are not unique.
    """
    return load_config(path, schema=ClassesConfig, field=field)


def is_target_class(
    sound_event_annotation: data.SoundEventAnnotation,
    tags: Set[data.Tag],
    match_all: bool = True,
) -> bool:
    """Check if a sound event annotation matches a set of required tags.

    Parameters
    ----------
    sound_event_annotation : data.SoundEventAnnotation
        The annotation to check.
    required_tags : Set[data.Tag]
        A set of `soundevent.data.Tag` objects that define the class criteria.
    match_all : bool, default=True
        If True, checks if *all* `required_tags` are present in the
        annotation's tags (subset check). If False, checks if *at least one*
        of the `required_tags` is present (intersection check).

    Returns
    -------
    bool
        True if the annotation meets the tag criteria, False otherwise.
    """
    annotation_tags = set(sound_event_annotation.tags)

    if match_all:
        return tags <= annotation_tags

    return bool(tags & annotation_tags)


def get_class_names_from_config(config: ClassesConfig) -> List[str]:
    """Extract the list of class names from a ClassesConfig object.

    Parameters
    ----------
    config : ClassesConfig
        The loaded classes configuration object.

    Returns
    -------
    List[str]
        An ordered list of unique class names defined in the configuration.
    """
    return [class_info.name for class_info in config.classes]


def build_encoder_from_config(
    config: ClassesConfig,
    term_registry: TermRegistry = term_registry,
) -> SoundEventEncoder:
    """Build a sound event encoder function from the classes configuration.

    The returned encoder function iterates through the class definitions in the
    order specified in the config. It assigns an annotation the name of the
    first class definition it matches.

    Parameters
    ----------
    config : ClassesConfig
        The loaded and validated classes configuration object.
    term_registry : TermRegistry, optional
        The TermRegistry instance used to look up term keys specified in the
        `TagInfo` objects within the configuration. Defaults to the global
        `batdetect2.targets.terms.registry`.

    Returns
    -------
    SoundEventEncoder
        A callable function that takes a `SoundEventAnnotation` and returns
        an optional string representing the matched class name, or None if no
        class matches.

    Raises
    ------
    KeyError
        If a term key specified in the configuration is not found in the
        provided `term_registry`.
    """
    binary_classifiers = [
        (
            class_info.name,
            partial(
                is_target_class,
                tags={
                    get_tag_from_info(tag_info, term_registry=term_registry)
                    for tag_info in class_info.tags
                },
                match_all=class_info.match_type == "all",
            ),
        )
        for class_info in config.classes
    ]

    def encoder(
        sound_event_annotation: data.SoundEventAnnotation,
    ) -> Optional[str]:
        """Assign a class name to an annotation based on configured rules.

        Iterates through pre-compiled classifiers in priority order. Returns
        the name of the first matching class, or None if no match is found.

        Parameters
        ----------
        sound_event_annotation : data.SoundEventAnnotation
            The annotation to encode.

        Returns
        -------
        str or None
            The name of the matched class, or None.
        """
        for class_name, classifier in binary_classifiers:
            if classifier(sound_event_annotation):
                return class_name

        return None

    return encoder


def load_encoder_from_config(
    path: data.PathLike,
    field: Optional[str] = None,
    term_registry: TermRegistry = term_registry,
) -> SoundEventEncoder:
    """Load a class encoder function directly from a configuration file.

    This is a convenience function that combines loading the `ClassesConfig`
    from a file and building the final `SoundEventEncoder` function.

    Parameters
    ----------
    path : data.PathLike
        Path to the configuration file (e.g., YAML).
    field : str, optional
        If the classes configuration is nested under a specific key in the
        file, specify the key here. Defaults to None.
    term_registry : TermRegistry, optional
        The TermRegistry instance used for term lookups. Defaults to the
        global `batdetect2.targets.terms.registry`.

    Returns
    -------
    SoundEventEncoder
        The final encoder function ready to classify annotations.

    Raises
    ------
    FileNotFoundError
        If the config file path does not exist.
    pydantic.ValidationError
        If the config file structure does not match the ClassesConfig schema
        or if class names are not unique.
    KeyError
        If a term key specified in the configuration is not found in the
        provided `term_registry` during the build process.
    """
    config = load_classes_config(path, field=field)
    return build_encoder_from_config(config, term_registry=term_registry)
