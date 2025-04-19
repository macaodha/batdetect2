from collections import Counter
from functools import partial
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple

from pydantic import Field, field_validator
from soundevent import data

from batdetect2.configs import BaseConfig, load_config
from batdetect2.targets.terms import (
    GENERIC_CLASS_KEY,
    TagInfo,
    TermRegistry,
    get_tag_from_info,
    term_registry,
)

__all__ = [
    "SoundEventEncoder",
    "SoundEventDecoder",
    "TargetClass",
    "ClassesConfig",
    "load_classes_config",
    "load_encoder_from_config",
    "load_decoder_from_config",
    "build_sound_event_encoder",
    "build_sound_event_decoder",
    "build_generic_class_tags",
    "get_class_names_from_config",
    "DEFAULT_SPECIES_LIST",
]

SoundEventEncoder = Callable[[data.SoundEventAnnotation], Optional[str]]
"""Type alias for a sound event class encoder function.

An encoder function takes a sound event annotation and returns the string name
of the target class it belongs to, based on a predefined set of rules.
If the annotation does not match any defined target class according to the
rules, the function returns None.
"""


SoundEventDecoder = Callable[[str], List[data.Tag]]
"""Type alias for a sound event class decoder function.

A decoder function takes a class name string (as predicted by the model or
assigned during encoding) and returns a list of `soundevent.data.Tag` objects
that represent that class according to the configuration. This is used to
translate model outputs back into meaningful annotations.
"""

DEFAULT_SPECIES_LIST = [
    "Barbastella barbastellus",
    "Eptesicus serotinus",
    "Myotis alcathoe",
    "Myotis bechsteinii",
    "Myotis brandtii",
    "Myotis daubentonii",
    "Myotis mystacinus",
    "Myotis nattereri",
    "Nyctalus leisleri",
    "Nyctalus noctula",
    "Pipistrellus nathusii",
    "Pipistrellus pipistrellus",
    "Pipistrellus pygmaeus",
    "Plecotus auritus",
    "Plecotus austriacus",
    "Rhinolophus ferrumequinum",
    "Rhinolophus hipposideros",
]
"""A default list of common bat species names found in the UK."""


class TargetClass(BaseConfig):
    """Defines criteria for encoding annotations and decoding predictions.

    Each instance represents one potential output class for the classification
    model. It specifies:
    1. A unique `name` for the class.
    2. The tag conditions (`tags` and `match_type`) an annotation must meet to
       be assigned this class name during training data preparation (encoding).
    3. An optional, alternative set of tags (`output_tags`) to be used when
       converting a model's prediction of this class name back into annotation
       tags (decoding).

    Attributes
    ----------
    name : str
        The unique name assigned to this target class (e.g., 'pippip',
        'myodau', 'noise'). This name is used as the label during model
        training and is the expected output from the model's prediction.
        Should be unique across all TargetClass definitions in a configuration.
    tags : List[TagInfo]
        A list of one or more tags (defined using `TagInfo`) used to identify
        if an existing annotation belongs to this class during encoding (data
        preparation for training). The `match_type` attribute determines how
        these tags are evaluated.
    match_type : Literal["all", "any"], default="all"
        Determines how the `tags` list is evaluated during encoding:
        - "all": The annotation must have *all* the tags listed to match.
        - "any": The annotation must have *at least one* of the tags listed
                 to match.
    output_tags: Optional[List[TagInfo]], default=None
        An optional list of tags (defined using `TagInfo`) to be assigned to a
        new annotation when the model predicts this class `name`. If `None`
        (default), the tags listed in the `tags` field will be used for
        decoding. If provided, this list overrides the `tags` field for the
        purpose of decoding predictions back into meaningful annotation tags.
        This allows, for example, training on broader categories but decoding
        to more specific representative tags.
    """

    name: str
    tags: List[TagInfo] = Field(min_length=1)
    match_type: Literal["all", "any"] = Field(default="all")
    output_tags: Optional[List[TagInfo]] = None


def _get_default_classes() -> List[TargetClass]:
    """Generate a list of default target classes.

    Returns
    -------
    List[TargetClass]
        A list of TargetClass objects, one for each species in
        DEFAULT_SPECIES_LIST. The class names are simplified versions of the
        species names.
    """
    return [
        TargetClass(
            name=_get_default_class_name(value),
            tags=[TagInfo(key=GENERIC_CLASS_KEY, value=value)],
        )
        for value in DEFAULT_SPECIES_LIST
    ]


def _get_default_class_name(species: str) -> str:
    """Generate a default class name from a species name.

    Parameters
    ----------
    species : str
        The species name (e.g., "Myotis daubentonii").

    Returns
    -------
    str
        A simplified class name (e.g., "myodau").
        The genus and species names are converted to lowercase,
        the first three letters of each are taken, and concatenated.
    """
    genus, species = species.strip().split(" ")
    return f"{genus.lower()[:3]}{species.lower()[:3]}"


def _get_default_generic_class() -> List[TagInfo]:
    """Generate the default list of TagInfo objects for the generic class.

    Provides a default set of tags used to represent the generic "Bat" category
    when decoding predictions that didn't match a specific class.

    Returns
    -------
    List[TagInfo]
        A list containing default TagInfo objects, typically representing
        `call_type: Echolocation` and `order: Chiroptera`.
    """
    return [
        TagInfo(key="call_type", value="Echolocation"),
        TagInfo(key="order", value="Chiroptera"),
    ]


class ClassesConfig(BaseConfig):
    """Configuration defining target classes and the generic fallback category.

    Holds the ordered list of specific target class definitions (`TargetClass`)
    and defines the tags representing the generic category for sounds that pass
    filtering but do not match any specific class.

    The order of `TargetClass` objects in the `classes` list defines the
    priority for classification during encoding. The system checks annotations
    against these definitions sequentially and assigns the name of the *first*
    matching class.

    Attributes
    ----------
    classes : List[TargetClass]
        An ordered list of specific target class definitions. The order
        determines matching priority (first match wins). Defaults to a
        standard set of classes via `get_default_classes`.
    generic_class : List[TagInfo]
        A list of tags defining the "generic" or "unclassified but relevant"
        category (e.g., representing a generic 'Bat' call that wasn't
        assigned to a specific species). These tags are typically assigned
        during decoding when a sound event was detected and passed filtering
        but did not match any specific class rule defined in the `classes` list.
        Defaults to a standard set of tags via `get_default_generic_class`.

    Raises
    ------
    ValueError
        If validation fails (e.g., non-unique class names in the `classes`
        list).

    Notes
    -----
    - It is crucial that the `name` attribute of each `TargetClass` in the
      `classes` list is unique. This configuration includes a validator to
      enforce this uniqueness.
    - The `generic_class` tags provide a baseline identity for relevant sounds
      that don't fit into more specific defined categories.
    """

    classes: List[TargetClass] = Field(default_factory=_get_default_classes)

    generic_class: List[TagInfo] = Field(
        default_factory=_get_default_generic_class
    )

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


def _is_target_class(
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


def _encode_with_multiple_classifiers(
    sound_event_annotation: data.SoundEventAnnotation,
    classifiers: List[Tuple[str, Callable[[data.SoundEventAnnotation], bool]]],
) -> Optional[str]:
    """Encode an annotation by checking against a list of classifiers.

    Internal helper function used by the `SoundEventEncoder`. It iterates
    through the provided list of (class_name, classifier_function) pairs.
    Returns the name associated with the first classifier function that
    returns True for the given annotation.

    Parameters
    ----------
    sound_event_annotation : data.SoundEventAnnotation
        The annotation to encode.
    classifiers : List[Tuple[str, Callable[[data.SoundEventAnnotation], bool]]]
        An ordered list where each tuple contains a class name and a function
        that returns True if the annotation matches that class. The order
        determines priority.

    Returns
    -------
    str or None
        The name of the first matching class, or None if no classifier matches.
    """
    for class_name, classifier in classifiers:
        if classifier(sound_event_annotation):
            return class_name

    return None


def build_sound_event_encoder(
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
                _is_target_class,
                tags={
                    get_tag_from_info(tag_info, term_registry=term_registry)
                    for tag_info in class_info.tags
                },
                match_all=class_info.match_type == "all",
            ),
        )
        for class_info in config.classes
    ]

    return partial(
        _encode_with_multiple_classifiers,
        classifiers=binary_classifiers,
    )


def _decode_class(
    name: str,
    mapping: Dict[str, List[data.Tag]],
    raise_on_error: bool = True,
) -> List[data.Tag]:
    """Decode a class name into a list of representative tags using a mapping.

    Internal helper function used by the `SoundEventDecoder`. Looks up the
    provided class `name` in the `mapping` dictionary.

    Parameters
    ----------
    name : str
        The class name to decode.
    mapping : Dict[str, List[data.Tag]]
        A dictionary mapping class names to lists of `soundevent.data.Tag`
        objects.
    raise_on_error : bool, default=True
        If True, raises a ValueError if the `name` is not found in the
        `mapping`. If False, returns an empty list if the `name` is not found.

    Returns
    -------
    List[data.Tag]
        The list of tags associated with the class name, or an empty list if
        not found and `raise_on_error` is False.

    Raises
    ------
    ValueError
        If `name` is not found in `mapping` and `raise_on_error` is True.
    """
    if name not in mapping and raise_on_error:
        raise ValueError(f"Class {name} not found in mapping.")

    if name not in mapping:
        return []

    return mapping[name]


def build_sound_event_decoder(
    config: ClassesConfig,
    term_registry: TermRegistry = term_registry,
    raise_on_unmapped: bool = False,
) -> SoundEventDecoder:
    """Build a sound event decoder function from the classes configuration.

    Creates a callable `SoundEventDecoder` that maps a class name string
    back to a list of representative `soundevent.data.Tag` objects based on
    the `ClassesConfig`. It uses the `output_tags` field if provided in a
    `TargetClass`, otherwise falls back to the `tags` field.

    Parameters
    ----------
    config : ClassesConfig
        The loaded and validated classes configuration object.
    term_registry : TermRegistry, optional
        The TermRegistry instance used to look up term keys. Defaults to the
        global `batdetect2.targets.terms.registry`.
    raise_on_unmapped : bool, default=False
        If True, the returned decoder function will raise a ValueError if asked
        to decode a class name that is not in the configuration. If False, it
        will return an empty list for unmapped names.

    Returns
    -------
    SoundEventDecoder
        A callable function that takes a class name string and returns a list
        of `soundevent.data.Tag` objects.

    Raises
    ------
    KeyError
        If a term key specified in the configuration (`output_tags`, `tags`, or
        `generic_class`) is not found in the provided `term_registry`.
    """
    mapping = {}
    for class_info in config.classes:
        tags_to_use = (
            class_info.output_tags
            if class_info.output_tags is not None
            else class_info.tags
        )
        mapping[class_info.name] = [
            get_tag_from_info(tag_info, term_registry=term_registry)
            for tag_info in tags_to_use
        ]

    return partial(
        _decode_class,
        mapping=mapping,
        raise_on_error=raise_on_unmapped,
    )


def build_generic_class_tags(
    config: ClassesConfig,
    term_registry: TermRegistry = term_registry,
) -> List[data.Tag]:
    """Extract and build the list of tags for the generic class from config.

    Converts the list of `TagInfo` objects defined in `config.generic_class`
    into a list of `soundevent.data.Tag` objects using the term registry.

    Parameters
    ----------
    config : ClassesConfig
        The loaded classes configuration object.
    term_registry : TermRegistry, optional
        The TermRegistry instance for term lookups. Defaults to the global
        `batdetect2.targets.terms.registry`.

    Returns
    -------
    List[data.Tag]
        The list of fully constructed tags representing the generic class.

    Raises
    ------
    KeyError
        If a term key specified in `config.generic_class` is not found in the
        provided `term_registry`.
    """
    return [
        get_tag_from_info(tag_info, term_registry=term_registry)
        for tag_info in config.generic_class
    ]


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
    return build_sound_event_encoder(config, term_registry=term_registry)


def load_decoder_from_config(
    path: data.PathLike,
    field: Optional[str] = None,
    term_registry: TermRegistry = term_registry,
    raise_on_unmapped: bool = False,
) -> SoundEventDecoder:
    """Load a class decoder function directly from a configuration file.

    This is a convenience function that combines loading the `ClassesConfig`
    from a file and building the final `SoundEventDecoder` function.

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
    raise_on_unmapped : bool, default=False
        If True, the returned decoder function will raise a ValueError if asked
        to decode a class name that is not in the configuration. If False, it
        will return an empty list for unmapped names.

    Returns
    -------
    SoundEventDecoder
        The final decoder function ready to convert class names back into tags.

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
    return build_sound_event_decoder(
        config,
        term_registry=term_registry,
        raise_on_unmapped=raise_on_unmapped,
    )
