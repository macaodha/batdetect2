import importlib
from functools import partial
from typing import (
    Annotated,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
)

from pydantic import Field
from soundevent import data

from batdetect2.configs import BaseConfig, load_config
from batdetect2.targets.terms import (
    TagInfo,
    TermRegistry,
    get_tag_from_info,
    get_term_from_key,
)
from batdetect2.targets.terms import (
    term_registry as default_term_registry,
)

__all__ = [
    "SoundEventTransformation",
    "MapValueRule",
    "DeriveTagRule",
    "ReplaceRule",
    "TransformConfig",
    "DerivationRegistry",
    "derivation_registry",
    "get_derivation",
    "build_transform_from_rule",
    "build_transformation_from_config",
    "load_transformation_config",
    "load_transformation_from_config",
]

SoundEventTransformation = Callable[
    [data.SoundEventAnnotation], data.SoundEventAnnotation
]
"""Type alias for a sound event transformation function.

A function that accepts a sound event annotation object and returns a
(potentially) modified sound event annotation object. Transformations
should generally return a copy of the annotation rather than modifying
it in place.
"""


Derivation = Callable[[str], str]
"""Type alias for a derivation function.

A function that accepts a single string (typically a tag value) and returns
a new string (the derived value).
"""


class MapValueRule(BaseConfig):
    """Configuration for mapping specific values of a source term.

    This rule replaces tags matching a specific term and one of the
    original values with a new tag (potentially having a different term)
    containing the corresponding replacement value. Useful for standardizing
    or grouping tag values.

    Attributes
    ----------
    rule_type : Literal["map_value"]
        Discriminator field identifying this rule type.
    source_term_key : str
        The key (registered in `TermRegistry`) of the term whose tags' values
        should be checked against the `value_mapping`.
    value_mapping : Dict[str, str]
        A dictionary mapping original string values to replacement string
        values. Only tags whose value is a key in this dictionary will be
        affected.
    target_term_key : str, optional
        The key (registered in `TermRegistry`) for the term of the *output*
        tag. If None (default), the output tag uses the same term as the
        source (`source_term_key`). If provided, the term of the affected
        tag is changed to this target term upon replacement.
    """

    rule_type: Literal["map_value"] = "map_value"
    source_term_key: str
    value_mapping: Dict[str, str]
    target_term_key: Optional[str] = None


def map_value_transform(
    sound_event_annotation: data.SoundEventAnnotation,
    source_term: data.Term,
    target_term: data.Term,
    mapping: Dict[str, str],
) -> data.SoundEventAnnotation:
    """Apply a value mapping transformation to an annotation's tags.

    Iterates through the annotation's tags. If a tag matches the `source_term`
    and its value is found in the `mapping`, it is replaced by a new tag with
    the `target_term` and the mapped value. Other tags are kept unchanged.

    Parameters
    ----------
    sound_event_annotation : data.SoundEventAnnotation
        The annotation to transform.
    source_term : data.Term
        The term of tags whose values should be mapped.
    target_term : data.Term
        The term to use for the newly created tags after mapping.
    mapping : Dict[str, str]
        The dictionary mapping original values to new values.

    Returns
    -------
    data.SoundEventAnnotation
        A new annotation object with the transformed tags.
    """
    tags = []

    for tag in sound_event_annotation.tags:
        if tag.term != source_term or tag.value not in mapping:
            tags.append(tag)
            continue

        new_value = mapping[tag.value]
        tags.append(data.Tag(term=target_term, value=new_value))

    return sound_event_annotation.model_copy(update=dict(tags=tags))


class DeriveTagRule(BaseConfig):
    """Configuration for deriving a new tag from an existing tag's value.

    This rule applies a specified function (`derivation_function`) to the
    value of tags matching the `source_term_key`. It then adds a new tag
    with the `target_term_key` and the derived value.

    Attributes
    ----------
    rule_type : Literal["derive_tag"]
        Discriminator field identifying this rule type.
    source_term_key : str
        The key (registered in `TermRegistry`) of the term whose tag values
        will be used as input to the derivation function.
    derivation_function : str
        The name/key identifying the derivation function to use. This can be
        a key registered in the `DerivationRegistry` or, if
        `import_derivation` is True, a full Python path like
        `'my_module.my_submodule.my_function'`.
    target_term_key : str, optional
        The key (registered in `TermRegistry`) for the term of the new tag
        that will be created with the derived value. If None (default), the
        derived tag uses the same term as the source (`source_term_key`),
        effectively performing an in-place value transformation.
    import_derivation : bool, default=False
        If True, treat `derivation_function` as a Python import path and
        attempt to dynamically import it if not found in the registry.
        Requires the function to be accessible in the Python environment.
    keep_source : bool, default=True
        If True, the original source tag (whose value was used for derivation)
        is kept in the annotation's tag list alongside the newly derived tag.
        If False, the original source tag is removed.
    """

    rule_type: Literal["derive_tag"] = "derive_tag"
    source_term_key: str
    derivation_function: str
    target_term_key: Optional[str] = None
    import_derivation: bool = False
    keep_source: bool = True


def derivation_tag_transform(
    sound_event_annotation: data.SoundEventAnnotation,
    source_term: data.Term,
    target_term: data.Term,
    derivation: Derivation,
    keep_source: bool = True,
) -> data.SoundEventAnnotation:
    """Apply a derivation transformation to an annotation's tags.

    Iterates through the annotation's tags. For each tag matching the
    `source_term`, its value is passed to the `derivation` function.
    A new tag is created with the `target_term` and the derived value,
    and added to the output tag list. The original source tag is kept
    or discarded based on `keep_source`. Other tags are kept unchanged.

    Parameters
    ----------
    sound_event_annotation : data.SoundEventAnnotation
        The annotation to transform.
    source_term : data.Term
        The term of tags whose values serve as input for the derivation.
    target_term : data.Term
        The term to use for the newly created derived tags.
    derivation : Derivation
        The function to apply to the source tag's value.
    keep_source : bool, default=True
        Whether to keep the original source tag in the output.

    Returns
    -------
    data.SoundEventAnnotation
        A new annotation object with the transformed tags (including derived
                                                           ones).
    """
    tags = []

    for tag in sound_event_annotation.tags:
        if tag.term != source_term:
            tags.append(tag)
            continue

        if keep_source:
            tags.append(tag)

        new_value = derivation(tag.value)
        tags.append(data.Tag(term=target_term, value=new_value))

    return sound_event_annotation.model_copy(update=dict(tags=tags))


class ReplaceRule(BaseConfig):
    """Configuration for exactly replacing one specific tag with another.

    This rule looks for an exact match of the `original` tag (both term and
    value) and replaces it with the specified `replacement` tag.

    Attributes
    ----------
    rule_type : Literal["replace"]
        Discriminator field identifying this rule type.
    original : TagInfo
        The exact tag to search for, defined using its value and term key.
    replacement : TagInfo
        The tag to substitute in place of the original tag, defined using
        its value and term key.
    """

    rule_type: Literal["replace"] = "replace"
    original: TagInfo
    replacement: TagInfo


def replace_tag_transform(
    sound_event_annotation: data.SoundEventAnnotation,
    source: data.Tag,
    target: data.Tag,
) -> data.SoundEventAnnotation:
    """Apply an exact tag replacement transformation.

    Iterates through the annotation's tags. If a tag exactly matches the
    `source` tag, it is replaced by the `target` tag. Other tags are kept
    unchanged.

    Parameters
    ----------
    sound_event_annotation : data.SoundEventAnnotation
        The annotation to transform.
    source : data.Tag
        The exact tag to find and replace.
    target : data.Tag
        The tag to replace the source tag with.

    Returns
    -------
    data.SoundEventAnnotation
        A new annotation object with the replaced tag (if found).
    """
    tags = []

    for tag in sound_event_annotation.tags:
        if tag == source:
            tags.append(target)
        else:
            tags.append(tag)

    return sound_event_annotation.model_copy(update=dict(tags=tags))


class TransformConfig(BaseConfig):
    """Configuration model for defining a sequence of transformation rules.

    Attributes
    ----------
    rules : List[Union[ReplaceRule, MapValueRule, DeriveTagRule]]
        A list of transformation rules to apply. The rules are applied
        sequentially in the order they appear in the list. The output of
        one rule becomes the input for the next. The `rule_type` field
        discriminates between the different rule models.
    """

    rules: List[
        Annotated[
            Union[ReplaceRule, MapValueRule, DeriveTagRule],
            Field(discriminator="rule_type"),
        ]
    ] = Field(
        default_factory=list,
    )


class DerivationRegistry(Mapping[str, Derivation]):
    """A registry for managing named derivation functions.

    Derivation functions are callables that take a string value and return
    a transformed string value, used by `DeriveTagRule`. This registry
    allows functions to be registered with a key and retrieved later.
    """

    def __init__(self):
        """Initialize an empty DerivationRegistry."""
        self._derivations: Dict[str, Derivation] = {}

    def __getitem__(self, key: str) -> Derivation:
        """Retrieve a derivation function by key."""
        return self._derivations[key]

    def __len__(self) -> int:
        """Return the number of registered derivation functions."""
        return len(self._derivations)

    def __iter__(self):
        """Return an iterator over the keys of registered functions."""
        return iter(self._derivations)

    def register(self, key: str, derivation: Derivation) -> None:
        """Register a derivation function with a unique key.

        Parameters
        ----------
        key : str
            The unique key to associate with the derivation function.
        derivation : Derivation
            The callable derivation function (takes str, returns str).

        Raises
        ------
        KeyError
            If a derivation function with the same key is already registered.
        """
        if key in self._derivations:
            raise KeyError(
                f"A derivation with the provided key {key} already exists"
            )

        self._derivations[key] = derivation

    def get_derivation(self, key: str) -> Derivation:
        """Retrieve a derivation function by its registered key.

        Parameters
        ----------
        key : str
            The key of the derivation function to retrieve.

        Returns
        -------
        Derivation
            The requested derivation function.

        Raises
        ------
        KeyError
            If no derivation function with the specified key is registered.
        """
        try:
            return self._derivations[key]
        except KeyError as err:
            raise KeyError(
                f"No derivation with key {key} is registered."
            ) from err

    def get_keys(self) -> List[str]:
        """Get a list of all registered derivation function keys.

        Returns
        -------
        List[str]
            The keys of all registered functions.
        """
        return list(self._derivations.keys())

    def get_derivations(self) -> List[Derivation]:
        """Get a list of all registered derivation functions.

        Returns
        -------
        List[Derivation]
            The registered derivation function objects.
        """
        return list(self._derivations.values())


derivation_registry = DerivationRegistry()
"""Global instance of the DerivationRegistry.

Register custom derivation functions here to make them available by key
in `DeriveTagRule` configuration.
"""


def get_derivation(
    key: str,
    import_derivation: bool = False,
    registry: DerivationRegistry = derivation_registry,
):
    """Retrieve a derivation function by key, optionally importing it.

    First attempts to find the function in the provided `registry`.
    If not found and `import_derivation` is True, attempts to dynamically
    import the function using the `key` as a full Python path
    (e.g., 'my_module.submodule.my_func').

    Parameters
    ----------
    key : str
        The key or Python path of the derivation function.
    import_derivation : bool, default=False
        If True, attempt dynamic import if key is not in the registry.
    registry : DerivationRegistry, optional
        The registry instance to check first. Defaults to the global
        `derivation_registry`.

    Returns
    -------
    Derivation
        The requested derivation function.

    Raises
    ------
    KeyError
        If the key is not found in the registry and either
        `import_derivation` is False or the dynamic import fails.
    ImportError
        If dynamic import fails specifically due to module not found.
    AttributeError
        If dynamic import fails because the function name isn't in the module.
    """
    if not import_derivation or key in registry:
        return registry.get_derivation(key)

    try:
        module_path, func_name = key.rsplit(".", 1)
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        return func
    except ImportError as err:
        raise KeyError(
            f"Unable to load derivation '{key}'. Check the path and ensure "
            "it points to a valid callable function in an importable module."
        ) from err


def build_transform_from_rule(
    rule: Union[ReplaceRule, MapValueRule, DeriveTagRule],
    derivation_registry: DerivationRegistry = derivation_registry,
    term_registry: TermRegistry = default_term_registry,
) -> SoundEventTransformation:
    """Build a specific SoundEventTransformation function from a rule config.

    Selects the appropriate transformation logic based on the rule's
    `rule_type`, fetches necessary terms and derivation functions, and
    returns a partially applied function ready to transform an annotation.

    Parameters
    ----------
    rule : Union[ReplaceRule, MapValueRule, DeriveTagRule]
        The configuration object for a single transformation rule.
    registry : DerivationRegistry, optional
        The derivation registry to use for `DeriveTagRule`. Defaults to the
        global `derivation_registry`.

    Returns
    -------
    SoundEventTransformation
        A callable that applies the specified rule to a SoundEventAnnotation.

    Raises
    ------
    KeyError
        If required term keys or derivation keys are not found.
    ValueError
        If the rule has an unknown `rule_type`.
    ImportError, AttributeError, TypeError
        If dynamic import of a derivation function fails.
    """
    if rule.rule_type == "replace":
        source = get_tag_from_info(
            rule.original,
            term_registry=term_registry,
        )
        target = get_tag_from_info(
            rule.replacement,
            term_registry=term_registry,
        )
        return partial(replace_tag_transform, source=source, target=target)

    if rule.rule_type == "derive_tag":
        source_term = get_term_from_key(
            rule.source_term_key,
            term_registry=term_registry,
        )
        target_term = (
            get_term_from_key(
                rule.target_term_key,
                term_registry=term_registry,
            )
            if rule.target_term_key
            else source_term
        )
        derivation = get_derivation(
            key=rule.derivation_function,
            import_derivation=rule.import_derivation,
            registry=derivation_registry,
        )
        return partial(
            derivation_tag_transform,
            source_term=source_term,
            target_term=target_term,
            derivation=derivation,
            keep_source=rule.keep_source,
        )

    if rule.rule_type == "map_value":
        source_term = get_term_from_key(
            rule.source_term_key,
            term_registry=term_registry,
        )
        target_term = (
            get_term_from_key(
                rule.target_term_key,
                term_registry=term_registry,
            )
            if rule.target_term_key
            else source_term
        )
        return partial(
            map_value_transform,
            source_term=source_term,
            target_term=target_term,
            mapping=rule.value_mapping,
        )

    # Handle unknown rule type
    valid_options = ["replace", "derive_tag", "map_value"]
    # Should be caught by Pydantic validation, but good practice
    raise ValueError(
        f"Invalid transform rule type '{getattr(rule, 'rule_type', 'N/A')}'. "
        f"Valid options are: {valid_options}"
    )


def build_transformation_from_config(
    config: TransformConfig,
    derivation_registry: DerivationRegistry = derivation_registry,
    term_registry: TermRegistry = default_term_registry,
) -> SoundEventTransformation:
    """Build a composite transformation function from a TransformConfig.

    Creates a sequence of individual transformation functions based on the
    rules defined in the configuration. Returns a single function that
    applies these transformations sequentially to an annotation.

    Parameters
    ----------
    config : TransformConfig
        The configuration object containing the list of transformation rules.
    derivation_reg : DerivationRegistry, optional
        The derivation registry to use when building `DeriveTagRule`
        transformations. Defaults to the global `derivation_registry`.

    Returns
    -------
    SoundEventTransformation
        A single function that applies all configured transformations in order.
    """
    transforms = [
        build_transform_from_rule(
            rule,
            derivation_registry=derivation_registry,
            term_registry=term_registry,
        )
        for rule in config.rules
    ]

    def transformation(
        sound_event_annotation: data.SoundEventAnnotation,
    ) -> data.SoundEventAnnotation:
        for transform in transforms:
            sound_event_annotation = transform(sound_event_annotation)
        return sound_event_annotation

    return transformation


def load_transformation_config(
    path: data.PathLike, field: Optional[str] = None
) -> TransformConfig:
    """Load the transformation configuration from a file.

    Parameters
    ----------
    path : data.PathLike
        Path to the configuration file (YAML).
    field : str, optional
        If the transformation configuration is nested under a specific key
        in the file, specify the key here. Defaults to None.

    Returns
    -------
    TransformConfig
        The loaded and validated transformation configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file path does not exist.
    pydantic.ValidationError
        If the config file structure does not match the TransformConfig schema.
    """
    return load_config(path=path, schema=TransformConfig, field=field)


def load_transformation_from_config(
    path: data.PathLike,
    field: Optional[str] = None,
    derivation_registry: DerivationRegistry = derivation_registry,
    term_registry: TermRegistry = default_term_registry,
) -> SoundEventTransformation:
    """Load transformation config from a file and build the final function.

    This is a convenience function that combines loading the configuration
    and building the final callable transformation function that applies
    all rules sequentially.

    Parameters
    ----------
    path : data.PathLike
        Path to the configuration file (YAML).
    field : str, optional
        If the transformation configuration is nested under a specific key
        in the file, specify the key here. Defaults to None.

    Returns
    -------
    SoundEventTransformation
        The final composite transformation function ready to be used.

    Raises
    ------
    FileNotFoundError
        If the config file path does not exist.
    pydantic.ValidationError
        If the config file structure does not match the TransformConfig schema.
    KeyError
        If required term keys or derivation keys specified in the config
        are not found during the build process.
    ImportError, AttributeError, TypeError
        If dynamic import of a derivation function specified in the config
        fails.
    """
    config = load_transformation_config(path=path, field=field)
    return build_transformation_from_config(
        config,
        derivation_registry=derivation_registry,
        term_registry=term_registry,
    )
