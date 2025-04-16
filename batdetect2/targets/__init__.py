"""Main entry point for the BatDetect2 Target Definition subsystem.

This package (`batdetect2.targets`) provides the tools and configurations
necessary to define precisely what the BatDetect2 model should learn to detect
and classify from audio data. It involves several conceptual steps, managed
through configuration files and culminating in executable functions:

1.  **Terms (`.terms`)**: Defining a controlled vocabulary for annotation tags.
2.  **Filtering (`.filtering`)**: Selecting relevant sound event annotations.
3.  **Transformation (`.transform`)**: Modifying tags (e.g., standardization,
    derivation).
4.  **Class Definition (`.classes`)**: Mapping tags to specific target class
    names (encoding) and defining how predicted class names map back to tags
    (decoding).

This module exposes the key components for users to configure and utilize this
target definition pipeline, primarily through the `TargetConfig` data structure
and the `Targets` class, which encapsulates the configured processing steps.
"""

from typing import List, Optional

from soundevent import data

from batdetect2.configs import BaseConfig, load_config
from batdetect2.targets.classes import (
    ClassesConfig,
    SoundEventDecoder,
    SoundEventEncoder,
    TargetClass,
    build_decoder_from_config,
    build_encoder_from_config,
    build_generic_class_tags_from_config,
    get_class_names_from_config,
    load_classes_config,
    load_decoder_from_config,
    load_encoder_from_config,
)
from batdetect2.targets.filtering import (
    FilterConfig,
    FilterRule,
    SoundEventFilter,
    build_filter_from_config,
    load_filter_config,
    load_filter_from_config,
)
from batdetect2.targets.terms import (
    TagInfo,
    TermInfo,
    TermRegistry,
    call_type,
    get_tag_from_info,
    get_term_from_key,
    individual,
    register_term,
    term_registry,
)
from batdetect2.targets.transform import (
    DerivationRegistry,
    DeriveTagRule,
    MapValueRule,
    ReplaceRule,
    SoundEventTransformation,
    TransformConfig,
    build_transformation_from_config,
    derivation_registry,
    get_derivation,
    load_transformation_config,
    load_transformation_from_config,
    register_derivation,
)

__all__ = [
    "ClassesConfig",
    "DeriveTagRule",
    "FilterConfig",
    "FilterRule",
    "MapValueRule",
    "ReplaceRule",
    "SoundEventDecoder",
    "SoundEventEncoder",
    "SoundEventFilter",
    "SoundEventTransformation",
    "TagInfo",
    "TargetClass",
    "TargetConfig",
    "Targets",
    "TermInfo",
    "TransformConfig",
    "build_decoder_from_config",
    "build_encoder_from_config",
    "build_filter_from_config",
    "build_generic_class_tags_from_config",
    "build_transformation_from_config",
    "call_type",
    "get_class_names_from_config",
    "get_derivation",
    "get_tag_from_info",
    "get_term_from_key",
    "individual",
    "load_classes_config",
    "load_decoder_from_config",
    "load_encoder_from_config",
    "load_filter_config",
    "load_filter_from_config",
    "load_target_config",
    "load_transformation_config",
    "load_transformation_from_config",
    "register_derivation",
    "register_term",
]


class TargetConfig(BaseConfig):
    """Unified configuration for the entire target definition pipeline.

    This model aggregates the configurations for the optional filtering and
    transformation steps, and the mandatory class definition step. It serves as
    the primary input for building a complete `Targets` processing object.

    Attributes
    ----------
    filtering : FilterConfig, optional
        Configuration for filtering sound event annotations. If None or
        omitted, no filtering is applied.
    transforms : TransformConfig, optional
        Configuration for transforming annotation tags. If None or omitted, no
        transformations are applied.
    classes : ClassesConfig
        Configuration defining the specific target classes, their matching
        rules, decoding rules (`output_tags`), and the generic class
        definition. This section is mandatory.
    """

    filtering: Optional[FilterConfig] = None

    transforms: Optional[TransformConfig] = None

    classes: ClassesConfig


def load_target_config(
    path: data.PathLike,
    field: Optional[str] = None,
) -> TargetConfig:
    """Load the unified target configuration from a file.

    Reads a configuration file (typically YAML) and validates it against the
    `TargetConfig` schema, potentially extracting data from a nested field.

    Parameters
    ----------
    path : data.PathLike
        Path to the configuration file.
    field : str, optional
        Dot-separated path to a nested section within the file containing the
        target configuration. If None, the entire file content is used.

    Returns
    -------
    TargetConfig
        The loaded and validated unified target configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file path does not exist.
    yaml.YAMLError
        If the file content is not valid YAML.
    pydantic.ValidationError
        If the loaded configuration data does not conform to the
        `TargetConfig` schema (including validation within nested configs
        like `ClassesConfig`).
    KeyError, TypeError
        If `field` specifies an invalid path within the loaded data.
    """
    return load_config(path=path, schema=TargetConfig, field=field)


class Targets:
    """Encapsulates the complete configured target definition pipeline.

    This class holds the functions for filtering, transforming, encoding, and
    decoding annotations based on a loaded `TargetConfig`. It provides a
    high-level interface to apply these steps and access relevant metadata
    like class names and generic class tags.

    Instances are typically created using the `Targets.from_config` or
    `Targets.from_file` classmethods.

    Attributes
    ----------
    class_names : list[str]
        An ordered list of the unique names of the specific target classes
        defined in the configuration.
    generic_class_tags : List[data.Tag]
        A list of `soundevent.data.Tag` objects representing the configured
        generic class (e.g., the default 'Bat' class).
    """

    class_names: list[str]
    generic_class_tags: List[data.Tag]

    def __init__(
        self,
        encode_fn: SoundEventEncoder,
        decode_fn: SoundEventDecoder,
        class_names: list[str],
        generic_class_tags: List[data.Tag],
        filter_fn: Optional[SoundEventFilter] = None,
        transform_fn: Optional[SoundEventTransformation] = None,
    ):
        """Initialize the Targets object.

        Parameters
        ----------
        encode_fn : SoundEventEncoder
            The configured function to encode annotations to class names.
        decode_fn : SoundEventDecoder
            The configured function to decode class names to tags.
        class_names : list[str]
            The ordered list of specific target class names.
        generic_class_tags : List[data.Tag]
            The list of tags representing the generic class.
        filter_fn : SoundEventFilter, optional
            The configured function to filter annotations. Defaults to None (no
            filtering).
        transform_fn : SoundEventTransformation, optional
            The configured function to transform annotation tags. Defaults to
            None (no transformation).
        """
        self.class_names = class_names
        self.generic_class_tags = generic_class_tags

        self._filter_fn = filter_fn
        self._encode_fn = encode_fn
        self._decode_fn = decode_fn
        self._transform_fn = transform_fn

    def filter(self, sound_event: data.SoundEventAnnotation) -> bool:
        """Apply the configured filter to a sound event annotation.

        Parameters
        ----------
        sound_event : data.SoundEventAnnotation
            The annotation to filter.

        Returns
        -------
        bool
            True if the annotation should be kept (passes the filter),
            False otherwise. If no filter was configured, always returns True.
        """
        if not self._filter_fn:
            return True
        return self._filter_fn(sound_event)

    def encode(self, sound_event: data.SoundEventAnnotation) -> Optional[str]:
        """Encode a sound event annotation to its target class name.

        Applies the configured class definition rules (including priority)
        to determine the specific class name for the annotation.

        Parameters
        ----------
        sound_event : data.SoundEventAnnotation
            The annotation to encode. Note: This should typically be called
            *after* applying any transformations via the `transform` method.

        Returns
        -------
        str or None
            The name of the matched target class, or None if the annotation
            does not match any specific class rule (i.e., it belongs to the
            generic category).
        """
        return self._encode_fn(sound_event)

    def decode(self, class_label: str) -> List[data.Tag]:
        """Decode a predicted class name back into representative tags.

        Uses the configured mapping (based on `TargetClass.output_tags` or
        `TargetClass.tags`) to convert a class name string into a list of
        `soundevent.data.Tag` objects.

        Parameters
        ----------
        class_label : str
            The class name to decode.

        Returns
        -------
        List[data.Tag]
            The list of tags corresponding to the input class name.
        """
        return self._decode_fn(class_label)

    def transform(
        self, sound_event: data.SoundEventAnnotation
    ) -> data.SoundEventAnnotation:
        """Apply the configured tag transformations to an annotation.

        Parameters
        ----------
        sound_event : data.SoundEventAnnotation
            The annotation whose tags should be transformed.

        Returns
        -------
        data.SoundEventAnnotation
            A new annotation object with the transformed tags. If no
            transformations were configured, the original annotation object is
            returned.
        """
        if self._transform_fn:
            return self._transform_fn(sound_event)
        return sound_event

    @classmethod
    def from_config(
        cls,
        config: TargetConfig,
        term_registry: TermRegistry = term_registry,
        derivation_registry: DerivationRegistry = derivation_registry,
    ) -> "Targets":
        """Build a Targets object from a loaded TargetConfig.

        This factory method takes the unified configuration object and
        constructs all the necessary functional components (filter, transform,
        encoder, decoder) and extracts metadata (class names, generic tags) to
        create a fully configured `Targets` instance.

        Parameters
        ----------
        config : TargetConfig
            The loaded and validated unified target configuration object.
        term_registry : TermRegistry, optional
            The TermRegistry instance to use for resolving term keys. Defaults
            to the global `batdetect2.targets.terms.term_registry`.
        derivation_registry : DerivationRegistry, optional
            The DerivationRegistry instance to use for resolving derivation
            function names. Defaults to the global
            `batdetect2.targets.transform.derivation_registry`.

        Returns
        -------
        Targets
            An initialized `Targets` object ready for use.

        Raises
        ------
        KeyError
            If term keys or derivation function keys specified in the `config`
            are not found in their respective registries.
        ImportError, AttributeError, TypeError
            If dynamic import of a derivation function fails (when configured).
        """
        filter_fn = (
            build_filter_from_config(
                config.filtering,
                term_registry=term_registry,
            )
            if config.filtering
            else None
        )
        encode_fn = build_encoder_from_config(
            config.classes,
            term_registry=term_registry,
        )
        decode_fn = build_decoder_from_config(
            config.classes,
            term_registry=term_registry,
        )
        transform_fn = (
            build_transformation_from_config(
                config.transforms,
                term_registry=term_registry,
                derivation_registry=derivation_registry,
            )
            if config.transforms
            else None
        )
        class_names = get_class_names_from_config(config.classes)
        generic_class_tags = build_generic_class_tags_from_config(
            config.classes,
            term_registry=term_registry,
        )

        return cls(
            filter_fn=filter_fn,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            class_names=class_names,
            generic_class_tags=generic_class_tags,
            transform_fn=transform_fn,
        )

    @classmethod
    def from_file(
        cls,
        config_path: data.PathLike,
        field: Optional[str] = None,
        term_registry: TermRegistry = term_registry,
        derivation_registry: DerivationRegistry = derivation_registry,
    ) -> "Targets":
        """Load a Targets object directly from a configuration file.

        This convenience factory method loads the `TargetConfig` from the
        specified file path and then calls `Targets.from_config` to build
        the fully initialized `Targets` object.

        Parameters
        ----------
        config_path : data.PathLike
            Path to the configuration file (e.g., YAML).
        field : str, optional
            Dot-separated path to a nested section within the file containing
            the target configuration. If None, the entire file content is used.
        term_registry : TermRegistry, optional
            The TermRegistry instance to use. Defaults to the global default.
        derivation_registry : DerivationRegistry, optional
            The DerivationRegistry instance to use. Defaults to the global
            default.

        Returns
        -------
        Targets
            An initialized `Targets` object ready for use.

        Raises
        ------
        FileNotFoundError, yaml.YAMLError, pydantic.ValidationError, KeyError,
        TypeError
            Errors raised during file loading, validation, or extraction via
            `load_target_config`.
        KeyError, ImportError, AttributeError, TypeError
            Errors raised during the build process by `Targets.from_config`
            (e.g., missing keys in registries, failed imports).
        """
        config = load_target_config(
            config_path,
            field=field,
        )
        return cls.from_config(
            config,
            term_registry=term_registry,
            derivation_registry=derivation_registry,
        )
