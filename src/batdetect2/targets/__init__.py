"""Main entry point for the BatDetect2 Target Definition subsystem.

This package (`batdetect2.targets`) provides the tools and configurations
necessary to define precisely what the BatDetect2 model should learn to detect,
classify, and localize from audio data. It involves several conceptual steps,
managed through configuration files and culminating in an executable pipeline:

1.  **Terms (`.terms`)**: Defining vocabulary for annotation tags.
2.  **Filtering (`.filtering`)**: Selecting relevant sound event annotations.
3.  **Transformation (`.transform`)**: Modifying tags (standardization,
    derivation).
4.  **ROI Mapping (`.roi`)**: Defining how annotation geometry (ROIs) maps to
    target position and size representations, and back.
5.  **Class Definition (`.classes`)**: Mapping tags to target class names
    (encoding) and mapping predicted names back to tags (decoding).

This module exposes the key components for users to configure and utilize this
target definition pipeline, primarily through the `TargetConfig` data structure
and the `Targets` class (implementing `TargetProtocol`), which encapsulates the
configured processing steps. The main way to create a functional `Targets`
object is via the `build_targets` or `load_targets` functions.
"""

from typing import Iterable, List, Optional, Tuple

from loguru import logger
from pydantic import Field
from soundevent import data

from batdetect2.configs import BaseConfig, load_config
from batdetect2.targets.classes import (
    ClassesConfig,
    SoundEventDecoder,
    SoundEventEncoder,
    TargetClass,
    build_generic_class_tags,
    build_sound_event_decoder,
    build_sound_event_encoder,
    get_class_names_from_config,
    load_classes_config,
    load_decoder_from_config,
    load_encoder_from_config,
)
from batdetect2.targets.filtering import (
    FilterConfig,
    FilterRule,
    SoundEventFilter,
    build_sound_event_filter,
    load_filter_config,
    load_filter_from_config,
)
from batdetect2.targets.rois import (
    AnchorBBoxMapperConfig,
    ROIMapperConfig,
    ROITargetMapper,
    build_roi_mapper,
)
from batdetect2.targets.terms import (
    TagInfo,
    call_type,
    get_tag_from_info,
    individual,
)
from batdetect2.targets.transform import (
    DerivationRegistry,
    DeriveTagRule,
    MapValueRule,
    ReplaceRule,
    SoundEventTransformation,
    TransformConfig,
    build_transformation_from_config,
    default_derivation_registry,
    get_derivation,
    load_transformation_config,
    load_transformation_from_config,
    register_derivation,
)
from batdetect2.typing.targets import Position, Size, TargetProtocol

__all__ = [
    "ClassesConfig",
    "DEFAULT_TARGET_CONFIG",
    "DeriveTagRule",
    "FilterConfig",
    "FilterRule",
    "MapValueRule",
    "AnchorBBoxMapperConfig",
    "ROITargetMapper",
    "ReplaceRule",
    "SoundEventDecoder",
    "SoundEventEncoder",
    "SoundEventFilter",
    "SoundEventTransformation",
    "TagInfo",
    "TargetClass",
    "TargetConfig",
    "Targets",
    "TransformConfig",
    "build_generic_class_tags",
    "build_roi_mapper",
    "build_sound_event_decoder",
    "build_sound_event_encoder",
    "build_sound_event_filter",
    "build_transformation_from_config",
    "call_type",
    "get_class_names_from_config",
    "get_derivation",
    "get_tag_from_info",
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
]


class TargetConfig(BaseConfig):
    """Unified configuration for the entire target definition pipeline.

    This model aggregates the configurations for semantic processing (filtering,
    transformation, class definition) and geometric processing (ROI mapping).
    It serves as the primary input for building a complete `Targets` object
    via `build_targets` or `load_targets`.

    Attributes
    ----------
    filtering : FilterConfig, optional
        Configuration for filtering sound event annotations based on tags.
        If None or omitted, no filtering is applied.
    transforms : TransformConfig, optional
        Configuration for transforming annotation tags
        (mapping, derivation, etc.). If None or omitted, no tag transformations
        are applied.
    classes : ClassesConfig
        Configuration defining the specific target classes, their tag matching
        rules for encoding, their representative tags for decoding
        (`output_tags`), and the definition of the generic class tags.
        This section is mandatory.
    roi : ROIConfig, optional
        Configuration defining how geometric ROIs (e.g., bounding boxes) are
        mapped to target representations (reference point, scaled size).
        Controls `position`, `time_scale`, `frequency_scale`. If None or
        omitted, default ROI mapping settings are used.
    """

    filtering: FilterConfig = Field(default_factory=FilterConfig)
    transforms: TransformConfig = Field(default_factory=TransformConfig)
    classes: ClassesConfig = Field(
        default_factory=lambda: DEFAULT_CLASSES_CONFIG
    )
    roi: ROIMapperConfig = Field(default_factory=AnchorBBoxMapperConfig)


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


class Targets(TargetProtocol):
    """Encapsulates the complete configured target definition pipeline.

    This class implements the `TargetProtocol`, holding the configured
    functions for filtering, transforming, encoding (tags to class name),
    decoding (class name to tags), and mapping ROIs (geometry to position/size
    and back). It provides a high-level interface to apply these steps and
    access relevant metadata like class names and dimension names.

    Instances are typically created using the `build_targets` factory function
    or the `load_targets` convenience loader.

    Attributes
    ----------
    class_names : List[str]
        An ordered list of the unique names of the specific target classes
        defined in the configuration.
    generic_class_tags : List[data.Tag]
        A list of `soundevent.data.Tag` objects representing the configured
        generic class category (used when no specific class matches).
    dimension_names : List[str]
        The names of the size dimensions handled by the ROI mapper
        (e.g., ['width', 'height']).
    """

    class_names: List[str]
    generic_class_tags: List[data.Tag]
    dimension_names: List[str]

    def __init__(
        self,
        encode_fn: SoundEventEncoder,
        decode_fn: SoundEventDecoder,
        roi_mapper: ROITargetMapper,
        class_names: list[str],
        generic_class_tags: List[data.Tag],
        filter_fn: Optional[SoundEventFilter] = None,
        transform_fn: Optional[SoundEventTransformation] = None,
        roi_mapper_overrides: Optional[dict[str, ROITargetMapper]] = None,
    ):
        """Initialize the Targets object.

        Note: This constructor is typically called internally by the
        `build_targets` factory function.

        Parameters
        ----------
        encode_fn : SoundEventEncoder
            Configured function to encode annotations to class names.
        decode_fn : SoundEventDecoder
            Configured function to decode class names to tags.
        roi_mapper : ROITargetMapper
            Configured object for mapping geometry to/from position/size.
        class_names : list[str]
            Ordered list of specific target class names.
        generic_class_tags : List[data.Tag]
            List of tags representing the generic class.
        filter_fn : SoundEventFilter, optional
            Configured function to filter annotations. Defaults to None.
        transform_fn : SoundEventTransformation, optional
            Configured function to transform annotation tags. Defaults to None.
        """
        self.class_names = class_names
        self.generic_class_tags = generic_class_tags
        self.dimension_names = roi_mapper.dimension_names

        self._roi_mapper = roi_mapper
        self._filter_fn = filter_fn
        self._encode_fn = encode_fn
        self._decode_fn = decode_fn
        self._transform_fn = transform_fn
        self._roi_mapper_overrides = roi_mapper_overrides or {}

        for class_name in self._roi_mapper_overrides:
            if class_name not in self.class_names:
                # TODO: improve this warning
                logger.warning(
                    "The ROI mapper overrides contains a class ({class_name}) "
                    "not present in the class names.",
                    class_name=class_name,
                )

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

    def encode_class(
        self, sound_event: data.SoundEventAnnotation
    ) -> Optional[str]:
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

    def decode_class(self, class_label: str) -> List[data.Tag]:
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

    def encode_roi(
        self, sound_event: data.SoundEventAnnotation
    ) -> tuple[Position, Size]:
        """Extract the target reference position from the annotation's roi.

        Delegates to the internal ROI mapper's `get_roi_position` method.

        Parameters
        ----------
        sound_event : data.SoundEventAnnotation
            The annotation containing the geometry (ROI).

        Returns
        -------
        Tuple[float, float]
            The reference position `(time, frequency)`.

        Raises
        ------
        ValueError
            If the annotation lacks geometry.
        """
        class_name = self.encode_class(sound_event)

        if class_name in self._roi_mapper_overrides:
            return self._roi_mapper_overrides[class_name].encode(
                sound_event.sound_event
            )

        return self._roi_mapper.encode(sound_event.sound_event)

    def decode_roi(
        self,
        position: Position,
        size: Size,
        class_name: Optional[str] = None,
    ) -> data.Geometry:
        """Recover an approximate geometric ROI from a position and dimensions.

        Delegates to the internal ROI mapper's `recover_roi` method, which
        un-scales the dimensions and reconstructs the geometry (typically a
        `BoundingBox`).

        Parameters
        ----------
        pos : Tuple[float, float]
            The reference position `(time, frequency)`.
        dims : np.ndarray
            NumPy array with size dimensions (e.g., from model prediction),
            matching the order in `self.dimension_names`.

        Returns
        -------
        data.Geometry
            The reconstructed geometry (typically `BoundingBox`).
        """
        if class_name in self._roi_mapper_overrides:
            return self._roi_mapper_overrides[class_name].decode(
                position,
                size,
            )

        return self._roi_mapper.decode(position, size)


DEFAULT_CLASSES = [
    TargetClass(
        tags=[TagInfo(value="Myotis mystacinus")],
        name="myomys",
    ),
    TargetClass(
        tags=[TagInfo(value="Myotis alcathoe")],
        name="myoalc",
    ),
    TargetClass(
        tags=[TagInfo(value="Eptesicus serotinus")],
        name="eptser",
    ),
    TargetClass(
        tags=[TagInfo(value="Pipistrellus nathusii")],
        name="pipnat",
    ),
    TargetClass(
        tags=[TagInfo(value="Barbastellus barbastellus")],
        name="barbar",
    ),
    TargetClass(
        tags=[TagInfo(value="Myotis nattereri")],
        name="myonat",
    ),
    TargetClass(
        tags=[TagInfo(value="Myotis daubentonii")],
        name="myodau",
    ),
    TargetClass(
        tags=[TagInfo(value="Myotis brandtii")],
        name="myobra",
    ),
    TargetClass(
        tags=[TagInfo(value="Pipistrellus pipistrellus")],
        name="pippip",
    ),
    TargetClass(
        tags=[TagInfo(value="Myotis bechsteinii")],
        name="myobec",
    ),
    TargetClass(
        tags=[TagInfo(value="Pipistrellus pygmaeus")],
        name="pippyg",
    ),
    TargetClass(
        tags=[TagInfo(value="Rhinolophus hipposideros")],
        name="rhihip",
    ),
    TargetClass(
        tags=[TagInfo(value="Nyctalus leisleri")],
        name="nyclei",
        roi=AnchorBBoxMapperConfig(anchor="top-left"),
    ),
    TargetClass(
        tags=[TagInfo(value="Rhinolophus ferrumequinum")],
        name="rhifer",
        roi=AnchorBBoxMapperConfig(anchor="top-left"),
    ),
    TargetClass(
        tags=[TagInfo(value="Plecotus auritus")],
        name="pleaur",
    ),
    TargetClass(
        tags=[TagInfo(value="Nyctalus noctula")],
        name="nycnoc",
    ),
    TargetClass(
        tags=[TagInfo(value="Plecotus austriacus")],
        name="pleaus",
    ),
]


DEFAULT_CLASSES_CONFIG: ClassesConfig = ClassesConfig(
    classes=DEFAULT_CLASSES,
    generic_class=[TagInfo(value="Bat")],
)


DEFAULT_TARGET_CONFIG: TargetConfig = TargetConfig(
    filtering=FilterConfig(
        rules=[
            FilterRule(
                match_type="all",
                tags=[TagInfo(key="event", value="Echolocation")],
            ),
            FilterRule(
                match_type="exclude",
                tags=[
                    TagInfo(key="event", value="Feeding"),
                    TagInfo(key="event", value="Unknown"),
                    TagInfo(key="event", value="Not Bat"),
                ],
            ),
        ]
    ),
    classes=DEFAULT_CLASSES_CONFIG,
    roi=AnchorBBoxMapperConfig(),
)


def build_targets(
    config: Optional[TargetConfig] = None,
    derivation_registry: DerivationRegistry = default_derivation_registry,
) -> Targets:
    """Build a Targets object from a loaded TargetConfig.

    This factory function takes the unified `TargetConfig` and constructs all
    necessary functional components (filter, transform, encoder,
    decoder, ROI mapper) by calling their respective builder functions. It also
    extracts metadata (class names, generic tags, dimension names) to create
    and return a fully initialized `Targets` instance, ready to process
    annotations.

    Parameters
    ----------
    config : TargetConfig
        The loaded and validated unified target configuration object.
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
    config = config or DEFAULT_TARGET_CONFIG
    logger.opt(lazy=True).debug(
        "Building targets with config: \n{}",
        lambda: config.to_yaml_string(),
    )

    filter_fn = (
        build_sound_event_filter(config.filtering)
        if config.filtering
        else None
    )
    encode_fn = build_sound_event_encoder(config.classes)
    decode_fn = build_sound_event_decoder(config.classes)
    transform_fn = (
        build_transformation_from_config(
            config.transforms,
            derivation_registry=derivation_registry,
        )
        if config.transforms
        else None
    )
    roi_mapper = build_roi_mapper(config.roi)
    class_names = get_class_names_from_config(config.classes)
    generic_class_tags = build_generic_class_tags(config.classes)
    roi_overrides = {
        class_config.name: build_roi_mapper(class_config.roi)
        for class_config in config.classes.classes
        if class_config.roi is not None
    }

    return Targets(
        filter_fn=filter_fn,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        class_names=class_names,
        roi_mapper=roi_mapper,
        generic_class_tags=generic_class_tags,
        transform_fn=transform_fn,
        roi_mapper_overrides=roi_overrides,
    )


def load_targets(
    config_path: data.PathLike,
    field: Optional[str] = None,
    derivation_registry: DerivationRegistry = default_derivation_registry,
) -> Targets:
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
    return build_targets(config, derivation_registry=derivation_registry)


def iterate_encoded_sound_events(
    sound_events: Iterable[data.SoundEventAnnotation],
    targets: TargetProtocol,
) -> Iterable[Tuple[Optional[str], Position, Size]]:
    for sound_event in sound_events:
        if not targets.filter(sound_event):
            continue

        geometry = sound_event.sound_event.geometry

        if geometry is None:
            continue

        sound_event = targets.transform(sound_event)

        class_name = targets.encode_class(sound_event)
        position, size = targets.encode_roi(sound_event)

        yield class_name, position, size
