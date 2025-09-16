from typing import Iterable, List, Optional, Tuple

from loguru import logger
from soundevent import data

from batdetect2.data.conditions import build_sound_event_condition
from batdetect2.targets.classes import (
    DEFAULT_CLASSES,
    DEFAULT_DETECTION_CLASS,
    build_sound_event_decoder,
    build_sound_event_encoder,
    get_class_names_from_config,
)
from batdetect2.targets.config import TargetConfig, load_target_config
from batdetect2.targets.rois import (
    AnchorBBoxMapperConfig,
    build_roi_mapper,
)
from batdetect2.typing.targets import Position, Size, TargetProtocol


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
    detection_class_tags: List[data.Tag]
    dimension_names: List[str]
    detection_class_name: str

    def __init__(self, config: TargetConfig):
        """Initialize the Targets object."""
        self.config = config

        self._filter_fn = build_sound_event_condition(
            config.detection_target.match_if
        )
        self._encode_fn = build_sound_event_encoder(
            config.classification_targets
        )
        self._decode_fn = build_sound_event_decoder(
            config.classification_targets
        )

        self._roi_mapper = build_roi_mapper(config.roi)

        self.dimension_names = self._roi_mapper.dimension_names

        self.class_names = get_class_names_from_config(
            config.classification_targets
        )

        self.detection_class_name = config.detection_target.name
        self.detection_class_tags = config.detection_target.assign_tags

        self._roi_mapper_overrides = {
            class_config.name: build_roi_mapper(class_config.roi)
            for class_config in config.classification_targets
            if class_config.roi is not None
        }

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


DEFAULT_TARGET_CONFIG: TargetConfig = TargetConfig(
    classification_targets=DEFAULT_CLASSES,
    detection_target=DEFAULT_DETECTION_CLASS,
    roi=AnchorBBoxMapperConfig(),
)


def build_targets(config: Optional[TargetConfig] = None) -> Targets:
    """Build a Targets object from a loaded TargetConfig.

    Parameters
    ----------
    config : TargetConfig
        The loaded and validated unified target configuration object.

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

    return Targets(config=config)


def load_targets(
    config_path: data.PathLike,
    field: Optional[str] = None,
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
    return build_targets(config)


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

        class_name = targets.encode_class(sound_event)
        position, size = targets.encode_roi(sound_event)

        yield class_name, position, size
