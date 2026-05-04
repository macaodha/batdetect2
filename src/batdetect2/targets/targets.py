from typing import Iterable

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
from batdetect2.targets.config import TargetConfig
from batdetect2.targets.types import (
    Position,
    ROIMapperProtocol,
    Size,
    TargetProtocol,
)


class Targets(TargetProtocol):
    """Encapsulates the configured target class definition pipeline.

    This class implements the `TargetProtocol`, holding the configured
    functions for filtering, encoding (tags to class name), and decoding
    (class name to tags). Geometry ROI mapping is handled separately by
    ``ROIMapperProtocol``.

    Instances are typically created using the `build_targets` factory function
    or the `load_targets` convenience loader.

    Attributes
    ----------
    class_names
        An ordered list of the unique names of the specific target classes
        defined in the configuration.
    generic_class_tags
        A list of `soundevent.data.Tag` objects representing the configured
        generic class category (used when no specific class matches).
    """

    class_names: list[str]
    detection_class_tags: list[data.Tag]
    detection_class_name: str

    def __init__(self, config: TargetConfig):
        """Initialize the Targets object."""
        self.config = config

        self._filter_fn = build_sound_event_condition(
            self.config.detection_target.match_if
        )
        self._encode_fn = build_sound_event_encoder(
            self.config.classification_targets
        )
        self._decode_fn = build_sound_event_decoder(
            self.config.classification_targets
        )

        self.class_names = get_class_names_from_config(
            self.config.classification_targets
        )

        self.detection_class_name = self.config.detection_target.name
        self.detection_class_tags = self.config.detection_target.assign_tags

    @classmethod
    def from_config(cls, config: dict) -> "Targets":
        """Build a Targets object from a serialized config dictionary."""
        validated_config = TargetConfig.model_validate(config)
        return cls(config=validated_config)

    def get_config(self) -> dict:
        """Return the serialized target config used to build this object."""
        return self.config.model_dump(mode="json")

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
    ) -> str | None:
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

    def decode_class(self, class_label: str) -> list[data.Tag]:
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
        list[data.Tag]
            The list of tags corresponding to the input class name.
        """
        return self._decode_fn(class_label)


DEFAULT_TARGET_CONFIG: TargetConfig = TargetConfig(
    classification_targets=DEFAULT_CLASSES,
    detection_target=DEFAULT_DETECTION_CLASS,
)


def build_targets(config: TargetConfig | dict | None = None) -> Targets:
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

    if not isinstance(config, TargetConfig):
        config = TargetConfig.model_validate(config)

    logger.opt(lazy=True).debug(
        "Building targets with config: \n{}",
        lambda: config.to_yaml_string(),
    )

    return Targets(config=config)


def load_targets(
    config_path: data.PathLike,
    field: str | None = None,
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
        Errors raised during file loading or validation via
        ``TargetConfig.load``.
    KeyError, ImportError, AttributeError, TypeError
        Errors raised during the build process by `Targets.from_config`
        (e.g., missing keys in registries, failed imports).
    """
    config = TargetConfig.load(
        config_path,
        field=field,
    )
    return build_targets(config)


def iterate_encoded_sound_events(
    sound_events: Iterable[data.SoundEventAnnotation],
    targets: TargetProtocol,
    roi_mapper: ROIMapperProtocol,
) -> Iterable[tuple[str | None, Position, Size]]:
    for sound_event in sound_events:
        if not targets.filter(sound_event):
            continue

        geometry = sound_event.sound_event.geometry

        if geometry is None:
            continue

        class_name = targets.encode_class(sound_event)
        position, size = roi_mapper.encode(
            sound_event.sound_event,
            class_name=class_name,
        )

        yield class_name, position, size
