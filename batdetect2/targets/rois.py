"""Handles mapping between geometric ROIs and target representations.

This module defines the interface and provides implementation for converting
a sound event's Region of Interest (ROI), typically represented by a
`soundevent.data.Geometry` object like a `BoundingBox`, into a format
suitable for use as a machine learning target. This usually involves:

1.  Extracting a single reference point (time, frequency) from the geometry.
2.  Calculating relevant size dimensions (e.g., duration/width,
    bandwidth/height) and applying scaling factors.

It also provides the inverse operation: recovering an approximate geometric ROI
(like a `BoundingBox`) from a predicted reference point and predicted size
dimensions.

This logic is encapsulated within components adhering to the `ROITargetMapper`
protocol. Configuration for this mapping (e.g., which reference point to use,
scaling factors) is managed by the `ROIConfig`. This module separates the
*geometric* aspect of target definition from the *semantic* classification
handled in `batdetect2.targets.classes`.
"""

from typing import Annotated, List, Literal, Optional, Protocol, Tuple, Union

import numpy as np
from pydantic import Field
from soundevent import data

from batdetect2.configs import BaseConfig, load_config
from batdetect2.preprocess import PreprocessingConfig, build_preprocessor
from batdetect2.preprocess.types import PreprocessorProtocol

__all__ = [
    "ROITargetMapper",
    "BBoxAnchorMapperConfig",
    "AnchorBBoxMapper",
    "build_roi_mapper",
    "load_roi_mapper",
    "DEFAULT_ANCHOR",
    "SIZE_WIDTH",
    "SIZE_HEIGHT",
    "SIZE_ORDER",
    "DEFAULT_TIME_SCALE",
    "DEFAULT_FREQUENCY_SCALE",
]

Anchor = Literal[
    "bottom-left",
    "bottom-right",
    "top-left",
    "top-right",
    "center-left",
    "center-right",
    "top-center",
    "bottom-center",
    "center",
    "centroid",
    "point_on_surface",
]

SIZE_WIDTH = "width"
"""Standard name for the width/time dimension component ('width')."""

SIZE_HEIGHT = "height"
"""Standard name for the height/frequency dimension component ('height')."""

SIZE_ORDER = (SIZE_WIDTH, SIZE_HEIGHT)
"""Standard order of dimensions for size arrays ([width, height])."""

DEFAULT_TIME_SCALE = 1000.0
"""Default scaling factor for time duration."""

DEFAULT_FREQUENCY_SCALE = 1 / 859.375
"""Default scaling factor for frequency bandwidth."""


DEFAULT_ANCHOR = "bottom-left"
"""Default reference position within the geometry ('bottom-left' corner)."""


Position = tuple[float, float]

Size = np.ndarray


class ROITargetMapper(Protocol):
    """Protocol defining the interface for ROI-to-target mapping.

    Specifies the methods required for converting a geometric region of interest
    (`soundevent.data.Geometry`) into a target representation (reference point
    and scaled dimensions) and for recovering an approximate ROI from that
    representation.

    Attributes
    ----------
    dimension_names : List[str]
        A list containing the names of the dimensions returned by
        `get_roi_size` and expected by `recover_roi`
        (e.g., ['width', 'height']).
    """

    dimension_names: List[str]

    def encode(self, sound_event: data.SoundEvent) -> tuple[Position, Size]:
        """Extract the reference position from a geometry.

        Parameters
        ----------
        geom : soundevent.data.Geometry
            The input geometry (e.g., BoundingBox, Polygon).

        Returns
        -------
        Tuple[float, float]
            The calculated reference position as (time, frequency) coordinates,
            based on the implementing class's configuration (e.g., "center",
            "bottom-left").

        Raises
        ------
        ValueError
            If the position cannot be calculated for the given geometry type
            or configured reference point.
        """
        ...

    def decode(self, position: Position, size: Size) -> data.Geometry:
        """Recover an approximate ROI from a position and target dimensions.

        Performs the inverse mapping: takes a reference position and the
        predicted dimensions and reconstructs a geometric representation.

        Parameters
        ----------
        position : Tuple[float, float]
            The reference position (time, frequency).
        size : np.ndarray
            NumPy array containing the dimensions, matching the order
            specified by `dimension_names`.

        Returns
        -------
        soundevent.data.Geometry
            The reconstructed geometry.

        Raises
        ------
        ValueError
            If the number of provided dimensions `dims` does not match
            `dimension_names` or if reconstruction fails.
        """
        ...


class BBoxAnchorMapperConfig(BaseConfig):
    """Configuration for mapping Regions of Interest (ROIs).

    Defines parameters controlling how geometric ROIs are converted into
    target representations (reference points and scaled sizes).

    Attributes
    ----------
    anchor : Anchor, default="bottom-left"
        Specifies the reference point within the geometry (e.g., bounding box)
        to use as the target location (e.g., "center", "bottom-left").
    time_scale : float, default=1000.0
        Scaling factor applied to the time duration (width) of the ROI
        when calculating the target size representation. Must match model
        expectations.
    frequency_scale : float, default=1/859.375
        Scaling factor applied to the frequency bandwidth (height) of the ROI
        when calculating the target size representation. Must match model
        expectations.
    """

    name: Literal["anchor_bbox"] = "anchor_bbox"
    anchor: Anchor = DEFAULT_ANCHOR
    time_scale: float = DEFAULT_TIME_SCALE
    frequency_scale: float = DEFAULT_FREQUENCY_SCALE


class AnchorBBoxMapper(ROITargetMapper):
    """Concrete implementation of `ROITargetMapper` focused on Bounding Boxes.

    This class implements the ROI mapping protocol primarily for
    `soundevent.data.BoundingBox` geometry. It extracts reference points,
    calculates scaled width/height, and recovers bounding boxes based on
    configured position and scaling factors.

    Attributes
    ----------
    dimension_names : List[str]
        Specifies the output dimension names as ['width', 'height'].
    anchor : Anchor
        The configured reference point type (e.g., "center", "bottom-left").
    time_scale : float
        The configured scaling factor for the time dimension (width).
    frequency_scale : float
        The configured scaling factor for the frequency dimension (height).
    """

    dimension_names = [SIZE_WIDTH, SIZE_HEIGHT]

    def __init__(
        self,
        anchor: Anchor = DEFAULT_ANCHOR,
        time_scale: float = DEFAULT_TIME_SCALE,
        frequency_scale: float = DEFAULT_FREQUENCY_SCALE,
    ):
        """Initialize the BBoxEncoder.

        Parameters
        ----------
        anchor : Anchor, default="bottom-left"
            Reference point type within the bounding box.
        time_scale : float, default=1000.0
            Scaling factor for time duration (width).
        frequency_scale : float, default=1/859.375
            Scaling factor for frequency bandwidth (height).
        """
        self.anchor: Anchor = anchor
        self.time_scale = time_scale
        self.frequency_scale = frequency_scale

    def encode(self, sound_event: data.SoundEvent) -> Tuple[Position, Size]:
        """Extract the configured reference position from the geometry.

        Uses `soundevent.geometry.get_geometry_point`.

        Parameters
        ----------
        geom : soundevent.data.Geometry
            Input geometry (e.g., BoundingBox).

        Returns
        -------
        Tuple[float, float]
            Reference position (time, frequency).
        """
        from soundevent import geometry

        geom = sound_event.geometry

        if geom is None:
            raise ValueError(
                "Cannot encode the geometry of a sound event without geometry."
                f" Sound event: {sound_event}"
            )

        position = geometry.get_geometry_point(geom, position=self.anchor)

        start_time, low_freq, end_time, high_freq = geometry.compute_bounds(
            geom
        )

        size = np.array(
            [
                (end_time - start_time) * self.time_scale,
                (high_freq - low_freq) * self.frequency_scale,
            ]
        )

        return position, size

    def decode(
        self,
        position: Position,
        size: Size,
    ) -> data.Geometry:
        """Recover a BoundingBox from a position and scaled dimensions.

        Un-scales the input dimensions using the configured factors and
        reconstructs a `soundevent.data.BoundingBox` centered or anchored at
        the given reference `pos` according to the configured `position` type.

        Parameters
        ----------
        pos : Tuple[float, float]
            Reference position (time, frequency).
        dims : np.ndarray
            NumPy array containing the *scaled* dimensions, expected order is
            [scaled_width, scaled_height].

        Returns
        -------
        soundevent.data.BoundingBox
            The reconstructed bounding box.

        Raises
        ------
        ValueError
            If `dims` does not have the expected shape (length 2).
        """

        if size.ndim != 1 or size.shape[0] != 2:
            raise ValueError(
                "Dimension array does not have the expected shape. "
                f"({size.shape = }) != ([2])"
            )

        width, height = size
        return _build_bounding_box(
            position,
            duration=float(width) / self.time_scale,
            bandwidth=float(height) / self.frequency_scale,
            anchor=self.anchor,
        )


class PeakEnergyBBoxMapperConfig(BaseConfig):
    name: Literal["peak_energy_bbox"]
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )
    loading_buffer: float = 0.01
    time_scale: float = DEFAULT_TIME_SCALE
    frequency_scale: float = DEFAULT_FREQUENCY_SCALE


class PeakEnergyBBoxMapper(ROITargetMapper):
    """
    Encodes the ROI using the location of the peak energy within the bounding box
    as the 'position' and the distances from that point to the box edges as the 'size'.
    """

    dimension_names = ["left", "bottom", "right", "top"]

    def __init__(
        self,
        preprocessor: PreprocessorProtocol,
        time_scale: float = DEFAULT_TIME_SCALE,
        frequency_scale: float = DEFAULT_FREQUENCY_SCALE,
        loading_buffer: float = 0.01,
    ):
        self.preprocessor = preprocessor
        self.time_scale = time_scale
        self.frequency_scale = frequency_scale
        self.loading_buffer = loading_buffer

    def encode(
        self,
        sound_event: data.SoundEvent,
    ) -> tuple[Position, Size]:
        from soundevent import geometry

        geom = sound_event.geometry

        if geom is None:
            raise ValueError(
                "Cannot encode the geometry of a sound event without geometry."
                f" Sound event: {sound_event}"
            )

        start_time, low_freq, end_time, high_freq = geometry.compute_bounds(
            geom
        )

        time, freq = get_peak_energy_coordinates(
            recording=sound_event.recording,
            preprocessor=self.preprocessor,
            start_time=start_time,
            end_time=end_time,
            low_freq=low_freq,
            high_freq=high_freq,
            loading_buffer=self.loading_buffer,
        )

        size = np.array(
            [
                (time - start_time) * self.time_scale,
                (freq - low_freq) * self.frequency_scale,
                (end_time - time) * self.time_scale,
                (high_freq - freq) * self.frequency_scale,
            ]
        )

        return (time, freq), size

    def decode(self, position: Position, size: Size) -> data.Geometry:
        time, freq = position
        left, bottom, right, top = size

        return data.BoundingBox(
            coordinates=[
                time - max(0, float(left)) / self.time_scale,
                freq - max(0, float(bottom)) / self.frequency_scale,
                time + max(0, float(right)) / self.time_scale,
                freq + max(0, float(top)) / self.frequency_scale,
            ]
        )


ROIMapperConfig = Annotated[
    Union[BBoxAnchorMapperConfig, PeakEnergyBBoxMapperConfig],
    Field(discriminator="name"),
]


def build_roi_mapper(config: ROIMapperConfig) -> ROITargetMapper:
    """Factory function to create an ROITargetMapper from configuration.

    Parameters
    ----------
    config : ROIConfig
        Configuration object specifying ROI mapping parameters.

    Returns
    -------
    ROITargetMapper
        An initialized `BBoxEncoder` instance configured with the settings
        from `config`.
    """
    if config.name == "anchor_bbox":
        return AnchorBBoxMapper(
            anchor=config.anchor,
            time_scale=config.time_scale,
            frequency_scale=config.frequency_scale,
        )

    if config.name == "peak_energy_bbox":
        preprocessor = build_preprocessor(config.preprocessing)
        return PeakEnergyBBoxMapper(
            preprocessor=preprocessor,
            time_scale=config.time_scale,
            frequency_scale=config.frequency_scale,
            loading_buffer=config.loading_buffer,
        )

    raise NotImplementedError(
        f"No ROI mapper of name {config.name} is implemented"
    )


def load_roi_mapper(
    path: data.PathLike, field: Optional[str] = None
) -> ROITargetMapper:
    """Load ROI mapping configuration from a file and build the mapper.

    Convenience function that loads an `ROIConfig` from the specified file
    (and optional field) and then uses `build_roi_mapper` to create the
    corresponding `ROITargetMapper` instance.

    Parameters
    ----------
    path : PathLike
        Path to the configuration file (e.g., YAML).
    field : str, optional
        Dot-separated path to a nested section within the file containing the
        ROI configuration. If None, the entire file content is used.

    Returns
    -------
    ROITargetMapper
        An initialized ROI mapper instance based on the configuration file.

    Raises
    ------
    FileNotFoundError, yaml.YAMLError, pydantic.ValidationError, KeyError,
    TypeError
        If the configuration file cannot be found, parsed, validated, or if
        the specified `field` is invalid.
    """
    config = load_config(path=path, schema=BBoxAnchorMapperConfig, field=field)
    return build_roi_mapper(config)


VALID_ANCHORS = [
    "bottom-left",
    "bottom-right",
    "top-left",
    "top-right",
    "center-left",
    "center-right",
    "top-center",
    "bottom-center",
    "center",
    "centroid",
    "point_on_surface",
]


def _build_bounding_box(
    pos: tuple[float, float],
    duration: float,
    bandwidth: float,
    anchor: Anchor = DEFAULT_ANCHOR,
) -> data.BoundingBox:
    """Construct a BoundingBox from a reference point, size, and position type.

    Internal helper for `BBoxEncoder.recover_roi`. Calculates the box
    coordinates [start_time, low_freq, end_time, high_freq] based on where
    the input `pos` (time, freq) is located relative to the box (e.g.,
    center, corner).

    Parameters
    ----------
    pos : Tuple[float, float]
        Reference position (time, frequency).
    duration : float
        The required *unscaled* duration (width) of the bounding box.
    bandwidth : float
        The required *unscaled* frequency bandwidth (height) of the bounding
        box.
    anchor : Anchor, default="bottom-left"
        Specifies which part of the bounding box the input `pos` corresponds to.

    Returns
    -------
    data.BoundingBox
        The constructed bounding box object.

    Raises
    ------
    ValueError
        If `anchor` is not a recognized value or format.
    """
    time, freq = map(float, pos)
    duration = max(0, duration)
    bandwidth = max(0, bandwidth)
    if anchor in ["center", "centroid", "point_on_surface"]:
        return data.BoundingBox(
            coordinates=[
                max(time - duration / 2, 0),
                max(freq - bandwidth / 2, 0),
                max(time + duration / 2, 0),
                max(freq + bandwidth / 2, 0),
            ]
        )

    if anchor not in VALID_ANCHORS:
        raise ValueError(
            f"Invalid anchor: {anchor}. Valid options are: {VALID_ANCHORS}"
        )

    y, x = anchor.split("-")

    start_time = {
        "left": time,
        "center": time - duration / 2,
        "right": time - duration,
    }[x]

    low_freq = {
        "bottom": freq,
        "center": freq - bandwidth / 2,
        "top": freq - bandwidth,
    }[y]

    return data.BoundingBox(
        coordinates=[
            max(0, start_time),
            max(0, low_freq),
            max(0, start_time + duration),
            max(0, low_freq + bandwidth),
        ]
    )


def get_peak_energy_coordinates(
    recording: data.Recording,
    preprocessor: PreprocessorProtocol,
    start_time: float = 0,
    end_time: Optional[float] = None,
    low_freq: float = 0,
    high_freq: Optional[float] = None,
    loading_buffer: float = 0.05,
) -> Position:
    if end_time is None:
        end_time = recording.duration
    end_time = min(end_time, recording.duration)

    if high_freq is None:
        high_freq = recording.samplerate / 2

    clip_start = max(0, start_time - loading_buffer)
    clip_end = min(recording.duration, end_time + loading_buffer)

    clip = data.Clip(
        recording=recording,
        start_time=clip_start,
        end_time=clip_end,
    )

    spec = preprocessor.preprocess_clip(clip)
    low_freq = max(low_freq, preprocessor.min_freq)
    high_freq = min(high_freq, preprocessor.max_freq)
    selection = spec.sel(
        time=slice(start_time, end_time),
        frequency=slice(low_freq, high_freq),
    )

    index = selection.argmax(dim=["time", "frequency"])
    point = selection.isel(index)  # type: ignore
    peak_time: float = point.time.item()
    peak_freq: float = point.frequency.item()
    return peak_time, peak_freq
