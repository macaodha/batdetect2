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

from typing import List, Literal, Optional, Protocol, Tuple

import numpy as np
from soundevent import data

from batdetect2.configs import BaseConfig, load_config

Positions = Literal[
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

__all__ = [
    "ROITargetMapper",
    "ROIConfig",
    "BBoxEncoder",
    "build_roi_mapper",
    "load_roi_mapper",
    "DEFAULT_POSITION",
    "SIZE_WIDTH",
    "SIZE_HEIGHT",
    "SIZE_ORDER",
    "DEFAULT_TIME_SCALE",
    "DEFAULT_FREQUENCY_SCALE",
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


DEFAULT_POSITION = "bottom-left"
"""Default reference position within the geometry ('bottom-left' corner)."""


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

    def get_roi_position(self, geom: data.Geometry) -> tuple[float, float]:
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

    def get_roi_size(self, geom: data.Geometry) -> np.ndarray:
        """Calculate the scaled target dimensions from a geometry.

        Computes the relevant size measures.

        Parameters
        ----------
        geom : soundevent.data.Geometry
            The input geometry.

        Returns
        -------
        np.ndarray
            A NumPy array containing the scaled dimensions corresponding to
            `dimension_names`. For bounding boxes, typically contains
            `[scaled_width, scaled_height]`.

        Raises
        ------
        TypeError, ValueError
            If the size cannot be computed for the given geometry type.
        """
        ...

    def recover_roi(
        self, pos: tuple[float, float], dims: np.ndarray
    ) -> data.Geometry:
        """Recover an approximate ROI from a position and target dimensions.

        Performs the inverse mapping: takes a reference position and the
        predicted dimensions and reconstructs a geometric representation.

        Parameters
        ----------
        pos : Tuple[float, float]
            The reference position (time, frequency).
        dims : np.ndarray
            The NumPy array containing the dimensions, matching the order
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


class ROIConfig(BaseConfig):
    """Configuration for mapping Regions of Interest (ROIs).

    Defines parameters controlling how geometric ROIs are converted into
    target representations (reference points and scaled sizes).

    Attributes
    ----------
    position : Positions, default="bottom-left"
        Specifies the reference point within the geometry (e.g., bounding box)
        to use as the target location (e.g., "center", "bottom-left").
        See `soundevent.geometry.operations.Positions`.
    time_scale : float, default=1000.0
        Scaling factor applied to the time duration (width) of the ROI
        when calculating the target size representation. Must match model
        expectations.
    frequency_scale : float, default=1/859.375
        Scaling factor applied to the frequency bandwidth (height) of the ROI
        when calculating the target size representation. Must match model
        expectations.
    """

    position: Positions = DEFAULT_POSITION
    time_scale: float = DEFAULT_TIME_SCALE
    frequency_scale: float = DEFAULT_FREQUENCY_SCALE


class BBoxEncoder(ROITargetMapper):
    """Concrete implementation of `ROITargetMapper` focused on Bounding Boxes.

    This class implements the ROI mapping protocol primarily for
    `soundevent.data.BoundingBox` geometry. It extracts reference points,
    calculates scaled width/height, and recovers bounding boxes based on
    configured position and scaling factors.

    Attributes
    ----------
    dimension_names : List[str]
        Specifies the output dimension names as ['width', 'height'].
    position : Positions
        The configured reference point type (e.g., "center", "bottom-left").
    time_scale : float
        The configured scaling factor for the time dimension (width).
    frequency_scale : float
        The configured scaling factor for the frequency dimension (height).
    """

    dimension_names = [SIZE_WIDTH, SIZE_HEIGHT]

    def __init__(
        self,
        position: Positions = DEFAULT_POSITION,
        time_scale: float = DEFAULT_TIME_SCALE,
        frequency_scale: float = DEFAULT_FREQUENCY_SCALE,
    ):
        """Initialize the BBoxEncoder.

        Parameters
        ----------
        position : Positions, default="bottom-left"
            Reference point type within the bounding box.
        time_scale : float, default=1000.0
            Scaling factor for time duration (width).
        frequency_scale : float, default=1/859.375
            Scaling factor for frequency bandwidth (height).
        """
        self.position: Positions = position
        self.time_scale = time_scale
        self.frequency_scale = frequency_scale

    def get_roi_position(self, geom: data.Geometry) -> Tuple[float, float]:
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

        return geometry.get_geometry_point(geom, position=self.position)

    def get_roi_size(self, geom: data.Geometry) -> np.ndarray:
        """Calculate the scaled [width, height] from the geometry's bounds.

        Computes the bounding box, extracts duration and bandwidth, and applies
        the configured `time_scale` and `frequency_scale`.

        Parameters
        ----------
        geom : soundevent.data.Geometry
            Input geometry.

        Returns
        -------
        np.ndarray
            A 1D NumPy array: `[scaled_width, scaled_height]`.
        """
        from soundevent import geometry

        start_time, low_freq, end_time, high_freq = geometry.compute_bounds(
            geom
        )
        return np.array(
            [
                (end_time - start_time) * self.time_scale,
                (high_freq - low_freq) * self.frequency_scale,
            ]
        )

    def recover_roi(
        self,
        pos: tuple[float, float],
        dims: np.ndarray,
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
        if dims.ndim != 1 or dims.shape[0] != 2:
            raise ValueError(
                "Dimension array does not have the expected shape. "
                f"({dims.shape = }) != ([2])"
            )

        width, height = dims
        return _build_bounding_box(
            pos,
            duration=float(width) / self.time_scale,
            bandwidth=float(height) / self.frequency_scale,
            position=self.position,
        )


def build_roi_mapper(config: ROIConfig) -> ROITargetMapper:
    """Factory function to create an ROITargetMapper from configuration.

    Currently creates a `BBoxEncoder` instance based on the provided
    `ROIConfig`.

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
    return BBoxEncoder(
        position=config.position,
        time_scale=config.time_scale,
        frequency_scale=config.frequency_scale,
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
    config = load_config(path=path, schema=ROIConfig, field=field)
    return build_roi_mapper(config)


VALID_POSITIONS = [
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
    position: Positions = DEFAULT_POSITION,
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
    position : Positions, default="bottom-left"
        Specifies which part of the bounding box the input `pos` corresponds to.

    Returns
    -------
    data.BoundingBox
        The constructed bounding box object.

    Raises
    ------
    ValueError
        If `position` is not a recognized value or format.
    """
    time, freq = map(float, pos)
    duration = max(0, duration)
    bandwidth = max(0, bandwidth)
    if position in ["center", "centroid", "point_on_surface"]:
        return data.BoundingBox(
            coordinates=[
                max(time - duration / 2, 0),
                max(freq - bandwidth / 2, 0),
                max(time + duration / 2, 0),
                max(freq + bandwidth / 2, 0),
            ]
        )

    if position not in VALID_POSITIONS:
        raise ValueError(
            f"Invalid position: {position}. "
            f"Valid options are: {VALID_POSITIONS}"
        )

    y, x = position.split("-")

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
