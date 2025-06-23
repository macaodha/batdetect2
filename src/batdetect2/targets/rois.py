"""Handles mapping between geometric ROIs and target representations.

This module defines a standardized interface (`ROITargetMapper`) for converting
a sound event's Region of Interest (ROI) into a target representation suitable
for machine learning models, and for decoding model outputs back into geometric
ROIs.

The core operations are:
1.  **Encoding**: A `soundevent.data.SoundEvent` is mapped to a reference
    `Position` (time, frequency) and a `Size` array. The method for
    determining the position and size varies by the mapper implementation
    (e.g., using a bounding box anchor or the point of peak energy).
2.  **Decoding**: A `Position` and `Size` array are mapped back to an
    approximate `soundevent.data.Geometry` (typically a `BoundingBox`).

This logic is encapsulated within specific mapper classes. Configuration for
each mapper (e.g., anchor point, scaling factors) is managed by a corresponding
Pydantic config object. The `ROIMapperConfig` type allows for flexibly
selecting and configuring the desired mapper. This module separates the
*geometric* aspect of target definition from *semantic* classification.
"""

from typing import Annotated, List, Literal, Optional, Protocol, Tuple, Union

import numpy as np
from pydantic import Field
from soundevent import data

from batdetect2.configs import BaseConfig
from batdetect2.preprocess import PreprocessingConfig, build_preprocessor
from batdetect2.preprocess.types import PreprocessorProtocol
from batdetect2.targets.types import Position, Size

__all__ = [
    "Anchor",
    "AnchorBBoxMapper",
    "AnchorBBoxMapperConfig",
    "DEFAULT_ANCHOR",
    "DEFAULT_FREQUENCY_SCALE",
    "DEFAULT_TIME_SCALE",
    "PeakEnergyBBoxMapper",
    "PeakEnergyBBoxMapperConfig",
    "ROIMapperConfig",
    "ROITargetMapper",
    "SIZE_HEIGHT",
    "SIZE_ORDER",
    "SIZE_WIDTH",
    "build_roi_mapper",
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


class ROITargetMapper(Protocol):
    """Protocol defining the interface for ROI-to-target mapping.

    Specifies the `encode` and `decode` methods required for converting a
    `soundevent.data.SoundEvent` into a target representation (a reference
    position and a size vector) and for recovering an approximate ROI from that
    representation.

    Attributes
    ----------
    dimension_names : List[str]
        A list containing the names of the dimensions in the `Size` array
        returned by `encode` and expected by `decode`.
    """

    dimension_names: List[str]

    def encode(self, sound_event: data.SoundEvent) -> tuple[Position, Size]:
        """Encode a SoundEvent's geometry into a position and size.

        Parameters
        ----------
        sound_event : data.SoundEvent
            The input sound event, which must have a geometry attribute.

        Returns
        -------
        Tuple[Position, Size]
            A tuple containing:
            - The reference position as (time, frequency) coordinates.
            - A NumPy array with the calculated size dimensions.

        Raises
        ------
        ValueError
            If the sound event does not have a geometry.
        """
        ...

    def decode(self, position: Position, size: Size) -> data.Geometry:
        """Decode a position and size back into a geometric ROI.

        Performs the inverse mapping: takes a reference position and size
        dimensions and reconstructs a geometric representation.

        Parameters
        ----------
        position : Position
            The reference position (time, frequency).
        size : Size
            NumPy array containing the size dimensions, matching the order
            and meaning specified by `dimension_names`.

        Returns
        -------
        soundevent.data.Geometry
            The reconstructed geometry, typically a `BoundingBox`.

        Raises
        ------
        ValueError
            If the `size` array has an unexpected shape or if reconstruction
            fails.
        """
        ...


class AnchorBBoxMapperConfig(BaseConfig):
    """Configuration for `AnchorBBoxMapper`.

    Defines parameters for converting ROIs into targets using a fixed anchor
    point on the bounding box.

    Attributes
    ----------
    name : Literal["anchor_bbox"]
        The unique identifier for this mapper type.
    anchor : Anchor
        Specifies the anchor point within the bounding box to use as the
        target's reference position (e.g., "center", "bottom-left").
    time_scale : float
        Scaling factor applied to the time duration (width) of the ROI.
    frequency_scale : float
        Scaling factor applied to the frequency bandwidth (height) of the ROI.
    """

    name: Literal["anchor_bbox"] = "anchor_bbox"
    anchor: Anchor = DEFAULT_ANCHOR
    time_scale: float = DEFAULT_TIME_SCALE
    frequency_scale: float = DEFAULT_FREQUENCY_SCALE


class AnchorBBoxMapper(ROITargetMapper):
    """Maps ROIs using a bounding box anchor point and width/height.

    This class implements the `ROITargetMapper` protocol for `BoundingBox`
    geometries.

    **Encoding**: The `position` is a fixed anchor point on the bounding box
    (e.g., "bottom-left"). The `size` is a 2-element array containing the
    scaled width and height of the box.

    **Decoding**: Reconstructs a `BoundingBox` from an anchor point and
    scaled width/height.

    Attributes
    ----------
    dimension_names : List[str]
        The output dimension names: `['width', 'height']`.
    anchor : Anchor
        The configured anchor point type (e.g., "center", "bottom-left").
    time_scale : float
        The scaling factor for the time dimension (width).
    frequency_scale : float
        The scaling factor for the frequency dimension (height).
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
        anchor : Anchor
            Reference point type within the bounding box.
        time_scale : float
            Scaling factor for time duration (width).
        frequency_scale : float
            Scaling factor for frequency bandwidth (height).
        """
        self.anchor: Anchor = anchor
        self.time_scale = time_scale
        self.frequency_scale = frequency_scale

    def encode(self, sound_event: data.SoundEvent) -> Tuple[Position, Size]:
        """Encode a SoundEvent into an anchor position and scaled box size.

        The position is determined by the configured anchor on the sound
        event's bounding box. The size is the scaled width and height.

        Parameters
        ----------
        sound_event : data.SoundEvent
            The input sound event with a geometry.

        Returns
        -------
        Tuple[Position, Size]
            A tuple of (anchor_position, [scaled_width, scaled_height]).
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
        """Recover a BoundingBox from an anchor position and scaled size.

        Un-scales the input dimensions and reconstructs a
        `soundevent.data.BoundingBox` relative to the given anchor position.

        Parameters
        ----------
        position : Position
            Reference anchor position (time, frequency).
        size : Size
            NumPy array containing the scaled [width, height].

        Returns
        -------
        data.BoundingBox
            The reconstructed bounding box.

        Raises
        ------
        ValueError
            If `size` does not have the expected shape (length 2).
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
    """Configuration for `PeakEnergyBBoxMapper`.

    Attributes
    ----------
    name : Literal["peak_energy_bbox"]
        The unique identifier for this mapper type.
    preprocessing : PreprocessingConfig
        Configuration for the spectrogram preprocessor needed to find the
        peak energy.
    loading_buffer : float
        Seconds to add to each side of the ROI when loading audio to ensure
        the peak is captured accurately, avoiding boundary effects.
    time_scale : float
        Scaling factor applied to the time dimensions.
    frequency_scale : float
        Scaling factor applied to the frequency dimensions.
    """

    name: Literal["peak_energy_bbox"] = "peak_energy_bbox"
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )
    loading_buffer: float = 0.01
    time_scale: float = DEFAULT_TIME_SCALE
    frequency_scale: float = DEFAULT_FREQUENCY_SCALE


class PeakEnergyBBoxMapper(ROITargetMapper):
    """Maps ROIs using the peak energy point and distances to edges.

    This class implements the `ROITargetMapper` protocol.

    **Encoding**: The `position` is the (time, frequency) coordinate of the
    point with the highest energy within the sound event's bounding box. The
    `size` is a 4-element array representing the scaled distances from this
    peak energy point to the left, bottom, right, and top edges of the box.

    **Decoding**: Reconstructs a `BoundingBox` by adding/subtracting the
    un-scaled distances from the peak energy point.

    Attributes
    ----------
    dimension_names : List[str]
        The output dimension names: `['left', 'bottom', 'right', 'top']`.
    preprocessor : PreprocessorProtocol
        The spectrogram preprocessor instance.
    time_scale : float
        The scaling factor for time-based distances.
    frequency_scale : float
        The scaling factor for frequency-based distances.
    loading_buffer : float
        The buffer used for loading audio around the ROI.
    """

    dimension_names = ["left", "bottom", "right", "top"]

    def __init__(
        self,
        preprocessor: PreprocessorProtocol,
        time_scale: float = DEFAULT_TIME_SCALE,
        frequency_scale: float = DEFAULT_FREQUENCY_SCALE,
        loading_buffer: float = 0.01,
    ):
        """Initialize the PeakEnergyBBoxMapper.

        Parameters
        ----------
        preprocessor : PreprocessorProtocol
            An initialized preprocessor for generating spectrograms.
        time_scale : float
            Scaling factor for time dimensions (left, right distances).
        frequency_scale : float
            Scaling factor for frequency dimensions (bottom, top distances).
        loading_buffer : float
            Buffer in seconds to add when loading audio clips.
        """
        self.preprocessor = preprocessor
        self.time_scale = time_scale
        self.frequency_scale = frequency_scale
        self.loading_buffer = loading_buffer

    def encode(
        self,
        sound_event: data.SoundEvent,
    ) -> tuple[Position, Size]:
        """Encode a SoundEvent into a peak energy position and edge distances.

        Finds the peak energy coordinates within the event's bounding box
        and calculates the scaled distances from this point to the box edges.

        Parameters
        ----------
        sound_event : data.SoundEvent
            The input sound event with a geometry and associated recording.

        Returns
        -------
        Tuple[Position, Size]
            A tuple of (peak_position, [l, b, r, t] distances).
        """
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
        """Recover a BoundingBox from a peak position and edge distances.

        Parameters
        ----------
        position : Position
            The reference peak energy position (time, frequency).
        size : Size
            NumPy array with scaled distances [left, bottom, right, top].

        Returns
        -------
        data.BoundingBox
            The reconstructed bounding box.
        """
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
    Union[AnchorBBoxMapperConfig, PeakEnergyBBoxMapperConfig],
    Field(discriminator="name"),
]
"""A discriminated union of all supported ROI mapper configurations.

This type allows for selecting and configuring different `ROITargetMapper`
implementations by using the `name` field as a discriminator.
"""


def build_roi_mapper(
    config: Optional[ROIMapperConfig] = None,
) -> ROITargetMapper:
    """Factory function to create an ROITargetMapper from a config object.

    Parameters
    ----------
    config : ROIMapperConfig
        A configuration object specifying the mapper type and its parameters.

    Returns
    -------
    ROITargetMapper
        An initialized ROI mapper instance.

    Raises
    ------
    NotImplementedError
        If the `name` in the config does not correspond to a known mapper.
    """
    config = config or AnchorBBoxMapperConfig()

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
        f"No ROI mapper of name '{config.name}' is implemented"
    )


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
    anchor : Anchor
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
    """Find the coordinates of the highest energy point in a spectrogram.

    Generates a spectrogram for a specified time-frequency region of a
    recording and returns the (time, frequency) coordinates of the pixel with
    the maximum value.

    Parameters
    ----------
    recording : data.Recording
        The recording to analyze.
    preprocessor : PreprocessorProtocol
        The processor to convert audio to a spectrogram.
    start_time : float, default=0
        The start time of the region of interest.
    end_time : float, optional
        The end time of the region of interest. Defaults to recording duration.
    low_freq : float, default=0
        The low frequency of the region of interest.
    high_freq : float, optional
        The high frequency of the region of interest. Defaults to Nyquist.
    loading_buffer : float, default=0.05
        Buffer in seconds to add around the time range when loading the clip
        to mitigate border effects from transformations like STFT.

    Returns
    -------
    Position
        A (time, frequency) tuple for the peak energy location.
    """
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
