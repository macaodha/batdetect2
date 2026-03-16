from typing import Annotated, Literal

from pydantic import Field
from soundevent import data
from soundevent.geometry import (
    buffer_geometry,
    compute_bbox_iou,
    compute_geometric_iou,
    compute_temporal_closeness,
    compute_temporal_iou,
)

from batdetect2.core import (
    BaseConfig,
    ImportConfig,
    Registry,
    add_import_config,
)
from batdetect2.typing import AffinityFunction, Detection

affinity_functions: Registry[AffinityFunction, []] = Registry(
    "affinity_function"
)


@add_import_config(affinity_functions)
class AffinityFunctionImportConfig(ImportConfig):
    """Use any callable as an affinity function.

    Set ``name="import"`` and provide a ``target`` pointing to any
    callable to use it instead of a built-in option.
    """

    name: Literal["import"] = "import"


class TimeAffinityConfig(BaseConfig):
    name: Literal["time_affinity"] = "time_affinity"
    position: Literal["start", "end", "center"] | float = "start"
    max_distance: float = 0.01


class TimeAffinity(AffinityFunction):
    def __init__(
        self,
        max_distance: float = 0.01,
        position: Literal["start", "end", "center"] | float = "start",
    ):
        if position == "start":
            position = 0
        elif position == "end":
            position = 1
        elif position == "center":
            position = 0.5

        self.position = position
        self.max_distance = max_distance

    def __call__(
        self,
        detection: Detection,
        ground_truth: data.SoundEventAnnotation,
    ) -> float:
        target_geometry = ground_truth.sound_event.geometry
        source_geometry = detection.geometry
        return compute_temporal_closeness(
            target_geometry,
            source_geometry,
            ratio=self.position,
            max_distance=self.max_distance,
        )

    @affinity_functions.register(TimeAffinityConfig)
    @staticmethod
    def from_config(config: TimeAffinityConfig):
        return TimeAffinity(
            max_distance=config.max_distance,
            position=config.position,
        )


class IntervalIOUConfig(BaseConfig):
    name: Literal["interval_iou"] = "interval_iou"
    time_buffer: float = 0.0


class IntervalIOU(AffinityFunction):
    def __init__(self, time_buffer: float):
        if time_buffer < 0:
            raise ValueError("time_buffer must be non-negative")

        self.time_buffer = time_buffer

    def __call__(
        self,
        detection: Detection,
        ground_truth: data.SoundEventAnnotation,
    ) -> float:
        target_geometry = ground_truth.sound_event.geometry
        source_geometry = detection.geometry

        if self.time_buffer > 0:
            target_geometry = buffer_geometry(
                target_geometry,
                time=self.time_buffer,
            )
            source_geometry = buffer_geometry(
                source_geometry,
                time=self.time_buffer,
            )

        return compute_temporal_iou(target_geometry, source_geometry)

    @affinity_functions.register(IntervalIOUConfig)
    @staticmethod
    def from_config(config: IntervalIOUConfig):
        return IntervalIOU(time_buffer=config.time_buffer)


class BBoxIOUConfig(BaseConfig):
    name: Literal["bbox_iou"] = "bbox_iou"
    time_buffer: float = 0.0
    freq_buffer: float = 0.0


class BBoxIOU(AffinityFunction):
    def __init__(self, time_buffer: float, freq_buffer: float):
        if time_buffer < 0:
            raise ValueError("time_buffer must be non-negative")

        if freq_buffer < 0:
            raise ValueError("freq_buffer must be non-negative")

        self.time_buffer = time_buffer
        self.freq_buffer = freq_buffer

    def __call__(
        self,
        detection: Detection,
        ground_truth: data.SoundEventAnnotation,
    ):
        target_geometry = ground_truth.sound_event.geometry
        source_geometry = detection.geometry

        if self.time_buffer > 0 or self.freq_buffer > 0:
            target_geometry = buffer_geometry(
                target_geometry,
                time=self.time_buffer,
                freq=self.freq_buffer,
            )
            source_geometry = buffer_geometry(
                source_geometry,
                time=self.time_buffer,
                freq=self.freq_buffer,
            )

        return compute_bbox_iou(target_geometry, source_geometry)

    @affinity_functions.register(BBoxIOUConfig)
    @staticmethod
    def from_config(config: BBoxIOUConfig):
        return BBoxIOU(
            time_buffer=config.time_buffer,
            freq_buffer=config.freq_buffer,
        )


class GeometricIOUConfig(BaseConfig):
    name: Literal["geometric_iou"] = "geometric_iou"
    time_buffer: float = 0.0
    freq_buffer: float = 0.0


class GeometricIOU(AffinityFunction):
    def __init__(self, time_buffer: float = 0, freq_buffer: float = 0):
        if time_buffer < 0:
            raise ValueError("time_buffer must be non-negative")

        if freq_buffer < 0:
            raise ValueError("freq_buffer must be non-negative")

        self.time_buffer = time_buffer
        self.freq_buffer = freq_buffer

    def __call__(
        self,
        detection: Detection,
        ground_truth: data.SoundEventAnnotation,
    ):
        target_geometry = ground_truth.sound_event.geometry
        source_geometry = detection.geometry

        if self.time_buffer > 0 or self.freq_buffer > 0:
            target_geometry = buffer_geometry(
                target_geometry,
                time=self.time_buffer,
                freq=self.freq_buffer,
            )
            source_geometry = buffer_geometry(
                source_geometry,
                time=self.time_buffer,
                freq=self.freq_buffer,
            )

        return compute_geometric_iou(target_geometry, source_geometry)

    @affinity_functions.register(GeometricIOUConfig)
    @staticmethod
    def from_config(config: GeometricIOUConfig):
        return GeometricIOU(time_buffer=config.time_buffer)


AffinityConfig = Annotated[
    TimeAffinityConfig
    | IntervalIOUConfig
    | BBoxIOUConfig
    | GeometricIOUConfig,
    Field(discriminator="name"),
]


def build_affinity_function(
    config: AffinityConfig | None = None,
) -> AffinityFunction:
    config = config or GeometricIOUConfig()
    return affinity_functions.build(config)
