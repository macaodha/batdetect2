from typing import Annotated, Literal

from pydantic import Field
from soundevent import data
from soundevent.evaluation import compute_affinity
from soundevent.geometry import compute_interval_overlap

from batdetect2.core.configs import BaseConfig
from batdetect2.core.registries import Registry
from batdetect2.typing.evaluate import AffinityFunction

affinity_functions: Registry[AffinityFunction, []] = Registry(
    "matching_strategy"
)


class TimeAffinityConfig(BaseConfig):
    name: Literal["time_affinity"] = "time_affinity"
    time_buffer: float = 0.01


class TimeAffinity(AffinityFunction):
    def __init__(self, time_buffer: float):
        self.time_buffer = time_buffer

    def __call__(self, geometry1: data.Geometry, geometry2: data.Geometry):
        return compute_timestamp_affinity(
            geometry1, geometry2, time_buffer=self.time_buffer
        )

    @affinity_functions.register(TimeAffinityConfig)
    @staticmethod
    def from_config(config: TimeAffinityConfig):
        return TimeAffinity(time_buffer=config.time_buffer)


def compute_timestamp_affinity(
    geometry1: data.Geometry,
    geometry2: data.Geometry,
    time_buffer: float = 0.01,
) -> float:
    assert isinstance(geometry1, data.TimeStamp)
    assert isinstance(geometry2, data.TimeStamp)

    start_time1 = geometry1.coordinates
    start_time2 = geometry2.coordinates

    a = min(start_time1, start_time2)
    b = max(start_time1, start_time2)

    if b - a >= 2 * time_buffer:
        return 0

    intersection = a - b + 2 * time_buffer
    union = b - a + 2 * time_buffer
    return intersection / union


class IntervalIOUConfig(BaseConfig):
    name: Literal["interval_iou"] = "interval_iou"
    time_buffer: float = 0.01


class IntervalIOU(AffinityFunction):
    def __init__(self, time_buffer: float):
        self.time_buffer = time_buffer

    def __call__(self, geometry1: data.Geometry, geometry2: data.Geometry):
        return compute_interval_iou(
            geometry1,
            geometry2,
            time_buffer=self.time_buffer,
        )

    @affinity_functions.register(IntervalIOUConfig)
    @staticmethod
    def from_config(config: IntervalIOUConfig):
        return IntervalIOU(time_buffer=config.time_buffer)


def compute_interval_iou(
    geometry1: data.Geometry,
    geometry2: data.Geometry,
    time_buffer: float = 0.01,
) -> float:
    assert isinstance(geometry1, data.TimeInterval)
    assert isinstance(geometry2, data.TimeInterval)

    start_time1, end_time1 = geometry1.coordinates
    start_time2, end_time2 = geometry1.coordinates

    start_time1 -= time_buffer
    start_time2 -= time_buffer
    end_time1 += time_buffer
    end_time2 += time_buffer

    intersection = compute_interval_overlap(
        (start_time1, end_time1),
        (start_time2, end_time2),
    )

    union = (
        (end_time1 - start_time1) + (end_time2 - start_time2) - intersection
    )

    if union == 0:
        return 0

    return intersection / union


class BBoxIOUConfig(BaseConfig):
    name: Literal["bbox_iou"] = "bbox_iou"
    time_buffer: float = 0.01
    freq_buffer: float = 1000


class BBoxIOU(AffinityFunction):
    def __init__(self, time_buffer: float, freq_buffer: float):
        self.time_buffer = time_buffer
        self.freq_buffer = freq_buffer

    def __call__(self, geometry1: data.Geometry, geometry2: data.Geometry):
        if not isinstance(geometry1, data.BoundingBox):
            raise TypeError(
                f"Expected geometry1 to be a BoundingBox, got {type(geometry1)}"
            )

        if not isinstance(geometry2, data.BoundingBox):
            raise TypeError(
                f"Expected geometry2 to be a BoundingBox, got {type(geometry2)}"
            )
        return bbox_iou(
            geometry1,
            geometry2,
            time_buffer=self.time_buffer,
            freq_buffer=self.freq_buffer,
        )

    @affinity_functions.register(BBoxIOUConfig)
    @staticmethod
    def from_config(config: BBoxIOUConfig):
        return BBoxIOU(
            time_buffer=config.time_buffer,
            freq_buffer=config.freq_buffer,
        )


def bbox_iou(
    geometry1: data.BoundingBox,
    geometry2: data.BoundingBox,
    time_buffer: float = 0.01,
    freq_buffer: float = 1000,
) -> float:
    start_time1, low_freq1, end_time1, high_freq1 = geometry1.coordinates
    start_time2, low_freq2, end_time2, high_freq2 = geometry2.coordinates

    start_time1 -= time_buffer
    start_time2 -= time_buffer
    end_time1 += time_buffer
    end_time2 += time_buffer

    low_freq1 -= freq_buffer
    low_freq2 -= freq_buffer
    high_freq1 += freq_buffer
    high_freq2 += freq_buffer

    time_intersection = compute_interval_overlap(
        (start_time1, end_time1),
        (start_time2, end_time2),
    )

    freq_intersection = max(
        0,
        min(high_freq1, high_freq2) - max(low_freq1, low_freq2),
    )

    intersection = time_intersection * freq_intersection

    if intersection == 0:
        return 0

    union = (
        (end_time1 - start_time1) * (high_freq1 - low_freq1)
        + (end_time2 - start_time2) * (high_freq2 - low_freq2)
        - intersection
    )

    return intersection / union


class GeometricIOUConfig(BaseConfig):
    name: Literal["geometric_iou"] = "geometric_iou"
    time_buffer: float = 0.01
    freq_buffer: float = 1000


class GeometricIOU(AffinityFunction):
    def __init__(self, time_buffer: float):
        self.time_buffer = time_buffer

    def __call__(self, geometry1: data.Geometry, geometry2: data.Geometry):
        return compute_affinity(
            geometry1,
            geometry2,
            time_buffer=self.time_buffer,
        )

    @affinity_functions.register(GeometricIOUConfig)
    @staticmethod
    def from_config(config: GeometricIOUConfig):
        return GeometricIOU(time_buffer=config.time_buffer)


AffinityConfig = Annotated[
    TimeAffinityConfig | IntervalIOUConfig | BBoxIOUConfig | GeometricIOUConfig,
    Field(discriminator="name"),
]


def build_affinity_function(
    config: AffinityConfig | None = None,
) -> AffinityFunction:
    config = config or GeometricIOUConfig()
    return affinity_functions.build(config)
