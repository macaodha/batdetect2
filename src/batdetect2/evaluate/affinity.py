from typing import Annotated, Literal, Optional, Union

from pydantic import Field
from soundevent import data
from soundevent.evaluation import compute_affinity

from batdetect2.configs import BaseConfig
from batdetect2.data._core import Registry
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

    @classmethod
    def from_config(cls, config: TimeAffinityConfig):
        return cls(time_buffer=config.time_buffer)


affinity_functions.register(TimeAffinityConfig, TimeAffinity)


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

    @classmethod
    def from_config(cls, config: IntervalIOUConfig):
        return cls(time_buffer=config.time_buffer)


affinity_functions.register(IntervalIOUConfig, IntervalIOU)


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

    intersection = max(
        0, min(end_time1, end_time2) - max(start_time1, start_time2)
    )
    union = (
        (end_time1 - start_time1) + (end_time2 - start_time2) - intersection
    )

    if union == 0:
        return 0

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

    @classmethod
    def from_config(cls, config: GeometricIOUConfig):
        return cls(time_buffer=config.time_buffer)


affinity_functions.register(GeometricIOUConfig, GeometricIOU)

AffinityConfig = Annotated[
    Union[
        TimeAffinityConfig,
        IntervalIOUConfig,
        GeometricIOUConfig,
    ],
    Field(discriminator="name"),
]


def build_affinity_function(
    config: Optional[AffinityConfig] = None,
) -> AffinityFunction:
    config = config or GeometricIOUConfig()
    return affinity_functions.build(config)
