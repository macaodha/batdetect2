from collections.abc import Sequence
from dataclasses import replace
from typing import Annotated, Literal

from pydantic import Field
from soundevent.geometry import shift_geometry

from batdetect2.core.registries import (
    ImportConfig,
    Registry,
    add_import_config,
)
from batdetect2.outputs.types import DetectionTransform
from batdetect2.postprocess.types import Detection

__all__ = [
    "DetectionTransformConfig",
    "detection_transforms",
    "shift_detection_time",
    "shift_detections_to_start_time",
]


detection_transforms: Registry[DetectionTransform, []] = Registry(
    "detection_transform"
)


@add_import_config(detection_transforms)
class DetectionTransformImportConfig(ImportConfig):
    name: Literal["import"] = "import"


DetectionTransformConfig = Annotated[
    DetectionTransformImportConfig,
    Field(discriminator="name"),
]


def shift_detection_time(detection: Detection, time: float) -> Detection:
    geometry = shift_geometry(detection.geometry, time=time)
    return replace(detection, geometry=geometry)


def shift_detections_to_start_time(
    detections: Sequence[Detection],
    start_time: float = 0,
) -> list[Detection]:
    if start_time == 0:
        return list(detections)

    return [
        shift_detection_time(detection, time=start_time)
        for detection in detections
    ]
