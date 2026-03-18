from collections.abc import Sequence

from pydantic import Field
from soundevent import data

from batdetect2.core.configs import BaseConfig
from batdetect2.outputs.transforms.clip_transforms import (
    ClipDetectionsTransformConfig,
)
from batdetect2.outputs.transforms.clip_transforms import (
    clip_transforms as clip_transform_registry,
)
from batdetect2.outputs.transforms.decoding import to_detections
from batdetect2.outputs.transforms.detection_transforms import (
    DetectionTransformConfig,
    shift_detections_to_start_time,
)
from batdetect2.outputs.transforms.detection_transforms import (
    detection_transforms as detection_transform_registry,
)
from batdetect2.outputs.types import (
    ClipDetectionsTransform,
    DetectionTransform,
    OutputTransformProtocol,
)
from batdetect2.postprocess.types import (
    ClipDetections,
    ClipDetectionsTensor,
    Detection,
)
from batdetect2.targets.types import TargetProtocol

__all__ = [
    "ClipDetectionsTransformConfig",
    "DetectionTransformConfig",
    "OutputTransform",
    "OutputTransformConfig",
    "build_output_transform",
]


class OutputTransformConfig(BaseConfig):
    detection_transforms: list[DetectionTransformConfig] = Field(
        default_factory=list
    )
    clip_transforms: list[ClipDetectionsTransformConfig] = Field(
        default_factory=list
    )


class OutputTransform(OutputTransformProtocol):
    detection_transform_steps: list[DetectionTransform]
    clip_transform_steps: list[ClipDetectionsTransform]

    def __init__(
        self,
        targets: TargetProtocol,
        detection_transform_steps: Sequence[DetectionTransform] = (),
        clip_transform_steps: Sequence[ClipDetectionsTransform] = (),
    ):
        self.targets = targets
        self.detection_transform_steps = list(detection_transform_steps)
        self.clip_transform_steps = list(clip_transform_steps)

    def __call__(
        self,
        predictions: Sequence[ClipDetections],
    ) -> list[ClipDetections]:
        return [
            self._transform_prediction(prediction)
            for prediction in predictions
        ]

    def _transform_prediction(
        self,
        prediction: ClipDetections,
    ) -> ClipDetections:
        detections = shift_detections_to_start_time(
            prediction.detections,
            start_time=prediction.clip.start_time,
        )
        detections = self.transform_detections(detections)
        return self.transform_clip_detections(
            ClipDetections(clip=prediction.clip, detections=detections)
        )

    def to_detections(
        self,
        detections: ClipDetectionsTensor,
        start_time: float = 0,
    ) -> list[Detection]:
        decoded = to_detections(detections.numpy(), targets=self.targets)
        shifted = shift_detections_to_start_time(
            decoded,
            start_time=start_time,
        )
        return self.transform_detections(shifted)

    def to_clip_detections(
        self,
        detections: ClipDetectionsTensor,
        clip: data.Clip,
    ) -> ClipDetections:
        prediction = ClipDetections(
            clip=clip,
            detections=self.to_detections(
                detections,
                start_time=clip.start_time,
            ),
        )
        return self.transform_clip_detections(prediction)

    def transform_detections(
        self,
        detections: Sequence[Detection],
    ) -> list[Detection]:
        out: list[Detection] = []
        for detection in detections:
            transformed = self.transform_detection(detection)

            if transformed is None:
                continue

            out.append(transformed)

        return out

    def transform_detection(
        self,
        detection: Detection,
    ) -> Detection | None:
        for transform in self.detection_transform_steps:
            transformed = transform(detection)

            if transformed is None:
                return None

            detection = transformed

        return detection

    def transform_clip_detections(
        self,
        prediction: ClipDetections,
    ) -> ClipDetections:
        for transform in self.clip_transform_steps:
            prediction = transform(prediction)
        return prediction


def build_output_transform(
    config: OutputTransformConfig | dict | None = None,
    targets: TargetProtocol | None = None,
) -> OutputTransformProtocol:
    from batdetect2.targets import build_targets

    if config is None:
        config = OutputTransformConfig()

    if not isinstance(config, OutputTransformConfig):
        config = OutputTransformConfig.model_validate(config)

    targets = targets or build_targets()

    return OutputTransform(
        targets=targets,
        detection_transform_steps=[
            detection_transform_registry.build(transform_config)
            for transform_config in config.detection_transforms
        ],
        clip_transform_steps=[
            clip_transform_registry.build(transform_config)
            for transform_config in config.clip_transforms
        ],
    )
