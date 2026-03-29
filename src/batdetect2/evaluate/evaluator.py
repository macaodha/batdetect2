from typing import Any, Dict, Iterable, List, Sequence, Tuple

from matplotlib.figure import Figure
from soundevent import data

from batdetect2.evaluate.config import EvaluationConfig
from batdetect2.evaluate.tasks import build_task
from batdetect2.evaluate.types import EvaluationTaskProtocol, EvaluatorProtocol
from batdetect2.outputs import OutputTransformProtocol, build_output_transform
from batdetect2.postprocess.types import ClipDetections, ClipDetectionsTensor
from batdetect2.targets import build_roi_mapping, build_targets
from batdetect2.targets.types import ROIMapperProtocol, TargetProtocol

__all__ = [
    "Evaluator",
    "build_evaluator",
]


class Evaluator:
    def __init__(
        self,
        targets: TargetProtocol,
        transform: OutputTransformProtocol,
        tasks: Sequence[EvaluationTaskProtocol],
    ):
        self.targets = targets
        self.transform = transform
        self.tasks = tasks

    def to_clip_detections_batch(
        self,
        clip_detections: Sequence[ClipDetectionsTensor],
        clips: Sequence[data.Clip],
    ) -> list[ClipDetections]:
        return [
            self.transform.to_clip_detections(detections=dets, clip=clip)
            for dets, clip in zip(clip_detections, clips, strict=False)
        ]

    def evaluate(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        predictions: Sequence[ClipDetections],
    ) -> List[Any]:
        return [
            task.evaluate(clip_annotations, predictions) for task in self.tasks
        ]

    def compute_metrics(self, eval_outputs: List[Any]) -> Dict[str, float]:
        results = {}

        for task, outputs in zip(self.tasks, eval_outputs, strict=False):
            results.update(task.compute_metrics(outputs))

        return results

    def generate_plots(
        self,
        eval_outputs: List[Any],
    ) -> Iterable[Tuple[str, Figure]]:
        for task, outputs in zip(self.tasks, eval_outputs, strict=False):
            for name, fig in task.generate_plots(outputs):
                yield name, fig


def build_evaluator(
    config: EvaluationConfig | dict | None = None,
    targets: TargetProtocol | None = None,
    roi_mapper: ROIMapperProtocol | None = None,
    transform: OutputTransformProtocol | None = None,
) -> EvaluatorProtocol:
    targets = targets or build_targets()

    roi_mapper = roi_mapper or build_roi_mapping()

    if config is None:
        config = EvaluationConfig()

    if not isinstance(config, EvaluationConfig):
        config = EvaluationConfig.model_validate(config)

    transform = transform or build_output_transform(
        targets=targets,
        roi_mapper=roi_mapper,
    )

    return Evaluator(
        targets=targets,
        transform=transform,
        tasks=[build_task(task, targets=targets) for task in config.tasks],
    )
