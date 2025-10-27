from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from matplotlib.figure import Figure
from soundevent import data

from batdetect2.evaluate.config import EvaluationConfig
from batdetect2.evaluate.tasks import build_task
from batdetect2.targets import build_targets
from batdetect2.typing import EvaluatorProtocol, TargetProtocol
from batdetect2.typing.postprocess import BatDetect2Prediction

__all__ = [
    "Evaluator",
    "build_evaluator",
]


class Evaluator:
    def __init__(
        self,
        targets: TargetProtocol,
        tasks: Sequence[EvaluatorProtocol],
    ):
        self.targets = targets
        self.tasks = tasks

    def evaluate(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        predictions: Sequence[BatDetect2Prediction],
    ) -> List[Any]:
        return [
            task.evaluate(clip_annotations, predictions) for task in self.tasks
        ]

    def compute_metrics(self, eval_outputs: List[Any]) -> Dict[str, float]:
        results = {}

        for task, outputs in zip(self.tasks, eval_outputs):
            results.update(task.compute_metrics(outputs))

        return results

    def generate_plots(
        self,
        eval_outputs: List[Any],
    ) -> Iterable[Tuple[str, Figure]]:
        for task, outputs in zip(self.tasks, eval_outputs):
            for name, fig in task.generate_plots(outputs):
                yield name, fig


def build_evaluator(
    config: Optional[Union[EvaluationConfig, dict]] = None,
    targets: Optional[TargetProtocol] = None,
) -> EvaluatorProtocol:
    targets = targets or build_targets()

    if config is None:
        config = EvaluationConfig()

    if not isinstance(config, EvaluationConfig):
        config = EvaluationConfig.model_validate(config)

    return Evaluator(
        targets=targets,
        tasks=[build_task(task, targets=targets) for task in config.tasks],
    )
