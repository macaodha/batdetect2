from typing import (
    Annotated,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from matplotlib.figure import Figure
from pydantic import Field
from soundevent import data

from batdetect2.core.configs import BaseConfig
from batdetect2.evaluate.evaluator.base import evaluators
from batdetect2.evaluate.evaluator.clip import ClipMetricsConfig
from batdetect2.evaluate.evaluator.per_class import ClassificationMetricsConfig
from batdetect2.evaluate.evaluator.single import GlobalEvaluatorConfig
from batdetect2.targets import build_targets
from batdetect2.typing import (
    EvaluatorProtocol,
    RawPrediction,
    TargetProtocol,
)

__all__ = [
    "EvaluatorConfig",
    "build_evaluator",
]


EvaluatorConfig = Annotated[
    Union[
        ClassificationMetricsConfig,
        GlobalEvaluatorConfig,
        ClipMetricsConfig,
        "MultipleEvaluatorConfig",
    ],
    Field(discriminator="name"),
]


class MultipleEvaluatorConfig(BaseConfig):
    name: Literal["multiple_evaluations"] = "multiple_evaluations"
    evaluations: List[EvaluatorConfig] = Field(
        default_factory=lambda: [
            ClassificationMetricsConfig(),
            GlobalEvaluatorConfig(),
        ]
    )


class MultipleEvaluator:
    def __init__(
        self,
        targets: TargetProtocol,
        evaluators: Sequence[EvaluatorProtocol],
    ):
        self.targets = targets
        self.evaluators = evaluators

    def evaluate(
        self,
        clip_annotations: Sequence[data.ClipAnnotation],
        predictions: Sequence[Sequence[RawPrediction]],
    ) -> List[Any]:
        return [
            evaluator.evaluate(
                clip_annotations,
                predictions,
            )
            for evaluator in self.evaluators
        ]

    def compute_metrics(self, eval_outputs: List[Any]) -> Dict[str, float]:
        results = {}

        for evaluator, outputs in zip(self.evaluators, eval_outputs):
            results.update(evaluator.compute_metrics(outputs))

        return results

    def generate_plots(
        self,
        eval_outputs: List[Any],
    ) -> Iterable[Tuple[str, Figure]]:
        for evaluator, outputs in zip(self.evaluators, eval_outputs):
            for name, fig in evaluator.generate_plots(outputs):
                yield name, fig

    @evaluators.register(MultipleEvaluatorConfig)
    @staticmethod
    def from_config(config: MultipleEvaluatorConfig, targets: TargetProtocol):
        return MultipleEvaluator(
            evaluators=[
                build_evaluator(conf, targets=targets)
                for conf in config.evaluations
            ],
            targets=targets,
        )


def build_evaluator(
    config: Optional[EvaluatorConfig] = None,
    targets: Optional[TargetProtocol] = None,
) -> EvaluatorProtocol:
    targets = targets or build_targets()

    config = config or MultipleEvaluatorConfig()
    return evaluators.build(config, targets)
