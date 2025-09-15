from typing import Annotated, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
from pydantic import Field
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from batdetect2.configs import BaseConfig
from batdetect2.data._core import Registry
from batdetect2.typing import MetricsProtocol
from batdetect2.typing.evaluate import ClipEvaluation

__all__ = ["DetectionAP", "ClassificationAP"]


metrics_registry: Registry[MetricsProtocol, [List[str]]] = Registry("metric")


class DetectionAPConfig(BaseConfig):
    name: Literal["detection_ap"] = "detection_ap"


class DetectionAP(MetricsProtocol):
    def __call__(
        self, clip_evaluations: Sequence[ClipEvaluation]
    ) -> Dict[str, float]:
        y_true, y_score = zip(
            *[
                (match.gt_det, match.pred_score)
                for clip_eval in clip_evaluations
                for match in clip_eval.matches
            ]
        )
        score = float(metrics.average_precision_score(y_true, y_score))
        return {"detection_AP": score}

    @classmethod
    def from_config(cls, config: DetectionAPConfig, class_names: List[str]):
        return cls()


metrics_registry.register(DetectionAPConfig, DetectionAP)


class ClassificationAPConfig(BaseConfig):
    name: Literal["classification_ap"] = "classification_ap"
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None


class ClassificationAP(MetricsProtocol):
    def __init__(
        self,
        class_names: List[str],
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        self.class_names = class_names

        self.selected = class_names

        if include is not None:
            self.selected = [
                class_name
                for class_name in self.selected
                if class_name in include
            ]

        if exclude is not None:
            self.selected = [
                class_name
                for class_name in self.selected
                if class_name not in exclude
            ]

    def __call__(
        self, clip_evaluations: Sequence[ClipEvaluation]
    ) -> Dict[str, float]:
        y_true = []
        y_pred = []

        for clip_eval in clip_evaluations:
            for match in clip_eval.matches:
                # Ignore generic unclassified targets
                if match.gt_det and match.gt_class is None:
                    continue

                y_true.append(
                    match.gt_class
                    if match.gt_class is not None
                    else "__NONE__"
                )

                y_pred.append(
                    np.array(
                        [
                            match.pred_class_scores.get(name, 0)
                            for name in self.class_names
                        ]
                    )
                )

        y_true = label_binarize(y_true, classes=self.class_names)
        y_pred = np.stack(y_pred)

        class_scores = {}
        for class_index, class_name in enumerate(self.class_names):
            y_true_class = y_true[:, class_index]
            y_pred_class = y_pred[:, class_index]
            class_ap = metrics.average_precision_score(
                y_true_class,
                y_pred_class,
            )
            class_scores[class_name] = float(class_ap)

        mean_ap = np.mean(
            [value for value in class_scores.values() if value != 0]
        )

        return {
            "classification_mAP": float(mean_ap),
            **{
                f"classification_AP/{class_name}": class_scores[class_name]
                for class_name in self.selected
            },
        }

    @classmethod
    def from_config(
        cls,
        config: ClassificationAPConfig,
        class_names: List[str],
    ):
        return cls(
            class_names,
            include=config.include,
            exclude=config.exclude,
        )


metrics_registry.register(ClassificationAPConfig, ClassificationAP)


MetricConfig = Annotated[
    Union[ClassificationAPConfig, DetectionAPConfig],
    Field(discriminator="name"),
]


def build_metric(config: MetricConfig, class_names: List[str]):
    return metrics_registry.build(config, class_names)
