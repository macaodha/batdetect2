from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

import numpy as np
from pydantic import Field
from sklearn import metrics, preprocessing

from batdetect2.core import BaseConfig, Registry
from batdetect2.typing import ClipEvaluation, MetricsProtocol

__all__ = ["DetectionAP", "ClassificationAP"]


metrics_registry: Registry[MetricsProtocol, [List[str]]] = Registry("metric")


APImplementation = Literal["sklearn", "pascal_voc"]


class DetectionAPConfig(BaseConfig):
    name: Literal["detection_ap"] = "detection_ap"
    ap_implementation: APImplementation = "pascal_voc"


class DetectionAP(MetricsProtocol):
    def __init__(
        self,
        implementation: APImplementation = "pascal_voc",
    ):
        self.implementation = implementation
        self.metric = _ap_impl_mapping[self.implementation]

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
        score = float(self.metric(y_true, y_score))
        return {"detection_AP": score}

    @classmethod
    def from_config(cls, config: DetectionAPConfig, class_names: List[str]):
        return cls(implementation=config.ap_implementation)


metrics_registry.register(DetectionAPConfig, DetectionAP)


class DetectionROCAUCConfig(BaseConfig):
    name: Literal["detection_roc_auc"] = "detection_roc_auc"


class DetectionROCAUC(MetricsProtocol):
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
        score = float(metrics.roc_auc_score(y_true, y_score))
        return {"detection_ROC_AUC": score}

    @classmethod
    def from_config(
        cls, config: DetectionROCAUCConfig, class_names: List[str]
    ):
        return cls()


metrics_registry.register(DetectionROCAUCConfig, DetectionROCAUC)


class ClassificationAPConfig(BaseConfig):
    name: Literal["classification_ap"] = "classification_ap"
    ap_implementation: APImplementation = "pascal_voc"
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None


class ClassificationAP(MetricsProtocol):
    def __init__(
        self,
        class_names: List[str],
        implementation: APImplementation = "pascal_voc",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        self.implementation = implementation
        self.metric = _ap_impl_mapping[self.implementation]
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

        y_true = preprocessing.label_binarize(y_true, classes=self.class_names)
        y_pred = np.stack(y_pred)

        class_scores = {}
        for class_index, class_name in enumerate(self.class_names):
            y_true_class = y_true[:, class_index]
            y_pred_class = y_pred[:, class_index]
            class_ap = self.metric(y_true_class, y_pred_class)
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
            implementation=config.ap_implementation,
            include=config.include,
            exclude=config.exclude,
        )


metrics_registry.register(ClassificationAPConfig, ClassificationAP)


class ClassificationROCAUCConfig(BaseConfig):
    name: Literal["classification_roc_auc"] = "classification_roc_auc"
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None


class ClassificationROCAUC(MetricsProtocol):
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

        y_true = preprocessing.label_binarize(y_true, classes=self.class_names)
        y_pred = np.stack(y_pred)

        class_scores = {}
        for class_index, class_name in enumerate(self.class_names):
            y_true_class = y_true[:, class_index]
            y_pred_class = y_pred[:, class_index]
            class_roc_auc = metrics.roc_auc_score(y_true_class, y_pred_class)
            class_scores[class_name] = float(class_roc_auc)

        mean_roc_auc = np.mean(
            [value for value in class_scores.values() if value != 0]
        )

        return {
            "classification_macro_average_ROC_AUC": float(mean_roc_auc),
            **{
                f"classification_ROC_AUC/{class_name}": class_scores[
                    class_name
                ]
                for class_name in self.selected
            },
        }

    @classmethod
    def from_config(
        cls,
        config: ClassificationROCAUCConfig,
        class_names: List[str],
    ):
        return cls(
            class_names,
            include=config.include,
            exclude=config.exclude,
        )


metrics_registry.register(ClassificationROCAUCConfig, ClassificationROCAUC)


class TopClassAPConfig(BaseConfig):
    name: Literal["top_class_ap"] = "top_class_ap"
    ap_implementation: APImplementation = "pascal_voc"


class TopClassAP(MetricsProtocol):
    def __init__(
        self,
        implementation: APImplementation = "pascal_voc",
    ):
        self.implementation = implementation
        self.metric = _ap_impl_mapping[self.implementation]

    def __call__(
        self, clip_evaluations: Sequence[ClipEvaluation]
    ) -> Dict[str, float]:
        y_true = []
        y_score = []

        for clip_eval in clip_evaluations:
            for match in clip_eval.matches:
                # Ignore generic unclassified targets
                if match.gt_det and match.gt_class is None:
                    continue

                top_class = match.pred_class

                y_true.append(top_class == match.gt_class)
                y_score.append(match.pred_class_score)

        score = float(self.metric(y_true, y_score))
        return {"top_class_AP": score}

    @classmethod
    def from_config(cls, config: TopClassAPConfig, class_names: List[str]):
        return cls(implementation=config.ap_implementation)


metrics_registry.register(TopClassAPConfig, TopClassAP)


class ClassificationBalancedAccuracyConfig(BaseConfig):
    name: Literal["classification_balanced_accuracy"] = (
        "classification_balanced_accuracy"
    )


class ClassificationBalancedAccuracy(MetricsProtocol):
    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def __call__(
        self, clip_evaluations: Sequence[ClipEvaluation]
    ) -> Dict[str, float]:
        y_true = []
        y_pred = []

        for clip_eval in clip_evaluations:
            for match in clip_eval.matches:
                top_class = match.pred_class

                # Focus on matches
                if match.gt_class is None or top_class is None:
                    continue

                y_true.append(self.class_names.index(match.gt_class))
                y_pred.append(self.class_names.index(top_class))

        score = float(metrics.balanced_accuracy_score(y_true, y_pred))
        return {"classification_balanced_accuracy": score}

    @classmethod
    def from_config(
        cls,
        config: ClassificationBalancedAccuracyConfig,
        class_names: List[str],
    ):
        return cls(class_names)


metrics_registry.register(
    ClassificationBalancedAccuracyConfig,
    ClassificationBalancedAccuracy,
)


class ClipDetectionAPConfig(BaseConfig):
    name: Literal["clip_detection_ap"] = "clip_detection_ap"
    ap_implementation: APImplementation = "pascal_voc"


class ClipDetectionAP(MetricsProtocol):
    def __init__(
        self,
        implementation: APImplementation,
    ):
        self.implementation = implementation
        self.metric = _ap_impl_mapping[self.implementation]

    def __call__(
        self, clip_evaluations: Sequence[ClipEvaluation]
    ) -> Dict[str, float]:
        y_true = []
        y_score = []

        for clip_eval in clip_evaluations:
            clip_det = []
            clip_scores = []

            for match in clip_eval.matches:
                clip_det.append(match.gt_det)
                clip_scores.append(match.pred_score)

            y_true.append(any(clip_det))
            y_score.append(max(clip_scores or [0]))

        return {"clip_detection_ap": self.metric(y_true, y_score)}

    @classmethod
    def from_config(
        cls,
        config: ClipDetectionAPConfig,
        class_names: List[str],
    ):
        return cls(implementation=config.ap_implementation)


metrics_registry.register(ClipDetectionAPConfig, ClipDetectionAP)


class ClipDetectionROCAUCConfig(BaseConfig):
    name: Literal["clip_detection_roc_auc"] = "clip_detection_roc_auc"


class ClipDetectionROCAUC(MetricsProtocol):
    def __call__(
        self, clip_evaluations: Sequence[ClipEvaluation]
    ) -> Dict[str, float]:
        y_true = []
        y_score = []

        for clip_eval in clip_evaluations:
            clip_det = []
            clip_scores = []

            for match in clip_eval.matches:
                clip_det.append(match.gt_det)
                clip_scores.append(match.pred_score)

            y_true.append(any(clip_det))
            y_score.append(max(clip_scores or [0]))

        return {
            "clip_detection_ap": float(metrics.roc_auc_score(y_true, y_score))
        }

    @classmethod
    def from_config(
        cls,
        config: ClipDetectionROCAUCConfig,
        class_names: List[str],
    ):
        return cls()


metrics_registry.register(ClipDetectionROCAUCConfig, ClipDetectionROCAUC)


class ClipMulticlassAPConfig(BaseConfig):
    name: Literal["clip_multiclass_ap"] = "clip_multiclass_ap"
    ap_implementation: APImplementation = "pascal_voc"
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None


class ClipMulticlassAP(MetricsProtocol):
    def __init__(
        self,
        class_names: List[str],
        implementation: APImplementation,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
    ):
        self.implementation = implementation
        self.metric = _ap_impl_mapping[self.implementation]
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
            clip_classes = set()
            clip_scores = defaultdict(list)

            for match in clip_eval.matches:
                if match.gt_class is not None:
                    clip_classes.add(match.gt_class)

                for class_name, score in match.pred_class_scores.items():
                    clip_scores[class_name].append(score)

            y_true.append(clip_classes)
            y_pred.append(
                np.array(
                    [
                        # Get max score for each class
                        max(clip_scores.get(class_name, [0]))
                        for class_name in self.class_names
                    ]
                )
            )

        y_true = preprocessing.MultiLabelBinarizer(
            classes=self.class_names
        ).fit_transform(y_true)
        y_pred = np.stack(y_pred)

        class_scores = {}
        for class_index, class_name in enumerate(self.class_names):
            y_true_class = y_true[:, class_index]
            y_pred_class = y_pred[:, class_index]
            class_ap = self.metric(y_true_class, y_pred_class)
            class_scores[class_name] = float(class_ap)

        mean_ap = np.mean(
            [value for value in class_scores.values() if value != 0]
        )
        return {
            "clip_multiclass_mAP": float(mean_ap),
            **{
                f"clip_multiclass_AP/{class_name}": class_scores[class_name]
                for class_name in self.selected
            },
        }

    @classmethod
    def from_config(
        cls, config: ClipMulticlassAPConfig, class_names: List[str]
    ):
        return cls(
            implementation=config.ap_implementation,
            include=config.include,
            exclude=config.exclude,
            class_names=class_names,
        )


metrics_registry.register(ClipMulticlassAPConfig, ClipMulticlassAP)


class ClipMulticlassROCAUCConfig(BaseConfig):
    name: Literal["clip_multiclass_roc_auc"] = "clip_multiclass_roc_auc"
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None


class ClipMulticlassROCAUC(MetricsProtocol):
    def __init__(
        self,
        class_names: List[str],
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
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
            clip_classes = set()
            clip_scores = defaultdict(list)

            for match in clip_eval.matches:
                if match.gt_class is not None:
                    clip_classes.add(match.gt_class)

                for class_name, score in match.pred_class_scores.items():
                    clip_scores[class_name].append(score)

            y_true.append(clip_classes)
            y_pred.append(
                np.array(
                    [
                        # Get maximum score for each class
                        max(clip_scores.get(class_name, [0]))
                        for class_name in self.class_names
                    ]
                )
            )

        y_true = preprocessing.MultiLabelBinarizer(
            classes=self.class_names
        ).fit_transform(y_true)
        y_pred = np.stack(y_pred)

        class_scores = {}
        for class_index, class_name in enumerate(self.class_names):
            y_true_class = y_true[:, class_index]
            y_pred_class = y_pred[:, class_index]
            class_roc_auc = metrics.roc_auc_score(y_true_class, y_pred_class)
            class_scores[class_name] = float(class_roc_auc)

        mean_roc_auc = np.mean(
            [value for value in class_scores.values() if value != 0]
        )
        return {
            "clip_multiclass_macro_ROC_AUC": float(mean_roc_auc),
            **{
                f"clip_multiclass_ROC_AUC/{class_name}": class_scores[
                    class_name
                ]
                for class_name in self.selected
            },
        }

    @classmethod
    def from_config(
        cls,
        config: ClipMulticlassROCAUCConfig,
        class_names: List[str],
    ):
        return cls(
            include=config.include,
            exclude=config.exclude,
            class_names=class_names,
        )


metrics_registry.register(ClipMulticlassROCAUCConfig, ClipMulticlassROCAUC)

MetricConfig = Annotated[
    Union[
        DetectionAPConfig,
        DetectionROCAUCConfig,
        ClassificationAPConfig,
        ClassificationROCAUCConfig,
        TopClassAPConfig,
        ClassificationBalancedAccuracyConfig,
        ClipDetectionAPConfig,
        ClipDetectionROCAUCConfig,
        ClipMulticlassAPConfig,
        ClipMulticlassROCAUCConfig,
    ],
    Field(discriminator="name"),
]


def build_metric(config: MetricConfig, class_names: List[str]):
    return metrics_registry.build(config, class_names)


def pascal_voc_average_precision(y_true, y_score) -> float:
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    sort_ind = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sort_ind]

    num_positives = y_true.sum()
    false_pos_c = np.cumsum(1 - y_true_sorted)
    true_pos_c = np.cumsum(y_true_sorted)

    recall = true_pos_c / num_positives
    precision = true_pos_c / np.maximum(
        true_pos_c + false_pos_c,
        np.finfo(np.float64).eps,
    )

    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0

    # pascal 12 way
    mprec = np.hstack((0, precision, 0))
    mrec = np.hstack((0, recall, 1))
    for ii in range(mprec.shape[0] - 2, -1, -1):
        mprec[ii] = np.maximum(mprec[ii], mprec[ii + 1])
    inds = np.where(np.not_equal(mrec[1:], mrec[:-1]))[0] + 1
    ave_prec = ((mrec[inds] - mrec[inds - 1]) * mprec[inds]).sum()

    return ave_prec


_ap_impl_mapping: Mapping[APImplementation, Callable[[Any, Any], float]] = {
    "sklearn": metrics.average_precision_score,
    "pascal_voc": pascal_voc_average_precision,
}
