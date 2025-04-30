from typing import Dict, List

import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from batdetect2.evaluate.types import Match, MetricsProtocol

__all__ = ["DetectionAveragePrecision"]


class DetectionAveragePrecision(MetricsProtocol):
    def __call__(self, matches: List[Match]) -> Dict[str, float]:
        y_true, y_score = zip(
            *[(match.gt_det, match.pred_score) for match in matches]
        )
        score = float(metrics.average_precision_score(y_true, y_score))
        return {"detection_AP": score}


class ClassificationMeanAveragePrecision(MetricsProtocol):
    def __init__(self, class_names: List[str], per_class: bool = True):
        self.class_names = class_names
        self.per_class = per_class

    def __call__(self, matches: List[Match]) -> Dict[str, float]:
        y_true = label_binarize(
            [
                match.gt_class if match.gt_class is not None else "__NONE__"
                for match in matches
            ],
            classes=self.class_names,
        )
        y_pred = pd.DataFrame(
            [
                {
                    name: match.class_scores.get(name, 0)
                    for name in self.class_names
                }
                for match in matches
            ]
        ).fillna(0)
        mAP = metrics.average_precision_score(y_true, y_pred[self.class_names])

        ret = {
            "classification_mAP": float(mAP),
        }

        if not self.per_class:
            return ret

        for class_index, class_name in enumerate(self.class_names):
            y_true_class = y_true[:, class_index]
            y_pred_class = y_pred[class_name]
            class_ap = metrics.average_precision_score(
                y_true_class,
                y_pred_class,
            )
            ret[f"classification_AP/{class_name}"] = float(class_ap)

        return ret


class ClassificationAccuracy(MetricsProtocol):
    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def __call__(self, matches: List[Match]) -> Dict[str, float]:
        y_true = [
            match.gt_class if match.gt_class is not None else "__NONE__"
            for match in matches
        ]

        y_pred = pd.DataFrame(
            [
                {
                    name: match.class_scores.get(name, 0)
                    for name in self.class_names
                }
                for match in matches
            ]
        ).fillna(0)
        y_pred = y_pred.apply(
            lambda row: row.idxmax()
            if row.max() >= (1 - row.sum())
            else "__NONE__",
            axis=1,
        )

        accuracy = metrics.balanced_accuracy_score(
            y_true,
            y_pred,
        )

        return {
            "classification_acc": float(accuracy),
        }
