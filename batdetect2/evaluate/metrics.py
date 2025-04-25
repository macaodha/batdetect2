from typing import List

import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from batdetect2.evaluate.types import Match, MetricsProtocol

__all__ = ["DetectionAveragePrecision"]


class DetectionAveragePrecision(MetricsProtocol):
    name: str = "detection/average_precision"

    def __call__(self, matches: List[Match]) -> float:
        y_true, y_score = zip(
            *[(match.gt_det, match.pred_score) for match in matches]
        )
        return float(metrics.average_precision_score(y_true, y_score))


class ClassificationMeanAveragePrecision(MetricsProtocol):
    name: str = "classification/average_precision"

    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def __call__(self, matches: List[Match]) -> float:
        y_true = label_binarize(
            [
                match.gt_class if match.gt_class is not None else "__NONE__"
                for match in matches
            ],
            classes=self.class_names,
        )
        y_pred = pd.DataFrame([match.class_scores for match in matches])
        return float(
            metrics.average_precision_score(y_true, y_pred[self.class_names])
        )
