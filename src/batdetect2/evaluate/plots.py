import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Annotated, Dict, List, Literal, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import Field
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from batdetect2.audio import AudioConfig, build_audio_loader
from batdetect2.core import BaseConfig, Registry
from batdetect2.plotting.gallery import plot_match_gallery
from batdetect2.plotting.matches import plot_matches
from batdetect2.preprocess import PreprocessingConfig, build_preprocessor
from batdetect2.typing import (
    AudioLoader,
    ClipMatches,
    MatchEvaluation,
    PlotterProtocol,
    PreprocessorProtocol,
)

__all__ = [
    "build_plotter",
    "ExampleGallery",
    "ExampleGalleryConfig",
]


plots_registry: Registry[PlotterProtocol, [List[str]]] = Registry("plot")


class ExampleGalleryConfig(BaseConfig):
    name: Literal["example_gallery"] = "example_gallery"
    examples_per_class: int = 5
    audio: AudioConfig = Field(default_factory=AudioConfig)
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )


class ExampleGallery(PlotterProtocol):
    def __init__(
        self,
        examples_per_class: int,
        preprocessor: Optional[PreprocessorProtocol] = None,
        audio_loader: Optional[AudioLoader] = None,
    ):
        self.examples_per_class = examples_per_class
        self.preprocessor = preprocessor or build_preprocessor()
        self.audio_loader = audio_loader or build_audio_loader()

    def __call__(self, clip_evaluations: Sequence[ClipMatches]):
        per_class_matches = group_matches(clip_evaluations)

        for class_name, matches in per_class_matches.items():
            true_positives = get_binned_sample(
                matches.true_positives,
                n_examples=self.examples_per_class,
            )

            false_positives = get_binned_sample(
                matches.false_positives,
                n_examples=self.examples_per_class,
            )

            false_negatives = random.sample(
                matches.false_negatives,
                k=min(self.examples_per_class, len(matches.false_negatives)),
            )

            cross_triggers = get_binned_sample(
                matches.cross_triggers,
                n_examples=self.examples_per_class,
            )

            fig = plot_match_gallery(
                true_positives,
                false_positives,
                false_negatives,
                cross_triggers,
                preprocessor=self.preprocessor,
                audio_loader=self.audio_loader,
                n_examples=self.examples_per_class,
            )

            yield f"example_gallery/{class_name}", fig

            plt.close(fig)

    @classmethod
    def from_config(cls, config: ExampleGalleryConfig, class_names: List[str]):
        audio_loader = build_audio_loader(config.audio)
        preprocessor = build_preprocessor(
            config.preprocessing,
            input_samplerate=audio_loader.samplerate,
        )
        return cls(
            examples_per_class=config.examples_per_class,
            preprocessor=preprocessor,
            audio_loader=audio_loader,
        )


plots_registry.register(ExampleGalleryConfig, ExampleGallery)


class ClipEvaluationPlotConfig(BaseConfig):
    name: Literal["example_clip"] = "example_clip"
    num_plots: int = 5
    audio: AudioConfig = Field(default_factory=AudioConfig)
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )


class PlotClipEvaluation(PlotterProtocol):
    def __init__(
        self,
        num_plots: int = 3,
        preprocessor: Optional[PreprocessorProtocol] = None,
        audio_loader: Optional[AudioLoader] = None,
    ):
        self.preprocessor = preprocessor
        self.audio_loader = audio_loader
        self.num_plots = num_plots

    def __call__(self, clip_evaluations: Sequence[ClipMatches]):
        examples = random.sample(
            clip_evaluations,
            k=min(self.num_plots, len(clip_evaluations)),
        )

        for index, clip_evaluation in enumerate(examples):
            fig, ax = plt.subplots()
            plot_matches(
                clip_evaluation.matches,
                clip=clip_evaluation.clip,
                audio_loader=self.audio_loader,
                ax=ax,
            )
            yield f"clip_evaluation/example_{index}", fig
            plt.close(fig)

    @classmethod
    def from_config(
        cls,
        config: ClipEvaluationPlotConfig,
        class_names: List[str],
    ):
        audio_loader = build_audio_loader(config.audio)
        preprocessor = build_preprocessor(
            config.preprocessing,
            input_samplerate=audio_loader.samplerate,
        )
        return cls(
            num_plots=config.num_plots,
            preprocessor=preprocessor,
            audio_loader=audio_loader,
        )


plots_registry.register(ClipEvaluationPlotConfig, PlotClipEvaluation)


class DetectionPRCurveConfig(BaseConfig):
    name: Literal["detection_pr_curve"] = "detection_pr_curve"


class DetectionPRCurve(PlotterProtocol):
    def __call__(self, clip_evaluations: Sequence[ClipMatches]):
        y_true, y_score = zip(
            *[
                (match.gt_det, match.pred_score)
                for clip_eval in clip_evaluations
                for match in clip_eval.matches
            ]
        )
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
        fig, ax = plt.subplots()

        ax.plot(recall, precision, label="Detector")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()

        yield "detection_pr_curve", fig

    @classmethod
    def from_config(
        cls,
        config: DetectionPRCurveConfig,
        class_names: List[str],
    ):
        return cls()


plots_registry.register(DetectionPRCurveConfig, DetectionPRCurve)


class ClassificationPRCurvesConfig(BaseConfig):
    name: Literal["classification_pr_curves"] = "classification_pr_curves"
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None


class ClassificationPRCurves(PlotterProtocol):
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

    def __call__(self, clip_evaluations: Sequence[ClipMatches]):
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

        fig, ax = plt.subplots(figsize=(10, 10))
        for class_index, class_name in enumerate(self.class_names):
            if class_name not in self.selected:
                continue

            y_true_class = y_true[:, class_index]
            y_pred_class = y_pred[:, class_index]
            precision, recall, _ = metrics.precision_recall_curve(
                y_true_class,
                y_pred_class,
            )
            ax.plot(recall, precision, label=class_name)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

        yield "classification_pr_curve", fig

    @classmethod
    def from_config(
        cls,
        config: ClassificationPRCurvesConfig,
        class_names: List[str],
    ):
        return cls(
            class_names=class_names,
            include=config.include,
            exclude=config.exclude,
        )


plots_registry.register(ClassificationPRCurvesConfig, ClassificationPRCurves)


class DetectionROCCurveConfig(BaseConfig):
    name: Literal["detection_roc_curve"] = "detection_roc_curve"


class DetectionROCCurve(PlotterProtocol):
    def __call__(self, clip_evaluations: Sequence[ClipMatches]):
        y_true, y_score = zip(
            *[
                (match.gt_det, match.pred_score)
                for clip_eval in clip_evaluations
                for match in clip_eval.matches
            ]
        )
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        fig, ax = plt.subplots()

        ax.plot(fpr, tpr, label="Detection")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()

        yield "detection_roc_curve", fig

    @classmethod
    def from_config(
        cls,
        config: DetectionROCCurveConfig,
        class_names: List[str],
    ):
        return cls()


plots_registry.register(DetectionROCCurveConfig, DetectionROCCurve)


class ClassificationROCCurvesConfig(BaseConfig):
    name: Literal["classification_roc_curves"] = "classification_roc_curves"
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None


class ClassificationROCCurves(PlotterProtocol):
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

    def __call__(self, clip_evaluations: Sequence[ClipMatches]):
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

        fig, ax = plt.subplots(figsize=(10, 10))
        for class_index, class_name in enumerate(self.class_names):
            if class_name not in self.selected:
                continue

            y_true_class = y_true[:, class_index]
            y_roced_class = y_pred[:, class_index]
            fpr, tpr, _ = metrics.roc_curve(
                y_true_class,
                y_roced_class,
            )
            ax.plot(fpr, tpr, label=class_name)

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

        yield "classification_roc_curve", fig

    @classmethod
    def from_config(
        cls,
        config: ClassificationROCCurvesConfig,
        class_names: List[str],
    ):
        return cls(
            class_names=class_names,
            include=config.include,
            exclude=config.exclude,
        )


plots_registry.register(ClassificationROCCurvesConfig, ClassificationROCCurves)


class ConfusionMatrixConfig(BaseConfig):
    name: Literal["confusion_matrix"] = "confusion_matrix"
    background_class: str = "noise"


class ConfusionMatrix(PlotterProtocol):
    def __init__(self, background_class: str, class_names: List[str]):
        self.background_class = background_class
        self.class_names = class_names

    def __call__(self, clip_evaluations: Sequence[ClipMatches]):
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
                    else self.background_class
                )

                top_class = match.top_class
                y_pred.append(
                    top_class
                    if top_class is not None
                    else self.background_class
                )

        display = metrics.ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            labels=[*self.class_names, self.background_class],
        )

        yield "confusion_matrix", display.figure_

    @classmethod
    def from_config(
        cls,
        config: ConfusionMatrixConfig,
        class_names: List[str],
    ):
        return cls(
            background_class=config.background_class,
            class_names=class_names,
        )


plots_registry.register(ConfusionMatrixConfig, ConfusionMatrix)


PlotConfig = Annotated[
    Union[
        ExampleGalleryConfig,
        ClipEvaluationPlotConfig,
        DetectionPRCurveConfig,
        ClassificationPRCurvesConfig,
        DetectionROCCurveConfig,
        ClassificationROCCurvesConfig,
        ConfusionMatrixConfig,
    ],
    Field(discriminator="name"),
]


def build_plotter(
    config: PlotConfig, class_names: List[str]
) -> PlotterProtocol:
    return plots_registry.build(config, class_names)


@dataclass
class ClassMatches:
    false_positives: List[MatchEvaluation] = field(default_factory=list)
    false_negatives: List[MatchEvaluation] = field(default_factory=list)
    true_positives: List[MatchEvaluation] = field(default_factory=list)
    cross_triggers: List[MatchEvaluation] = field(default_factory=list)


def group_matches(
    clip_evaluations: Sequence[ClipMatches],
) -> Dict[str, ClassMatches]:
    class_examples = defaultdict(ClassMatches)

    for clip_evaluation in clip_evaluations:
        for match in clip_evaluation.matches:
            gt_class = match.gt_class
            pred_class = match.top_class

            if pred_class is None:
                class_examples[gt_class].false_negatives.append(match)
                continue

            if gt_class is None:
                class_examples[pred_class].false_positives.append(match)
                continue

            if gt_class != pred_class:
                class_examples[gt_class].cross_triggers.append(match)
                class_examples[pred_class].cross_triggers.append(match)
                continue

            class_examples[gt_class].true_positives.append(match)

    return class_examples


def get_binned_sample(matches: List[MatchEvaluation], n_examples: int = 5):
    if len(matches) < n_examples:
        return matches

    indices, pred_scores = zip(
        *[
            (index, match.pred_class_scores[pred_class])
            for index, match in enumerate(matches)
            if (pred_class := match.top_class) is not None
        ]
    )

    bins = pd.qcut(pred_scores, q=n_examples, labels=False, duplicates="drop")
    df = pd.DataFrame({"indices": indices, "bins": bins})
    sample = df.groupby("bins").sample(1)
    return [matches[ind] for ind in sample["indices"]]
