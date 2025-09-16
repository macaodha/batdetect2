import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Annotated, Dict, List, Literal, Optional, Sequence, Union

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import Field

from batdetect2.core.configs import BaseConfig
from batdetect2.core.registries import Registry
from batdetect2.plotting.clips import PreprocessorProtocol, build_audio_loader
from batdetect2.plotting.gallery import plot_match_gallery
from batdetect2.preprocess import PreprocessingConfig, build_preprocessor
from batdetect2.typing.evaluate import (
    ClipEvaluation,
    MatchEvaluation,
    PlotterProtocol,
)
from batdetect2.typing.preprocess import AudioLoader

__all__ = [
    "build_plotter",
    "ExampleGallery",
    "ExampleGalleryConfig",
]


plots_registry: Registry[PlotterProtocol, []] = Registry("plot")


class ExampleGalleryConfig(BaseConfig):
    name: Literal["example_gallery"] = "example_gallery"
    examples_per_class: int = 5
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

    def __call__(self, clip_evaluations: Sequence[ClipEvaluation]):
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
    def from_config(cls, config: ExampleGalleryConfig):
        preprocessor = build_preprocessor(config.preprocessing)
        audio_loader = build_audio_loader(config.preprocessing.audio)
        return cls(
            examples_per_class=config.examples_per_class,
            preprocessor=preprocessor,
            audio_loader=audio_loader,
        )


plots_registry.register(ExampleGalleryConfig, ExampleGallery)


PlotConfig = Annotated[
    Union[ExampleGalleryConfig,], Field(discriminator="name")
]


def build_plotter(config: PlotConfig) -> PlotterProtocol:
    return plots_registry.build(config)


@dataclass
class ClassMatches:
    false_positives: List[MatchEvaluation] = field(default_factory=list)
    false_negatives: List[MatchEvaluation] = field(default_factory=list)
    true_positives: List[MatchEvaluation] = field(default_factory=list)
    cross_triggers: List[MatchEvaluation] = field(default_factory=list)


def group_matches(
    clip_evaluations: Sequence[ClipEvaluation],
) -> Dict[str, ClassMatches]:
    class_examples = defaultdict(ClassMatches)

    for clip_evaluation in clip_evaluations:
        for match in clip_evaluation.matches:
            gt_class = match.gt_class
            pred_class = match.pred_class

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
            if (pred_class := match.pred_class) is not None
        ]
    )

    bins = pd.qcut(pred_scores, q=n_examples, labels=False, duplicates="drop")
    df = pd.DataFrame({"indices": indices, "bins": bins})
    sample = df.groupby("bins").sample(1)
    return [matches[ind] for ind in sample["indices"]]
