from typing import List, Optional, Tuple

import pandas as pd
from soundevent import data

from batdetect2.evaluate.dataframe import extract_matches_dataframe
from batdetect2.evaluate.match import match_all_predictions
from batdetect2.evaluate.metrics import (
    ClassificationAccuracy,
    ClassificationMeanAveragePrecision,
    DetectionAveragePrecision,
)
from batdetect2.models import Model
from batdetect2.plotting.clips import build_audio_loader
from batdetect2.postprocess import get_raw_predictions
from batdetect2.preprocess import build_preprocessor
from batdetect2.targets import build_targets
from batdetect2.train.config import FullTrainingConfig
from batdetect2.train.dataset import ValidationDataset
from batdetect2.train.labels import build_clip_labeler
from batdetect2.train.train import build_val_loader


def evaluate(
    model: Model,
    test_annotations: List[data.ClipAnnotation],
    config: Optional[FullTrainingConfig] = None,
    num_workers: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    config = config or FullTrainingConfig()

    audio_loader = build_audio_loader(config.preprocess.audio)

    preprocessor = build_preprocessor(config.preprocess)

    targets = build_targets(config.targets)

    labeller = build_clip_labeler(
        targets,
        min_freq=preprocessor.min_freq,
        max_freq=preprocessor.max_freq,
        config=config.train.labels,
    )

    loader = build_val_loader(
        test_annotations,
        audio_loader=audio_loader,
        labeller=labeller,
        preprocessor=preprocessor,
        config=config.train,
        num_workers=num_workers,
    )

    dataset: ValidationDataset = loader.dataset  # type: ignore

    clip_annotations = []
    predictions = []

    for batch in loader:
        outputs = model.detector(batch.spec)

        clip_annotations = [
            dataset.clip_annotations[int(example_idx)]
            for example_idx in batch.idx
        ]

        predictions = get_raw_predictions(
            outputs,
            clips=[
                clip_annotation.clip for clip_annotation in clip_annotations
            ],
            targets=targets,
            postprocessor=model.postprocessor,
        )

        clip_annotations.extend(clip_annotations)
        predictions.extend(predictions)

    matches = match_all_predictions(
        clip_annotations,
        predictions,
        targets=targets,
        config=config.evaluation.match,
    )

    df = extract_matches_dataframe(matches)

    metrics = [
        DetectionAveragePrecision(),
        ClassificationMeanAveragePrecision(class_names=targets.class_names),
        ClassificationAccuracy(class_names=targets.class_names),
    ]

    results = {
        name: value
        for metric in metrics
        for name, value in metric(matches).items()
    }

    return df, results
