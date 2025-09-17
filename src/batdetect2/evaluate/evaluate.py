from typing import List, Optional, Tuple

import pandas as pd
from soundevent import data

from batdetect2.audio import build_audio_loader
from batdetect2.evaluate.config import EvaluationConfig
from batdetect2.evaluate.dataframe import extract_matches_dataframe
from batdetect2.evaluate.evaluator import build_evaluator
from batdetect2.evaluate.metrics import ClassificationAP, DetectionAP
from batdetect2.models import Model
from batdetect2.plotting.clips import AudioLoader, PreprocessorProtocol
from batdetect2.postprocess import get_raw_predictions
from batdetect2.preprocess import build_preprocessor
from batdetect2.targets import build_targets
from batdetect2.train.dataset import ValidationDataset
from batdetect2.train.labels import build_clip_labeler
from batdetect2.train.train import build_val_loader
from batdetect2.typing import ClipLabeller, TargetProtocol


def evaluate(
    model: Model,
    test_annotations: List[data.ClipAnnotation],
    targets: Optional[TargetProtocol] = None,
    audio_loader: Optional[AudioLoader] = None,
    preprocessor: Optional[PreprocessorProtocol] = None,
    labeller: Optional[ClipLabeller] = None,
    config: Optional[EvaluationConfig] = None,
    num_workers: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    config = config or EvaluationConfig()

    audio_loader = audio_loader or build_audio_loader()

    preprocessor = preprocessor or build_preprocessor(
        input_samplerate=audio_loader.samplerate,
    )

    targets = targets or build_targets()

    labeller = labeller or build_clip_labeler(
        targets,
        min_freq=preprocessor.min_freq,
        max_freq=preprocessor.max_freq,
    )

    loader = build_val_loader(
        test_annotations,
        audio_loader=audio_loader,
        labeller=labeller,
        preprocessor=preprocessor,
        num_workers=num_workers,
    )

    dataset: ValidationDataset = loader.dataset  # type: ignore

    clip_annotations = []
    predictions = []

    evaluator = build_evaluator(config=config, targets=targets)

    for batch in loader:
        outputs = model.detector(batch.spec)

        clip_annotations = [
            dataset.clip_annotations[int(example_idx)]
            for example_idx in batch.idx
        ]

        predictions = get_raw_predictions(
            outputs,
            start_times=[
                clip_annotation.clip.start_time
                for clip_annotation in clip_annotations
            ],
            targets=targets,
            postprocessor=model.postprocessor,
        )

        clip_annotations.extend(clip_annotations)
        predictions.extend(predictions)

    matches = evaluator.evaluate(clip_annotations, predictions)
    df = extract_matches_dataframe(matches)

    metrics = [
        DetectionAP(),
        ClassificationAP(class_names=targets.class_names),
    ]

    results = {
        name: value
        for metric in metrics
        for name, value in metric(matches).items()
    }

    return df, results
