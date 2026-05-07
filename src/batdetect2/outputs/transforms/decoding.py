"""Decode extracted tensors into output-friendly detection objects."""

from typing import List

import numpy as np
from soundevent import data

from batdetect2.postprocess.types import ClipDetectionsArray, Detection
from batdetect2.targets.types import ROIMapperProtocol, TargetProtocol

__all__ = [
    "DEFAULT_CLASSIFICATION_THRESHOLD",
    "convert_raw_prediction_to_sound_event_prediction",
    "convert_raw_predictions_to_clip_prediction",
    "get_class_tags",
    "get_generic_tags",
    "get_prediction_features",
    "to_detections",
]


DEFAULT_CLASSIFICATION_THRESHOLD = 0.1


def to_detections(
    detections: ClipDetectionsArray,
    targets: TargetProtocol,
    roi_mapper: ROIMapperProtocol,
) -> List[Detection]:
    predictions = []

    for score, class_scores, time, freq, dims, feats in zip(
        detections.scores,
        detections.class_scores,
        detections.times,
        detections.frequencies,
        detections.sizes,
        detections.features,
        strict=False,
    ):
        highest_scoring_class = targets.class_names[class_scores.argmax()]

        geom = roi_mapper.decode(
            (time, freq),
            dims,
            class_name=highest_scoring_class,
        )

        predictions.append(
            Detection(
                detection_score=score,
                geometry=geom,
                class_scores=class_scores,
                features=feats,
            )
        )

    return predictions


def convert_raw_predictions_to_clip_prediction(
    raw_predictions: List[Detection],
    clip: data.Clip,
    targets: TargetProtocol,
    classification_threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    top_class_only: bool = False,
) -> data.ClipPrediction:
    return data.ClipPrediction(
        clip=clip,
        sound_events=[
            convert_raw_prediction_to_sound_event_prediction(
                prediction,
                recording=clip.recording,
                targets=targets,
                classification_threshold=classification_threshold,
                top_class_only=top_class_only,
            )
            for prediction in raw_predictions
        ],
    )


def convert_raw_prediction_to_sound_event_prediction(
    raw_prediction: Detection,
    recording: data.Recording,
    targets: TargetProtocol,
    classification_threshold: float | None = DEFAULT_CLASSIFICATION_THRESHOLD,
    top_class_only: bool = False,
):
    sound_event = data.SoundEvent(
        recording=recording,
        geometry=raw_prediction.geometry,
        features=get_prediction_features(raw_prediction.features),
    )

    tags = [
        *get_generic_tags(
            raw_prediction.detection_score,
            generic_class_tags=targets.detection_class_tags,
        ),
        *get_class_tags(
            raw_prediction.class_scores,
            targets=targets,
            top_class_only=top_class_only,
            threshold=classification_threshold,
        ),
    ]

    return data.SoundEventPrediction(
        sound_event=sound_event,
        score=raw_prediction.detection_score,
        tags=tags,
    )


def get_generic_tags(
    detection_score: float,
    generic_class_tags: List[data.Tag],
) -> List[data.PredictedTag]:
    return [
        data.PredictedTag(tag=tag, score=detection_score)
        for tag in generic_class_tags
    ]


def get_prediction_features(features: np.ndarray) -> List[data.Feature]:
    return [
        data.Feature(
            term=data.Term(
                name=f"batdetect2:f{index}",
                label=f"BatDetect Feature {index}",
                definition="Automatically extracted features by BatDetect2",
            ),
            value=value,
        )
        for index, value in enumerate(features)
    ]


def get_class_tags(
    class_scores: np.ndarray,
    targets: TargetProtocol,
    top_class_only: bool = False,
    threshold: float | None = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> List[data.PredictedTag]:
    tags = []

    for class_name, score in _iterate_sorted(
        class_scores,
        targets.class_names,
    ):
        if threshold is not None and score < threshold:
            continue

        class_tags = targets.decode_class(class_name)

        for tag in class_tags:
            tags.append(
                data.PredictedTag(
                    tag=tag,
                    score=score,
                )
            )

        if top_class_only:
            break

    return tags


def _iterate_sorted(array: np.ndarray, class_names: List[str]):
    indices = np.argsort(-array)
    for index in indices:
        yield str(class_names[index]), float(array[index])
