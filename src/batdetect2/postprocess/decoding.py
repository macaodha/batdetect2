"""Decodes extracted detection data into standard soundevent predictions."""

from typing import List, Optional

import numpy as np
from soundevent import data

from batdetect2.typing.postprocess import (
    Detections,
    RawPrediction,
)
from batdetect2.typing.targets import TargetProtocol

__all__ = [
    "convert_detections_to_raw_predictions",
    "convert_raw_predictions_to_clip_prediction",
    "convert_raw_prediction_to_sound_event_prediction",
    "DEFAULT_CLASSIFICATION_THRESHOLD",
]


DEFAULT_CLASSIFICATION_THRESHOLD = 0.1
"""Default threshold applied to classification scores.

Class predictions with scores below this value are typically ignored during
decoding.
"""


def convert_detections_to_raw_predictions(
    detections: Detections,
    targets: TargetProtocol,
) -> List[RawPrediction]:
    predictions = []

    for score, class_scores, time, freq, dims, feats in zip(
        detections.scores,
        detections.class_scores,
        detections.times,
        detections.frequencies,
        detections.sizes,
        detections.features,
    ):
        highest_scoring_class = targets.class_names[class_scores.argmax()]

        geom = targets.decode_roi(
            (time, freq),
            dims,
            class_name=highest_scoring_class,
        )

        predictions.append(
            RawPrediction(
                detection_score=score,
                geometry=geom,
                class_scores=class_scores,
                features=feats,
            )
        )

    return predictions


def convert_raw_predictions_to_clip_prediction(
    raw_predictions: List[RawPrediction],
    clip: data.Clip,
    targets: TargetProtocol,
    classification_threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    top_class_only: bool = False,
) -> data.ClipPrediction:
    """Convert a list of RawPredictions into a soundevent ClipPrediction."""
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
    raw_prediction: RawPrediction,
    recording: data.Recording,
    targets: TargetProtocol,
    classification_threshold: Optional[
        float
    ] = DEFAULT_CLASSIFICATION_THRESHOLD,
    top_class_only: bool = False,
):
    """Convert a single RawPrediction into a soundevent SoundEventPrediction."""
    sound_event = data.SoundEvent(
        recording=recording,
        geometry=raw_prediction.geometry,
        features=get_prediction_features(raw_prediction.features),
    )

    tags = [
        *get_generic_tags(
            raw_prediction.detection_score,
            generic_class_tags=targets.generic_class_tags,
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
    """Create PredictedTag objects for the generic category."""
    return [
        data.PredictedTag(tag=tag, score=detection_score)
        for tag in generic_class_tags
    ]


def get_prediction_features(features: np.ndarray) -> List[data.Feature]:
    """Convert an extracted feature vector DataArray into soundevent Features."""
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
    threshold: Optional[float] = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> List[data.PredictedTag]:
    """Generate specific PredictedTags based on class scores and decoder.

    Filters class scores by the threshold, sorts remaining scores descending,
    decodes the class name(s) into base tags using the `sound_event_decoder`,
    and creates `PredictedTag` objects associating the class score. Stops after
    the first (top) class if `top_class_only` is True.

    Parameters
    ----------
    class_scores : xr.DataArray
        A 1D xarray DataArray containing class probabilities/scores, indexed
        by a 'category' coordinate holding the class names.
    sound_event_decoder : SoundEventDecoder
        Function to map a class name string to a list of base `data.Tag`
        objects.
    top_class_only : bool, default=False
        If True, only generate tags for the single highest-scoring class above
        the threshold.
    threshold : float, optional
        Minimum score for a class to be considered. If None, all classes are
        processed (or top-1 if `top_class_only` is True). Defaults to
        `DEFAULT_CLASSIFICATION_THRESHOLD`.

    Returns
    -------
    List[data.PredictedTag]
        A list of `PredictedTag` objects for the class(es) that passed the
        threshold, ordered by score if `top_class_only` is False.
    """
    tags = []

    for class_name, score in _iterate_sorted(
        class_scores, targets.class_names
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
