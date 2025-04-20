"""Decodes extracted detection data into standard soundevent predictions.

This module handles the final stages of the BatDetect2 postprocessing pipeline.
It takes the structured detection data extracted by the `extraction` module
(typically an `xarray.Dataset` containing scores, positions, predicted sizes,
 class probabilities, and features for each detection point) and converts it
into meaningful, standardized prediction objects based on the `soundevent` data
model.

The process involves:
1.  Converting the `xarray.Dataset` into a list of intermediate `RawPrediction`
    objects, using a configured geometry builder to recover bounding boxes from
    predicted positions and sizes (`convert_xr_dataset_to_raw_prediction`).
2.  Converting each `RawPrediction` into a
    `soundevent.data.SoundEventPrediction`, which involves:
    - Creating the `soundevent.data.SoundEvent` with geometry and features.
    - Decoding the predicted class probabilities into representative tags using
      a configured class decoder (`SoundEventDecoder`).
    - Applying a classification threshold.
    - Optionally selecting only the single highest-scoring class (top-1) or
      including tags for all classes above the threshold (multi-label).
    - Adding generic class tags as a baseline.
    - Associating scores with the final prediction and tags.
    (`convert_raw_prediction_to_sound_event_prediction`)
3.  Grouping the `SoundEventPrediction` objects for a given audio segment into
    a `soundevent.data.ClipPrediction`
    (`convert_raw_predictions_to_clip_prediction`).
"""

from typing import List, Optional

import xarray as xr
from soundevent import data
from soundevent.geometry import compute_bounds

from batdetect2.postprocess.types import GeometryBuilder, RawPrediction
from batdetect2.targets.classes import SoundEventDecoder

__all__ = [
    "convert_xr_dataset_to_raw_prediction",
    "convert_raw_predictions_to_clip_prediction",
    "convert_raw_prediction_to_sound_event_prediction",
    "DEFAULT_CLASSIFICATION_THRESHOLD",
]


DEFAULT_CLASSIFICATION_THRESHOLD = 0.1
"""Default threshold applied to classification scores.

Class predictions with scores below this value are typically ignored during
decoding.
"""


def convert_xr_dataset_to_raw_prediction(
    detection_dataset: xr.Dataset,
    geometry_builder: GeometryBuilder,
) -> List[RawPrediction]:
    """Convert an xarray.Dataset of detections to RawPrediction objects.

    Takes the output of the extraction step (`extract_detection_xr_dataset`)
    and transforms each detection entry into an intermediate `RawPrediction`
    object. This involves recovering the geometry (e.g., bounding box) from
    the predicted position and scaled size dimensions using the provided
    `geometry_builder` function.

    Parameters
    ----------
    detection_dataset : xr.Dataset
        An xarray Dataset containing aligned detection information, typically
        output by `extract_detection_xr_dataset`. Expected variables include
        'scores' (with time/freq coords), 'dimensions', 'classes', 'features'.
        Must have a 'detection' dimension.
    geometry_builder : GeometryBuilder
        A function that takes a position tuple `(time, freq)` and a NumPy array
        of dimensions, and returns the corresponding reconstructed
        `soundevent.data.Geometry`.

    Returns
    -------
    List[RawPrediction]
        A list of `RawPrediction` objects, each containing the detection score,
        recovered bounding box coordinates (start/end time, low/high freq),
        the vector of class scores, and the feature vector for one detection.

    Raises
    ------
    AttributeError, KeyError, ValueError
        If `detection_dataset` is missing expected variables ('scores',
        'dimensions', 'classes', 'features') or coordinates ('time', 'freq'
        associated with 'scores'), or if `geometry_builder` fails.
    """
    detections = []

    for det_num in range(detection_dataset.dims["detection"]):
        det_info = detection_dataset.sel(detection=det_num)

        geom = geometry_builder(
            (det_info.time, det_info.freq),
            det_info.dimensions,
        )

        start_time, low_freq, end_time, high_freq = compute_bounds(geom)

        classes = det_info.classes
        features = det_info.features

        detections.append(
            RawPrediction(
                detection_score=det_info.score,
                start_time=start_time,
                end_time=end_time,
                low_freq=low_freq,
                high_freq=high_freq,
                class_scores=classes,
                features=features,
            )
        )

    return detections


def convert_raw_predictions_to_clip_prediction(
    raw_predictions: List[RawPrediction],
    clip: data.Clip,
    sound_event_decoder: SoundEventDecoder,
    generic_class_tags: List[data.Tag],
    classification_threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    top_class_only: bool = False,
) -> data.ClipPrediction:
    """Convert a list of RawPredictions into a soundevent ClipPrediction.

    Iterates through `raw_predictions` (assumed to belong to a single clip),
    converts each one into a `soundevent.data.SoundEventPrediction` using
    `convert_raw_prediction_to_sound_event_prediction`, and packages them
    into a `soundevent.data.ClipPrediction` associated with the original `clip`.

    Parameters
    ----------
    raw_predictions : List[RawPrediction]
        List of raw prediction objects for a single clip.
    clip : data.Clip
        The original `soundevent.data.Clip` object these predictions belong to.
    sound_event_decoder : SoundEventDecoder
        Function to decode class names into representative tags.
    generic_class_tags : List[data.Tag]
        List of tags representing the generic class category.
    classification_threshold : float, default=DEFAULT_CLASSIFICATION_THRESHOLD
        Threshold applied to class scores during decoding.
    top_class_only : bool, default=False
        If True, only decode tags for the single highest-scoring class above
        the threshold. If False, decode tags for all classes above threshold.

    Returns
    -------
    data.ClipPrediction
        A `ClipPrediction` object containing a list of `SoundEventPrediction`
        objects corresponding to the input `raw_predictions`.
    """
    return data.ClipPrediction(
        clip=clip,
        sound_events=[
            convert_raw_prediction_to_sound_event_prediction(
                prediction,
                recording=clip.recording,
                sound_event_decoder=sound_event_decoder,
                generic_class_tags=generic_class_tags,
                classification_threshold=classification_threshold,
                top_class_only=top_class_only,
            )
            for prediction in raw_predictions
        ],
    )


def convert_raw_prediction_to_sound_event_prediction(
    raw_prediction: RawPrediction,
    recording: data.Recording,
    sound_event_decoder: SoundEventDecoder,
    generic_class_tags: List[data.Tag],
    classification_threshold: Optional[
        float
    ] = DEFAULT_CLASSIFICATION_THRESHOLD,
    top_class_only: bool = False,
):
    """Convert a single RawPrediction into a soundevent SoundEventPrediction.

    This function performs the core decoding steps for a single detected event:
    1. Creates a `soundevent.data.SoundEvent` containing the geometry
       (BoundingBox derived from `raw_prediction` bounds) and any associated
       feature vectors.
    2. Initializes a list of predicted tags using the provided
       `generic_class_tags`, assigning the overall `detection_score` from the
       `raw_prediction` to these generic tags.
    3. Processes the `class_scores` from the `raw_prediction`:
        a. Optionally filters out scores below `classification_threshold`
           (if it's not None).
        b. Sorts the remaining scores in descending order.
        c. Iterates through the sorted, thresholded class scores.
        d. For each class, uses the `sound_event_decoder` to get the
           representative base tags for that class name.
        e. Wraps these base tags in `soundevent.data.PredictedTag`, associating
           the specific `score` of that class prediction.
        f. Appends these specific predicted tags to the list.
        g. If `top_class_only` is True, stops after processing the first
           (highest-scoring) class that passed the threshold.
    4. Creates and returns the final `soundevent.data.SoundEventPrediction`,
       associating the `SoundEvent`, the overall `detection_score`, and the
       compiled list of `PredictedTag` objects.

    Parameters
    ----------
    raw_prediction : RawPrediction
        The raw prediction object containing score, bounds, class scores,
        features. Assumes `class_scores` is an `xr.DataArray` with a 'category'
        coordinate. Assumes `features` is an `xr.DataArray` with a 'feature'
        coordinate.
    recording : data.Recording
        The recording the sound event belongs to.
    sound_event_decoder : SoundEventDecoder
        Configured function mapping class names (str) to lists of base
        `data.Tag` objects.
    generic_class_tags : List[data.Tag]
        List of base tags representing the generic category.
    classification_threshold : float, optional
        The minimum score a class prediction must have to be considered
        significant enough to have its tags decoded and added. If None, no
        thresholding is applied based on class score (all predicted classes,
        or the top one if `top_class_only` is True, will be processed).
        Defaults to `DEFAULT_CLASSIFICATION_THRESHOLD`.
    top_class_only : bool, default=False
        If True, only includes tags for the single highest-scoring class that
        exceeds the threshold. If False (default), includes tags for all classes
        exceeding the threshold.

    Returns
    -------
    data.SoundEventPrediction
        The fully formed sound event prediction object.

    Raises
    ------
    ValueError
        If `raw_prediction.features` has unexpected structure or if
        `data.term_from_key` (if used internally) fails.
        If `sound_event_decoder` fails for a class name and errors are raised.
    """
    sound_event = data.SoundEvent(
        recording=recording,
        geometry=data.BoundingBox(
            coordinates=[
                raw_prediction.start_time,
                raw_prediction.low_freq,
                raw_prediction.end_time,
                raw_prediction.high_freq,
            ]
        ),
        features=[
            data.Feature(term=data.term_from_key(feat_name), value=value)
            for feat_name, value in raw_prediction.features
        ],
    )

    tags = [
        data.PredictedTag(tag=tag, score=raw_prediction.detection_score)
        for tag in generic_class_tags
    ]

    class_scores = raw_prediction.class_scores

    if classification_threshold is not None:
        class_scores = class_scores.where(
            class_scores > classification_threshold,
            drop=True,
        )

    for class_name, score in class_scores.sortby(
        class_scores, ascending=False
    ):
        class_tags = sound_event_decoder(class_name)

        for tag in class_tags:
            tags.append(
                data.PredictedTag(
                    tag=tag,
                    score=score,
                )
            )

        if top_class_only:
            break

    return data.SoundEventPrediction(
        sound_event=sound_event,
        score=raw_prediction.detection_score,
        tags=tags,
    )
