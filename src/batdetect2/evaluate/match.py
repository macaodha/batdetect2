from typing import Annotated, List, Literal, Optional, Union

from pydantic import Field
from soundevent import data
from soundevent.evaluation import match_geometries
from soundevent.geometry import compute_bounds

from batdetect2.configs import BaseConfig
from batdetect2.evaluate.types import MatchEvaluation
from batdetect2.postprocess.types import BatDetect2Prediction
from batdetect2.targets.types import TargetProtocol
from batdetect2.utils.arrays import iterate_over_array


class BBoxMatchConfig(BaseConfig):
    match_method: Literal["BBoxIOU"] = "BBoxIOU"
    affinity_threshold: float = 0.5
    time_buffer: float = 0.01
    frequency_buffer: float = 1_000


class IntervalMatchConfig(BaseConfig):
    match_method: Literal["IntervalIOU"] = "IntervalIOU"
    affinity_threshold: float = 0.5
    time_buffer: float = 0.01


class StartTimeMatchConfig(BaseConfig):
    match_method: Literal["StartTime"] = "StartTime"
    time_buffer: float = 0.01


MatchConfig = Annotated[
    Union[BBoxMatchConfig, IntervalMatchConfig, StartTimeMatchConfig],
    Field(discriminator="match_method"),
]


DEFAULT_MATCH_CONFIG = BBoxMatchConfig()


def prepare_geometry(
    geometry: data.Geometry, config: MatchConfig
) -> data.Geometry:
    start_time, low_freq, end_time, high_freq = compute_bounds(geometry)

    if config.match_method == "BBoxIOU":
        return data.BoundingBox(
            coordinates=[start_time, low_freq, end_time, high_freq]
        )

    if config.match_method == "IntervalIOU":
        return data.TimeInterval(coordinates=[start_time, end_time])

    if config.match_method == "StartTime":
        return data.TimeStamp(coordinates=start_time)

    raise NotImplementedError(
        f"Invalid matching configuration. Unknown match method: {config.match_method}"
    )


def _get_frequency_buffer(config: MatchConfig) -> float:
    if config.match_method == "BBoxIOU":
        return config.frequency_buffer

    return 0


def _get_affinity_threshold(config: MatchConfig) -> float:
    if (
        config.match_method == "BBoxIOU"
        or config.match_method == "IntervalIOU"
    ):
        return config.affinity_threshold

    return 0


def match_sound_events_and_raw_predictions(
    clip_annotation: data.ClipAnnotation,
    raw_predictions: List[BatDetect2Prediction],
    targets: TargetProtocol,
    config: Optional[MatchConfig] = None,
) -> List[MatchEvaluation]:
    config = config or DEFAULT_MATCH_CONFIG

    target_sound_events = [
        targets.transform(sound_event_annotation)
        for sound_event_annotation in clip_annotation.sound_events
        if targets.filter(sound_event_annotation)
        and sound_event_annotation.sound_event.geometry is not None
    ]

    target_geometries: List[data.Geometry] = [  # type: ignore
        prepare_geometry(
            sound_event_annotation.sound_event.geometry,
            config=config,
        )
        for sound_event_annotation in target_sound_events
        if sound_event_annotation.sound_event.geometry is not None
    ]

    predicted_geometries = [
        prepare_geometry(raw_prediction.raw.geometry, config=config)
        for raw_prediction in raw_predictions
    ]

    matches = []

    for id1, id2, affinity in match_geometries(
        target_geometries,
        predicted_geometries,
        time_buffer=config.time_buffer,
        freq_buffer=_get_frequency_buffer(config),
        affinity_threshold=_get_affinity_threshold(config),
    ):
        target = target_sound_events[id1] if id1 is not None else None
        prediction = raw_predictions[id2] if id2 is not None else None

        gt_det = target is not None
        gt_class = targets.encode_class(target) if target is not None else None

        pred_score = float(prediction.raw.detection_score) if prediction else 0

        class_scores = (
            {
                str(class_name): float(score)
                for class_name, score in iterate_over_array(
                    prediction.raw.class_scores
                )
            }
            if prediction is not None
            else {}
        )

        matches.append(
            MatchEvaluation(
                match=data.Match(
                    source=None
                    if prediction is None
                    else prediction.sound_event_prediction,
                    target=target,
                    affinity=affinity,
                ),
                gt_det=gt_det,
                gt_class=gt_class,
                pred_score=pred_score,
                pred_class_scores=class_scores,
            )
        )

    return matches


def match_predictions_and_annotations(
    clip_annotation: data.ClipAnnotation,
    clip_prediction: data.ClipPrediction,
    config: Optional[MatchConfig] = None,
) -> List[data.Match]:
    config = config or DEFAULT_MATCH_CONFIG

    annotated_sound_events = [
        sound_event_annotation
        for sound_event_annotation in clip_annotation.sound_events
        if sound_event_annotation.sound_event.geometry is not None
    ]

    predicted_sound_events = [
        sound_event_prediction
        for sound_event_prediction in clip_prediction.sound_events
        if sound_event_prediction.sound_event.geometry is not None
    ]

    annotated_geometries: List[data.Geometry] = [
        prepare_geometry(sound_event.sound_event.geometry, config=config)
        for sound_event in annotated_sound_events
        if sound_event.sound_event.geometry is not None
    ]

    predicted_geometries: List[data.Geometry] = [
        prepare_geometry(sound_event.sound_event.geometry, config=config)
        for sound_event in predicted_sound_events
        if sound_event.sound_event.geometry is not None
    ]

    matches = []
    for id1, id2, affinity in match_geometries(
        annotated_geometries,
        predicted_geometries,
        time_buffer=config.time_buffer,
        freq_buffer=_get_frequency_buffer(config),
        affinity_threshold=_get_affinity_threshold(config),
    ):
        target = annotated_sound_events[id1] if id1 is not None else None
        source = predicted_sound_events[id2] if id2 is not None else None
        matches.append(
            data.Match(source=source, target=target, affinity=affinity)
        )

    return matches
