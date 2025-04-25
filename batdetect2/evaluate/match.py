from typing import List

from soundevent import data
from soundevent.evaluation import match_geometries

from batdetect2.evaluate.types import Match
from batdetect2.postprocess.types import RawPrediction
from batdetect2.targets.types import TargetProtocol
from batdetect2.utils.arrays import iterate_over_array


def match_sound_events_and_raw_predictions(
    sound_events: List[data.SoundEventAnnotation],
    raw_predictions: List[RawPrediction],
    targets: TargetProtocol,
) -> List[Match]:
    target_sound_events = [
        targets.transform(sound_event_annotation)
        for sound_event_annotation in sound_events
        if targets.filter(sound_event_annotation)
        and sound_event_annotation.sound_event.geometry is not None
    ]

    target_geometries: List[data.Geometry] = [  # type: ignore
        sound_event_annotation.sound_event.geometry
        for sound_event_annotation in target_sound_events
    ]

    predicted_geometries = [
        raw_prediction.geometry for raw_prediction in raw_predictions
    ]

    matches = []
    for id1, id2, affinity in match_geometries(
        target_geometries,
        predicted_geometries,
    ):
        target = target_sound_events[id1] if id1 is not None else None
        prediction = raw_predictions[id2] if id2 is not None else None

        gt_uuid = target.uuid if target is not None else None
        gt_det = target is not None
        gt_class = targets.encode(target) if target is not None else None

        pred_score = float(prediction.detection_score) if prediction else 0

        class_scores = (
            {
                str(class_name): float(score)
                for class_name, score in iterate_over_array(
                    prediction.class_scores
                )
            }
            if prediction is not None
            else {}
        )

        matches.append(
            Match(
                gt_uuid=gt_uuid,
                gt_det=gt_det,
                gt_class=gt_class,
                pred_score=pred_score,
                affinity=affinity,
                class_scores=class_scores,
            )
        )

    return matches


def match_predictions_and_annotations(
    clip_annotation: data.ClipAnnotation,
    clip_prediction: data.ClipPrediction,
) -> List[data.Match]:
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
        sound_event.sound_event.geometry
        for sound_event in annotated_sound_events
        if sound_event.sound_event.geometry is not None
    ]

    predicted_geometries: List[data.Geometry] = [
        sound_event.sound_event.geometry
        for sound_event in predicted_sound_events
        if sound_event.sound_event.geometry is not None
    ]

    matches = []
    for id1, id2, affinity in match_geometries(
        annotated_geometries,
        predicted_geometries,
    ):
        target = annotated_sound_events[id1] if id1 is not None else None
        source = predicted_sound_events[id2] if id2 is not None else None
        matches.append(
            data.Match(source=source, target=target, affinity=affinity)
        )

    return matches
