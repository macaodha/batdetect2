from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np
from soundevent import data
from soundevent.evaluation import compute_affinity
from soundevent.evaluation import match_geometries as optimal_match
from soundevent.geometry import compute_bounds

from batdetect2.configs import BaseConfig
from batdetect2.typing import (
    BatDetect2Prediction,
    MatchEvaluation,
    TargetProtocol,
)

MatchingStrategy = Literal["greedy", "optimal"]
"""The type of matching algorithm to use: 'greedy' or 'optimal'."""


MatchingGeometry = Literal["bbox", "interval", "timestamp"]
"""The geometry representation to use for matching."""


class MatchConfig(BaseConfig):
    """Configuration for matching geometries.

    Attributes
    ----------
    strategy : MatchingStrategy, default="greedy"
        The matching algorithm to use. 'greedy' prioritizes high-confidence
        predictions, while 'optimal' finds the globally best set of matches.
    geometry : MatchingGeometry, default="timestamp"
        The geometric representation to use when computing affinity.
    affinity_threshold : float, default=0.0
        The minimum affinity score (e.g., IoU) required for a valid match.
    time_buffer : float, default=0.005
        Time tolerance in seconds used in affinity calculations.
    frequency_buffer : float, default=1000
        Frequency tolerance in Hertz used in affinity calculations.
    """

    strategy: MatchingStrategy = "greedy"
    geometry: MatchingGeometry = "timestamp"
    affinity_threshold: float = 0.0
    time_buffer: float = 0.005
    frequency_buffer: float = 1_000


def _to_bbox(geometry: data.Geometry) -> data.BoundingBox:
    start_time, low_freq, end_time, high_freq = compute_bounds(geometry)
    return data.BoundingBox(
        coordinates=[start_time, low_freq, end_time, high_freq]
    )


def _to_interval(geometry: data.Geometry) -> data.TimeInterval:
    start_time, _, end_time, _ = compute_bounds(geometry)
    return data.TimeInterval(coordinates=[start_time, end_time])


def _to_timestamp(geometry: data.Geometry) -> data.TimeStamp:
    start_time = compute_bounds(geometry)[0]
    return data.TimeStamp(coordinates=start_time)


_geometry_cast_functions: Mapping[
    MatchingGeometry, Callable[[data.Geometry], data.Geometry]
] = {
    "bbox": _to_bbox,
    "interval": _to_interval,
    "timestamp": _to_timestamp,
}


def match_geometries(
    source: List[data.Geometry],
    target: List[data.Geometry],
    config: MatchConfig,
    scores: Optional[List[float]] = None,
) -> Iterable[Tuple[Optional[int], Optional[int], float]]:
    geometry_cast = _geometry_cast_functions[config.geometry]

    if config.strategy == "optimal":
        return optimal_match(
            source=[geometry_cast(geom) for geom in source],
            target=[geometry_cast(geom) for geom in target],
            time_buffer=config.time_buffer,
            freq_buffer=config.frequency_buffer,
            affinity_threshold=config.affinity_threshold,
        )

    if config.strategy == "greedy":
        return greedy_match(
            source=[geometry_cast(geom) for geom in source],
            target=[geometry_cast(geom) for geom in target],
            time_buffer=config.time_buffer,
            freq_buffer=config.frequency_buffer,
            affinity_threshold=config.affinity_threshold,
            scores=scores,
        )

    raise NotImplementedError(
        f"Matching strategy not implemented {config.strategy}"
    )


def greedy_match(
    source: List[data.Geometry],
    target: List[data.Geometry],
    scores: Optional[List[float]] = None,
    affinity_threshold: float = 0.5,
    time_buffer: float = 0.001,
    freq_buffer: float = 1000,
) -> Iterable[Tuple[Optional[int], Optional[int], float]]:
    """Performs a greedy, one-to-one matching of source to target geometries.

    Iterates through source geometries, prioritizing by score if provided. Each
    source is matched to the best available target, provided the affinity
    exceeds the threshold and the target has not already been assigned.

    Parameters
    ----------
    source
        A list of source geometries (e.g., predictions).
    target
        A list of target geometries (e.g., ground truths).
    scores
        Confidence scores for each source geometry for prioritization.
    affinity_threshold
        The minimum affinity score required for a valid match.
    time_buffer
        Time tolerance in seconds for affinity calculation.
    freq_buffer
        Frequency tolerance in Hertz for affinity calculation.

    Yields
    ------
    Tuple[Optional[int], Optional[int], float]
        A 3-element tuple describing a match or a miss. There are three
        possible formats:
        - Successful Match: `(source_idx, target_idx, affinity)`
        - Unmatched Source (False Positive): `(source_idx, None, 0)`
        - Unmatched Target (False Negative): `(None, target_idx, 0)`
    """
    assigned = set()

    if not source:
        for target_idx in range(len(target)):
            yield None, target_idx, 0

        return

    if not target:
        for source_idx in range(len(source)):
            yield source_idx, None, 0

        return

    if scores is None:
        indices = np.arange(len(source))
    else:
        indices = np.argsort(scores)[::-1]

    for source_idx in indices:
        source_geometry = source[source_idx]

        affinities = np.array(
            [
                compute_affinity(
                    source_geometry,
                    target_geometry,
                    time_buffer=time_buffer,
                    freq_buffer=freq_buffer,
                )
                for target_geometry in target
            ]
        )

        closest_target = int(np.argmax(affinities))
        affinity = affinities[closest_target]

        if affinities[closest_target] <= affinity_threshold:
            yield source_idx, None, 0
            continue

        if closest_target in assigned:
            yield source_idx, None, 0
            continue

        assigned.add(closest_target)
        yield source_idx, closest_target, affinity

    missed_ground_truth = set(range(len(target))) - assigned
    for target_idx in missed_ground_truth:
        yield None, target_idx, 0


def match_sound_events_and_raw_predictions(
    clip_annotation: data.ClipAnnotation,
    raw_predictions: List[BatDetect2Prediction],
    targets: TargetProtocol,
    config: Optional[MatchConfig] = None,
) -> List[MatchEvaluation]:
    config = config or MatchConfig()

    target_sound_events = [
        targets.transform(sound_event_annotation)
        for sound_event_annotation in clip_annotation.sound_events
        if targets.filter(sound_event_annotation)
        and sound_event_annotation.sound_event.geometry is not None
    ]

    target_geometries: List[data.Geometry] = [  # type: ignore
        sound_event_annotation.sound_event.geometry
        for sound_event_annotation in target_sound_events
        if sound_event_annotation.sound_event.geometry is not None
    ]

    predicted_geometries = [
        raw_prediction.raw.geometry for raw_prediction in raw_predictions
    ]

    scores = [
        raw_prediction.raw.detection_score
        for raw_prediction in raw_predictions
    ]

    matches = []

    for source_idx, target_idx, affinity in match_geometries(
        source=predicted_geometries,
        target=target_geometries,
        config=config,
        scores=scores,
    ):
        target = (
            target_sound_events[target_idx] if target_idx is not None else None
        )
        prediction = (
            raw_predictions[source_idx] if source_idx is not None else None
        )

        gt_det = target is not None
        gt_class = targets.encode_class(target) if target is not None else None

        pred_score = float(prediction.raw.detection_score) if prediction else 0

        class_scores = (
            {
                str(class_name): float(score)
                for class_name, score in zip(
                    targets.class_names,
                    prediction.raw.class_scores,
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
    config = config or MatchConfig()

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

    scores = [
        sound_event.score
        for sound_event in predicted_sound_events
        if sound_event.sound_event.geometry is not None
    ]

    matches = []
    for source_idx, target_idx, affinity in match_geometries(
        source=predicted_geometries,
        target=annotated_geometries,
        config=config,
        scores=scores,
    ):
        target = (
            annotated_sound_events[target_idx]
            if target_idx is not None
            else None
        )
        source = (
            predicted_sound_events[source_idx]
            if source_idx is not None
            else None
        )
        matches.append(
            data.Match(
                source=source,
                target=target,
                affinity=affinity,
            )
        )

    return matches


@dataclass
class ClassExamples:
    false_positives: List[MatchEvaluation] = field(default_factory=list)
    false_negatives: List[MatchEvaluation] = field(default_factory=list)
    true_positives: List[MatchEvaluation] = field(default_factory=list)
    cross_triggers: List[MatchEvaluation] = field(default_factory=list)


def group_matches(matches: List[MatchEvaluation]) -> ClassExamples:
    class_examples = ClassExamples()

    for match in matches:
        gt_class = match.gt_class
        pred_class = match.pred_class

        if pred_class is None:
            class_examples.false_negatives.append(match)
            continue

        if gt_class is None:
            class_examples.false_positives.append(match)
            continue

        if gt_class != pred_class:
            class_examples.cross_triggers.append(match)
            class_examples.cross_triggers.append(match)
            continue

        class_examples.true_positives.append(match)

    return class_examples
