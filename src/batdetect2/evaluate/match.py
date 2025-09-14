from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Annotated, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from loguru import logger
from pydantic import Field
from soundevent import data
from soundevent.evaluation import compute_affinity
from soundevent.evaluation import match_geometries as optimal_match
from soundevent.geometry import compute_bounds

from batdetect2.configs import BaseConfig
from batdetect2.data._core import Registry
from batdetect2.targets import build_targets
from batdetect2.typing import (
    MatchEvaluation,
    TargetProtocol,
)
from batdetect2.typing.evaluate import AffinityFunction, MatcherProtocol
from batdetect2.typing.postprocess import RawPrediction

MatchingGeometry = Literal["bbox", "interval", "timestamp"]
"""The geometry representation to use for matching."""

matching_strategy = Registry("matching_strategy")


class StartTimeMatchConfig(BaseConfig):
    name: Literal["start_time"] = "start_time"
    distance_threshold: float = 0.01


@matching_strategy.register(StartTimeMatchConfig)
class StartTimeMatcher(MatcherProtocol):
    def __init__(self, distance_threshold: float):
        self.distance_threshold = distance_threshold

    def __call__(
        self,
        ground_truth: Sequence[data.Geometry],
        predictions: Sequence[data.Geometry],
        scores: Sequence[float],
    ):
        return match_start_times(
            ground_truth,
            predictions,
            scores,
            distance_threshold=self.distance_threshold,
        )

    @classmethod
    def from_config(cls, config: StartTimeMatchConfig) -> "StartTimeMatcher":
        return cls(distance_threshold=config.distance_threshold)


def match_start_times(
    ground_truth: Sequence[data.Geometry],
    predictions: Sequence[data.Geometry],
    scores: Sequence[float],
    distance_threshold: float = 0.01,
) -> Iterable[Tuple[Optional[int], Optional[int], float]]:
    if not ground_truth:
        for index in range(len(predictions)):
            yield index, None, 0

        return

    if not predictions:
        for index in range(len(ground_truth)):
            yield None, index, 0

        return

    gt_times = np.array([compute_bounds(geom)[0] for geom in ground_truth])
    pred_times = np.array([compute_bounds(geom)[0] for geom in predictions])
    scores = np.array(scores)

    sort_args = np.argsort(scores)[::-1]

    distances = np.abs(gt_times[None, :] - pred_times[:, None])
    closests = np.argmin(distances, axis=-1)

    unmatched_gt = set(range(len(gt_times)))

    for pred_index in sort_args:
        # Get the closest ground truth
        gt_closest_index = closests[pred_index]

        if gt_closest_index not in unmatched_gt:
            # Does not match if closest has been assigned
            yield pred_index, None, 0
            continue

        # Get the actual distance
        distance = distances[pred_index, gt_closest_index]

        if distance > distance_threshold:
            # Does not match if too far from closest
            yield pred_index, None, 0
            continue

        # Return affinity value: linear interpolation between 0 to 1, where a
        # distance at the threshold maps to 0 affinity and a zero distance maps
        # to 1.
        affinity = np.interp(
            distance,
            [0, distance_threshold],
            [1, 0],
            left=1,
            right=0,
        )
        unmatched_gt.remove(gt_closest_index)
        yield pred_index, gt_closest_index, affinity

    for missing_index in unmatched_gt:
        yield None, missing_index, 0


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


def _timestamp_affinity(
    geometry1: data.Geometry,
    geometry2: data.Geometry,
    time_buffer: float = 0.01,
    freq_buffer: float = 1000,
) -> float:
    assert isinstance(geometry1, data.TimeStamp)
    assert isinstance(geometry2, data.TimeStamp)

    start_time1 = geometry1.coordinates
    start_time2 = geometry2.coordinates

    a = min(start_time1, start_time2)
    b = max(start_time1, start_time2)

    if b - a >= 2 * time_buffer:
        return 0

    intersection = a - b + 2 * time_buffer
    union = b - a + 2 * time_buffer
    return intersection / union


def _interval_affinity(
    geometry1: data.Geometry,
    geometry2: data.Geometry,
    time_buffer: float = 0.01,
    freq_buffer: float = 1000,
) -> float:
    assert isinstance(geometry1, data.TimeInterval)
    assert isinstance(geometry2, data.TimeInterval)

    start_time1, end_time1 = geometry1.coordinates
    start_time2, end_time2 = geometry1.coordinates

    start_time1 -= time_buffer
    start_time2 -= time_buffer
    end_time1 += time_buffer
    end_time2 += time_buffer

    intersection = max(
        0, min(end_time1, end_time2) - max(start_time1, start_time2)
    )
    union = (
        (end_time1 - start_time1) + (end_time2 - start_time2) - intersection
    )

    if union == 0:
        return 0

    return intersection / union


_affinity_functions: Mapping[MatchingGeometry, AffinityFunction] = {
    "timestamp": _timestamp_affinity,
    "interval": _interval_affinity,
    "bbox": compute_affinity,
}


class GreedyMatchConfig(BaseConfig):
    name: Literal["greedy_match"] = "greedy_match"
    geometry: MatchingGeometry = "timestamp"
    affinity_threshold: float = 0.0
    time_buffer: float = 0.005
    frequency_buffer: float = 1_000


@matching_strategy.register(GreedyMatchConfig)
class GreedyMatcher(MatcherProtocol):
    def __init__(
        self,
        geometry: MatchingGeometry,
        affinity_threshold: float,
        time_buffer: float,
        frequency_buffer: float,
    ):
        self.geometry = geometry
        self.affinity_threshold = affinity_threshold
        self.time_buffer = time_buffer
        self.frequency_buffer = frequency_buffer

        self.affinity_function = _affinity_functions[self.geometry]
        self.cast_geometry = _geometry_cast_functions[self.geometry]

    def __call__(
        self,
        ground_truth: Sequence[data.Geometry],
        predictions: Sequence[data.Geometry],
        scores: Sequence[float],
    ):
        return greedy_match(
            ground_truth=[self.cast_geometry(geom) for geom in ground_truth],
            predictions=[self.cast_geometry(geom) for geom in predictions],
            scores=scores,
            affinity_function=self.affinity_function,
            affinity_threshold=self.affinity_threshold,
            time_buffer=self.time_buffer,
            freq_buffer=self.frequency_buffer,
        )

    @classmethod
    def from_config(cls, config: GreedyMatchConfig):
        return cls(
            geometry=config.geometry,
            affinity_threshold=config.affinity_threshold,
            time_buffer=config.time_buffer,
            frequency_buffer=config.frequency_buffer,
        )


def greedy_match(
    ground_truth: Sequence[data.Geometry],
    predictions: Sequence[data.Geometry],
    scores: Sequence[float],
    affinity_threshold: float = 0.5,
    affinity_function: AffinityFunction = compute_affinity,
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
    unassigned_gt = set(range(len(ground_truth)))

    if not predictions:
        for target_idx in range(len(ground_truth)):
            yield None, target_idx, 0

        return

    if not ground_truth:
        for source_idx in range(len(predictions)):
            yield source_idx, None, 0

        return

    indices = np.argsort(scores)[::-1]

    for source_idx in indices:
        source_geometry = predictions[source_idx]

        affinities = np.array(
            [
                affinity_function(
                    source_geometry,
                    target_geometry,
                    time_buffer=time_buffer,
                    freq_buffer=freq_buffer,
                )
                for target_geometry in ground_truth
            ]
        )

        closest_target = int(np.argmax(affinities))
        affinity = affinities[closest_target]

        if affinities[closest_target] <= affinity_threshold:
            yield source_idx, None, 0
            continue

        if closest_target not in unassigned_gt:
            yield source_idx, None, 0
            continue

        unassigned_gt.remove(closest_target)
        yield source_idx, closest_target, affinity

    for target_idx in unassigned_gt:
        yield None, target_idx, 0


class OptimalMatchConfig(BaseConfig):
    name: Literal["optimal_match"] = "optimal_match"
    affinity_threshold: float = 0.0
    time_buffer: float = 0.005
    frequency_buffer: float = 1_000


@matching_strategy.register(OptimalMatchConfig)
class OptimalMatcher(MatcherProtocol):
    def __init__(
        self,
        affinity_threshold: float,
        time_buffer: float,
        frequency_buffer: float,
    ):
        self.affinity_threshold = affinity_threshold
        self.time_buffer = time_buffer
        self.frequency_buffer = frequency_buffer

    def __call__(
        self,
        ground_truth: Sequence[data.Geometry],
        predictions: Sequence[data.Geometry],
        scores: Sequence[float],
    ):
        return optimal_match(
            source=predictions,
            target=ground_truth,
            time_buffer=self.time_buffer,
            freq_buffer=self.frequency_buffer,
            affinity_threshold=self.affinity_threshold,
        )

    @classmethod
    def from_config(cls, config: OptimalMatchConfig):
        return cls(
            affinity_threshold=config.affinity_threshold,
            time_buffer=config.time_buffer,
            frequency_buffer=config.frequency_buffer,
        )


MatchConfig = Annotated[
    Union[
        GreedyMatchConfig,
        StartTimeMatchConfig,
        OptimalMatchConfig,
    ],
    Field(discriminator="name"),
]


def build_matcher(config: Optional[MatchConfig] = None) -> MatcherProtocol:
    config = config or StartTimeMatchConfig()
    return matching_strategy.build(config)


def _is_in_bounds(
    geometry: data.Geometry,
    clip: data.Clip,
    buffer: float,
) -> bool:
    start_time = compute_bounds(geometry)[0]
    return (start_time >= clip.start_time + buffer) and (
        start_time <= clip.end_time - buffer
    )


def match_sound_events_and_predictions(
    clip_annotation: data.ClipAnnotation,
    raw_predictions: List[RawPrediction],
    targets: Optional[TargetProtocol] = None,
    matcher: Optional[MatcherProtocol] = None,
    ignore_start_end: float = 0.01,
) -> List[MatchEvaluation]:
    if matcher is None:
        matcher = build_matcher()

    if targets is None:
        targets = build_targets()

    target_sound_events = [
        sound_event_annotation
        for sound_event_annotation in clip_annotation.sound_events
        if targets.filter(sound_event_annotation)
        and sound_event_annotation.sound_event.geometry is not None
        and _is_in_bounds(
            sound_event_annotation.sound_event.geometry,
            clip=clip_annotation.clip,
            buffer=ignore_start_end,
        )
    ]

    target_geometries: List[data.Geometry] = [
        sound_event_annotation.sound_event.geometry
        for sound_event_annotation in target_sound_events
        if sound_event_annotation.sound_event.geometry is not None
    ]

    raw_predictions = [
        raw_prediction
        for raw_prediction in raw_predictions
        if _is_in_bounds(
            raw_prediction.geometry,
            clip=clip_annotation.clip,
            buffer=ignore_start_end,
        )
    ]

    predicted_geometries = [
        raw_prediction.geometry for raw_prediction in raw_predictions
    ]

    scores = [
        raw_prediction.detection_score for raw_prediction in raw_predictions
    ]

    matches = []

    for source_idx, target_idx, affinity in matcher(
        ground_truth=target_geometries,
        predictions=predicted_geometries,
        scores=scores,
    ):
        target = (
            target_sound_events[target_idx] if target_idx is not None else None
        )
        prediction = (
            raw_predictions[source_idx] if source_idx is not None else None
        )

        gt_det = target_idx is not None
        gt_class = targets.encode_class(target) if target is not None else None

        pred_score = float(prediction.detection_score) if prediction else 0

        pred_geometry = (
            predicted_geometries[source_idx]
            if source_idx is not None
            else None
        )

        class_scores = (
            {
                str(class_name): float(score)
                for class_name, score in zip(
                    targets.class_names,
                    prediction.class_scores,
                )
            }
            if prediction is not None
            else {}
        )

        matches.append(
            MatchEvaluation(
                clip=clip_annotation.clip,
                sound_event_annotation=target,
                gt_det=gt_det,
                gt_class=gt_class,
                pred_score=pred_score,
                pred_class_scores=class_scores,
                pred_geometry=pred_geometry,
                affinity=affinity,
            )
        )

    return matches


def match_all_predictions(
    clip_annotations: List[data.ClipAnnotation],
    predictions: List[List[RawPrediction]],
    targets: Optional[TargetProtocol] = None,
    matcher: Optional[MatcherProtocol] = None,
    ignore_start_end: float = 0.01,
) -> List[MatchEvaluation]:
    logger.info("Matching all annotations and predictions...")
    return [
        match
        for clip_annotation, raw_predictions in zip(
            clip_annotations,
            predictions,
        )
        for match in match_sound_events_and_predictions(
            clip_annotation,
            raw_predictions,
            targets=targets,
            matcher=matcher,
            ignore_start_end=ignore_start_end,
        )
    ]


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
