from collections.abc import Callable, Iterable, Mapping
from typing import Annotated, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import Field
from soundevent import data
from soundevent.evaluation import compute_affinity
from soundevent.evaluation import match_geometries as optimal_match
from soundevent.geometry import compute_bounds

from batdetect2.configs import BaseConfig
from batdetect2.data._core import Registry
from batdetect2.evaluate.affinity import (
    AffinityConfig,
    GeometricIOUConfig,
    build_affinity_function,
)
from batdetect2.targets import build_targets
from batdetect2.typing import (
    MatchEvaluation,
    TargetProtocol,
)
from batdetect2.typing.evaluate import AffinityFunction, MatcherProtocol
from batdetect2.typing.postprocess import RawPrediction

MatchingGeometry = Literal["bbox", "interval", "timestamp"]
"""The geometry representation to use for matching."""

matching_strategies = Registry("matching_strategy")


def match(
    sound_event_annotations: Sequence[data.SoundEventAnnotation],
    raw_predictions: Sequence[RawPrediction],
    clip: data.Clip,
    targets: Optional[TargetProtocol] = None,
    matcher: Optional[MatcherProtocol] = None,
) -> List[MatchEvaluation]:
    if matcher is None:
        matcher = build_matcher()

    if targets is None:
        targets = build_targets()

    target_geometries: List[data.Geometry] = [  # type: ignore
        sound_event_annotation.sound_event.geometry
        for sound_event_annotation in sound_event_annotations
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
            sound_event_annotations[target_idx]
            if target_idx is not None
            else None
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
                clip=clip,
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


class StartTimeMatchConfig(BaseConfig):
    name: Literal["start_time"] = "start_time"
    distance_threshold: float = 0.01


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


matching_strategies.register(StartTimeMatchConfig, StartTimeMatcher)


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


class GreedyMatchConfig(BaseConfig):
    name: Literal["greedy_match"] = "greedy_match"
    geometry: MatchingGeometry = "timestamp"
    affinity_threshold: float = 0.5
    affinity_function: AffinityConfig = Field(
        default_factory=GeometricIOUConfig
    )


class GreedyMatcher(MatcherProtocol):
    def __init__(
        self,
        geometry: MatchingGeometry,
        affinity_threshold: float,
        affinity_function: AffinityFunction,
    ):
        self.geometry = geometry
        self.affinity_function = affinity_function
        self.affinity_threshold = affinity_threshold
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
        )

    @classmethod
    def from_config(cls, config: GreedyMatchConfig):
        affinity_function = build_affinity_function(config.affinity_function)
        return cls(
            geometry=config.geometry,
            affinity_threshold=config.affinity_threshold,
            affinity_function=affinity_function,
        )


matching_strategies.register(GreedyMatchConfig, GreedyMatcher)


def greedy_match(
    ground_truth: Sequence[data.Geometry],
    predictions: Sequence[data.Geometry],
    scores: Sequence[float],
    affinity_threshold: float = 0.5,
    affinity_function: AffinityFunction = compute_affinity,
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
                affinity_function(source_geometry, target_geometry)
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
    affinity_threshold: float = 0.5
    time_buffer: float = 0.005
    frequency_buffer: float = 1_000


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


matching_strategies.register(OptimalMatchConfig, OptimalMatcher)

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
    return matching_strategies.build(config)
