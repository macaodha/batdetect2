from collections.abc import Callable, Iterable, Mapping
from typing import Annotated, List, Literal, Sequence, Tuple

import numpy as np
from pydantic import Field
from scipy.optimize import linear_sum_assignment
from soundevent import data
from soundevent.evaluation import compute_affinity
from soundevent.geometry import buffer_geometry, compute_bounds, scale_geometry

from batdetect2.core import BaseConfig, Registry
from batdetect2.evaluate.affinity import (
    AffinityConfig,
    BBoxIOUConfig,
    GeometricIOUConfig,
    build_affinity_function,
)
from batdetect2.targets import build_targets
from batdetect2.typing import (
    AffinityFunction,
    MatcherProtocol,
    MatchEvaluation,
    RawPrediction,
    TargetProtocol,
)
from batdetect2.typing.evaluate import ClipMatches

MatchingGeometry = Literal["bbox", "interval", "timestamp"]
"""The geometry representation to use for matching."""

matching_strategies = Registry("matching_strategy")


class StartTimeMatchConfig(BaseConfig):
    name: Literal["start_time_match"] = "start_time_match"
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

    @matching_strategies.register(StartTimeMatchConfig)
    @staticmethod
    def from_config(config: StartTimeMatchConfig):
        return StartTimeMatcher(distance_threshold=config.distance_threshold)


def match_start_times(
    ground_truth: Sequence[data.Geometry],
    predictions: Sequence[data.Geometry],
    scores: Sequence[float],
    distance_threshold: float = 0.01,
) -> Iterable[Tuple[int | None, int | None, float]]:
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

    @matching_strategies.register(GreedyMatchConfig)
    @staticmethod
    def from_config(config: GreedyMatchConfig):
        affinity_function = build_affinity_function(config.affinity_function)
        return GreedyMatcher(
            geometry=config.geometry,
            affinity_threshold=config.affinity_threshold,
            affinity_function=affinity_function,
        )


def greedy_match(
    ground_truth: Sequence[data.Geometry],
    predictions: Sequence[data.Geometry],
    scores: Sequence[float],
    affinity_threshold: float = 0.5,
    affinity_function: AffinityFunction = compute_affinity,
) -> Iterable[Tuple[int | None, int | None, float]]:
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
        for gt_idx in range(len(ground_truth)):
            yield None, gt_idx, 0

        return

    if not ground_truth:
        for pred_idx in range(len(predictions)):
            yield pred_idx, None, 0

        return

    indices = np.argsort(scores)[::-1]

    for pred_idx in indices:
        source_geometry = predictions[pred_idx]

        affinities = np.array(
            [
                affinity_function(source_geometry, target_geometry)
                for target_geometry in ground_truth
            ]
        )

        closest_target = int(np.argmax(affinities))
        affinity = affinities[closest_target]

        if affinities[closest_target] <= affinity_threshold:
            yield pred_idx, None, 0
            continue

        if closest_target not in unassigned_gt:
            yield pred_idx, None, 0
            continue

        unassigned_gt.remove(closest_target)
        yield pred_idx, closest_target, affinity

    for gt_idx in unassigned_gt:
        yield None, gt_idx, 0


class GreedyAffinityMatchConfig(BaseConfig):
    name: Literal["greedy_affinity_match"] = "greedy_affinity_match"
    affinity_function: AffinityConfig = Field(default_factory=BBoxIOUConfig)
    affinity_threshold: float = 0.5
    time_buffer: float = 0
    frequency_buffer: float = 0
    time_scale: float = 1.0
    frequency_scale: float = 1.0


class GreedyAffinityMatcher(MatcherProtocol):
    def __init__(
        self,
        affinity_threshold: float,
        affinity_function: AffinityFunction,
        time_buffer: float = 0,
        frequency_buffer: float = 0,
        time_scale: float = 1.0,
        frequency_scale: float = 1.0,
    ):
        self.affinity_threshold = affinity_threshold
        self.affinity_function = affinity_function
        self.time_buffer = time_buffer
        self.frequency_buffer = frequency_buffer
        self.time_scale = time_scale
        self.frequency_scale = frequency_scale

    def __call__(
        self,
        ground_truth: Sequence[data.Geometry],
        predictions: Sequence[data.Geometry],
        scores: Sequence[float],
    ):
        if self.time_buffer != 0 or self.frequency_buffer != 0:
            ground_truth = [
                buffer_geometry(
                    geometry,
                    time_buffer=self.time_buffer,
                    freq_buffer=self.frequency_buffer,
                )
                for geometry in ground_truth
            ]

            predictions = [
                buffer_geometry(
                    geometry,
                    time_buffer=self.time_buffer,
                    freq_buffer=self.frequency_buffer,
                )
                for geometry in predictions
            ]

        affinity_matrix = compute_affinity_matrix(
            ground_truth,
            predictions,
            self.affinity_function,
            time_scale=self.time_scale,
            frequency_scale=self.frequency_scale,
        )

        return select_greedy_matches(
            affinity_matrix,
            affinity_threshold=self.affinity_threshold,
        )

    @matching_strategies.register(GreedyAffinityMatchConfig)
    @staticmethod
    def from_config(config: GreedyAffinityMatchConfig):
        affinity_function = build_affinity_function(config.affinity_function)
        return GreedyAffinityMatcher(
            affinity_threshold=config.affinity_threshold,
            affinity_function=affinity_function,
            time_scale=config.time_scale,
            frequency_scale=config.frequency_scale,
        )


class OptimalMatchConfig(BaseConfig):
    name: Literal["optimal_affinity_match"] = "optimal_affinity_match"
    affinity_function: AffinityConfig = Field(default_factory=BBoxIOUConfig)
    affinity_threshold: float = 0.5
    time_buffer: float = 0
    frequency_buffer: float = 0
    time_scale: float = 1.0
    frequency_scale: float = 1.0


class OptimalMatcher(MatcherProtocol):
    def __init__(
        self,
        affinity_threshold: float,
        affinity_function: AffinityFunction,
        time_buffer: float = 0,
        frequency_buffer: float = 0,
        time_scale: float = 1.0,
        frequency_scale: float = 1.0,
    ):
        self.affinity_threshold = affinity_threshold
        self.affinity_function = affinity_function
        self.time_buffer = time_buffer
        self.frequency_buffer = frequency_buffer
        self.time_scale = time_scale
        self.frequency_scale = frequency_scale

    def __call__(
        self,
        ground_truth: Sequence[data.Geometry],
        predictions: Sequence[data.Geometry],
        scores: Sequence[float],
    ):
        if self.time_buffer != 0 or self.frequency_buffer != 0:
            ground_truth = [
                buffer_geometry(
                    geometry,
                    time_buffer=self.time_buffer,
                    freq_buffer=self.frequency_buffer,
                )
                for geometry in ground_truth
            ]

            predictions = [
                buffer_geometry(
                    geometry,
                    time_buffer=self.time_buffer,
                    freq_buffer=self.frequency_buffer,
                )
                for geometry in predictions
            ]

        affinity_matrix = compute_affinity_matrix(
            ground_truth,
            predictions,
            self.affinity_function,
            time_scale=self.time_scale,
            frequency_scale=self.frequency_scale,
        )
        return select_optimal_matches(
            affinity_matrix,
            affinity_threshold=self.affinity_threshold,
        )

    @matching_strategies.register(OptimalMatchConfig)
    @staticmethod
    def from_config(config: OptimalMatchConfig):
        affinity_function = build_affinity_function(config.affinity_function)
        return OptimalMatcher(
            affinity_threshold=config.affinity_threshold,
            affinity_function=affinity_function,
            time_buffer=config.time_buffer,
            frequency_buffer=config.frequency_buffer,
            time_scale=config.time_scale,
            frequency_scale=config.frequency_scale,
        )


MatchConfig = Annotated[
    GreedyMatchConfig
    | StartTimeMatchConfig
    | OptimalMatchConfig
    | GreedyAffinityMatchConfig,
    Field(discriminator="name"),
]
