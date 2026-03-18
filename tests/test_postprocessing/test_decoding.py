from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest
import xarray as xr
from soundevent import data

from batdetect2.outputs.transforms.decoding import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    convert_raw_prediction_to_sound_event_prediction,
    convert_raw_predictions_to_clip_prediction,
    get_class_tags,
    get_generic_tags,
    get_prediction_features,
)
from batdetect2.postprocess.types import Detection
from batdetect2.targets.types import TargetProtocol


@pytest.fixture
def dummy_targets() -> TargetProtocol:
    tag_map = {
        "bat": [
            data.Tag(term=data.term_from_key(key="species"), value="Myotis")
        ],
        "noise": [
            data.Tag(term=data.term_from_key(key="category"), value="noise")
        ],
        "unknown": [
            data.Tag(term=data.term_from_key(key="status"), value="uncertain")
        ],
    }

    class DummyTargets(TargetProtocol):
        class_names = [
            "bat",
            "noise",
            "unknown",
        ]

        dimension_names = ["width", "height"]

        detection_class_tags = [
            data.Tag(
                term=data.term_from_key(key="detector"), value="batdetect2"
            )
        ]

        detection_class_name = "bat"

        def filter(self, sound_event: data.SoundEventAnnotation):
            return True

        def transform(self, sound_event: data.SoundEventAnnotation):
            return sound_event

        def encode_class(
            self, sound_event: data.SoundEventAnnotation
        ) -> Optional[str]:
            return "bat"

        def decode_class(self, class_label: str) -> List[data.Tag]:
            return tag_map.get(class_label.lower(), [])

        def encode_roi(self, sound_event: data.SoundEventAnnotation):
            return np.array([0.0, 0.0]), np.array([0.0, 0.0])

        def decode_roi(
            self,
            position,
            size: np.ndarray,
            class_name: Optional[str] = None,
        ):
            time, freq = position
            width, height = size
            return data.BoundingBox(
                coordinates=[
                    time - width / 2,
                    freq - height / 2,
                    time + width / 2,
                    freq + height / 2,
                ]
            )

    t: TargetProtocol = DummyTargets()
    return t


@pytest.fixture
def sample_recording() -> data.Recording:
    """A sample soundevent Recording."""
    return data.Recording(
        path=Path("/path/to/recording.wav"),
        duration=60.0,
        channels=1,
        samplerate=192000,
    )


@pytest.fixture
def sample_clip(sample_recording) -> data.Clip:
    """A sample soundevent Clip."""
    return data.Clip(
        recording=sample_recording,
        start_time=10.0,
        end_time=20.0,
    )


@pytest.fixture
def sample_detection_dataset() -> xr.Dataset:
    """Creates a sample detection dataset suitable for decoding."""
    expected_times = np.array([20, 10])
    expected_freqs = np.array([300, 200])
    detection_coords = {
        "time": ("detection", expected_times),
        "frequency": ("detection", expected_freqs),
    }

    scores_data = np.array([0.9, 0.8], dtype=np.float64)
    scores = xr.DataArray(
        scores_data,
        coords=detection_coords,
        dims=["detection"],
        name="scores",
    )

    dimensions_data = np.array([[7.0, 16.0], [3.0, 12.0]], dtype=np.float32)
    dimensions = xr.DataArray(
        dimensions_data,
        coords={**detection_coords, "dimension": ["width", "height"]},
        dims=["detection", "dimension"],
        name="dimensions",
    )

    classes_data = np.array(
        [[0.43, 0.85], [0.24, 0.66]],
        dtype=np.float32,
    )
    classes = xr.DataArray(
        classes_data,
        coords={**detection_coords, "category": ["bat", "noise"]},
        dims=["detection", "category"],
        name="classes",
    )

    features_data = np.array(
        [[7.0, 16.0, 25.0, 34.0], [3.0, 12.0, 21.0, 30.0]], dtype=np.float32
    )
    features = xr.DataArray(
        features_data,
        coords={**detection_coords, "feature": ["f0", "f1", "f2", "f3"]},
        dims=["detection", "feature"],
        name="features",
    )

    ds = xr.Dataset(
        {
            "scores": scores,
            "dimensions": dimensions,
            "classes": classes,
            "features": features,
        },
        coords=detection_coords,
    )
    return ds


@pytest.fixture
def empty_detection_dataset() -> xr.Dataset:
    """Creates an empty detection dataset with correct structure."""
    detection_coords = {
        "time": ("detection", np.array([], dtype=np.float64)),
        "frequency": ("detection", np.array([], dtype=np.float64)),
    }
    scores = xr.DataArray(
        np.array([], dtype=np.float64),
        coords=detection_coords,
        dims=["detection"],
        name="scores",
    )
    dimensions = xr.DataArray(
        np.empty((0, 2), dtype=np.float32),
        coords={**detection_coords, "dimension": ["width", "height"]},
        dims=["detection", "dimension"],
        name="dimensions",
    )
    classes = xr.DataArray(
        np.empty((0, 2), dtype=np.float32),
        coords={**detection_coords, "category": ["bat", "noise"]},
        dims=["detection", "category"],
        name="classes",
    )
    features = xr.DataArray(
        np.empty((0, 4), dtype=np.float32),
        coords={**detection_coords, "feature": ["f0", "f1", "f2", "f3"]},
        dims=["detection", "feature"],
        name="features",
    )
    return xr.Dataset(
        {
            "scores": scores,
            "dimensions": dimensions,
            "classes": classes,
            "features": features,
        },
        coords=detection_coords,
    )


@pytest.fixture
def sample_raw_predictions() -> List[Detection]:
    """Manually crafted RawPrediction objects using the actual type."""

    pred1_classes = xr.DataArray(
        [0.43, 0.85], coords={"category": ["bat", "noise"]}, dims=["category"]
    )
    pred1_features = xr.DataArray(
        [7.0, 16.0, 25.0, 34.0],
        coords={"feature": ["f0", "f1", "f2", "f3"]},
        dims=["feature"],
    )
    pred1 = Detection(
        detection_score=0.9,
        geometry=data.BoundingBox(
            coordinates=[
                20 - 7 / 2,
                300 - 16 / 2,
                20 + 7 / 2,
                300 + 16 / 2,
            ]
        ),
        class_scores=pred1_classes.values,
        features=pred1_features.values,
    )

    pred2_classes = xr.DataArray(
        [0.24, 0.66], coords={"category": ["bat", "noise"]}, dims=["category"]
    )
    pred2_features = xr.DataArray(
        [3.0, 12.0, 21.0, 30.0],
        coords={"feature": ["f0", "f1", "f2", "f3"]},
        dims=["feature"],
    )
    pred2 = Detection(
        detection_score=0.8,
        geometry=data.BoundingBox(
            coordinates=[
                10 - 3 / 2,
                200 - 12 / 2,
                10 + 3 / 2,
                200 + 12 / 2,
            ]
        ),
        class_scores=pred2_classes.values,
        features=pred2_features.values,
    )

    pred3_classes = xr.DataArray(
        [0.05, 0.02], coords={"category": ["bat", "noise"]}, dims=["category"]
    )
    pred3_features = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0],
        coords={"feature": ["f0", "f1", "f2", "f3"]},
        dims=["feature"],
    )
    pred3 = Detection(
        detection_score=0.15,
        geometry=data.BoundingBox(
            coordinates=[
                5.0,
                50.0,
                6.0,
                60.0,
            ]
        ),
        class_scores=pred3_classes.values,
        features=pred3_features.values,
    )
    return [pred1, pred2, pred3]


def test_convert_raw_to_sound_event_basic(
    sample_raw_predictions: List[Detection],
    sample_recording: data.Recording,
    dummy_targets: TargetProtocol,
):
    """Test basic conversion, default threshold, multi-label."""

    raw_pred = sample_raw_predictions[0]

    se_pred = convert_raw_prediction_to_sound_event_prediction(
        raw_prediction=raw_pred,
        recording=sample_recording,
        targets=dummy_targets,
    )

    assert isinstance(se_pred, data.SoundEventPrediction)
    assert se_pred.score == raw_pred.detection_score

    se = se_pred.sound_event
    assert isinstance(se, data.SoundEvent)
    assert se.recording == sample_recording
    assert isinstance(se.geometry, data.BoundingBox)
    assert se.geometry == raw_pred.geometry
    assert len(se.features) == len(raw_pred.features)

    feat_dict = {f.term.name: f.value for f in se.features}
    assert "batdetect2:f0" in feat_dict and isinstance(
        feat_dict["batdetect2:f0"], float
    )
    assert feat_dict["batdetect2:f0"] == 7.0

    generic_tags = dummy_targets.detection_class_tags
    expected_tags = {
        (generic_tags[0].term.name, generic_tags[0].value, 0.9),
        ("category", "noise", 0.85),
        ("dwc:scientificName", "Myotis", 0.43),
    }
    actual_tags = {
        (pt.tag.term.name, pt.tag.value, pt.score) for pt in se_pred.tags
    }
    assert actual_tags == expected_tags


def test_convert_raw_to_sound_event_thresholding(
    sample_raw_predictions: List[Detection],
    sample_recording: data.Recording,
    dummy_targets: TargetProtocol,
):
    """Test effect of classification threshold."""
    raw_pred = sample_raw_predictions[0]
    high_threshold = 0.5

    se_pred = convert_raw_prediction_to_sound_event_prediction(
        raw_prediction=raw_pred,
        recording=sample_recording,
        targets=dummy_targets,
        classification_threshold=high_threshold,
        top_class_only=False,
    )

    generic_tags = dummy_targets.detection_class_tags
    expected_tags = {
        (generic_tags[0].term.name, generic_tags[0].value, 0.9),
        ("category", "noise", 0.85),
    }
    actual_tags = {
        (pt.tag.term.name, pt.tag.value, pt.score) for pt in se_pred.tags
    }
    assert actual_tags == expected_tags


def test_convert_raw_to_sound_event_no_threshold(
    sample_raw_predictions: List[Detection],
    sample_recording: data.Recording,
    dummy_targets: TargetProtocol,
):
    """Test when classification_threshold is None."""
    raw_pred = sample_raw_predictions[2]

    se_pred = convert_raw_prediction_to_sound_event_prediction(
        raw_prediction=raw_pred,
        recording=sample_recording,
        targets=dummy_targets,
        classification_threshold=None,
        top_class_only=False,
    )

    generic_tags = dummy_targets.detection_class_tags
    expected_tags = {
        (generic_tags[0].term.name, generic_tags[0].value, 0.15),
        ("dwc:scientificName", "Myotis", 0.05),
        ("category", "noise", 0.02),
    }
    actual_tags = {
        (pt.tag.term.name, pt.tag.value, pt.score) for pt in se_pred.tags
    }
    assert actual_tags == expected_tags


def test_convert_raw_to_sound_event_top_class(
    sample_raw_predictions: List[Detection],
    sample_recording: data.Recording,
    dummy_targets: TargetProtocol,
):
    """Test top_class_only=True behavior."""
    raw_pred = sample_raw_predictions[0]

    se_pred = convert_raw_prediction_to_sound_event_prediction(
        raw_prediction=raw_pred,
        recording=sample_recording,
        targets=dummy_targets,
        classification_threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
        top_class_only=True,
    )

    generic_tags = dummy_targets.detection_class_tags
    expected_tags = {
        (generic_tags[0].term.name, generic_tags[0].value, 0.9),
        ("category", "noise", 0.85),
    }
    actual_tags = {
        (pt.tag.term.name, pt.tag.value, pt.score) for pt in se_pred.tags
    }
    assert actual_tags == expected_tags


def test_convert_raw_to_sound_event_all_below_threshold(
    sample_raw_predictions: List[Detection],
    sample_recording: data.Recording,
    dummy_targets: TargetProtocol,
):
    """Test when all class scores are below the default threshold."""
    raw_pred = sample_raw_predictions[2]

    se_pred = convert_raw_prediction_to_sound_event_prediction(
        raw_prediction=raw_pred,
        recording=sample_recording,
        targets=dummy_targets,
        classification_threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
        top_class_only=False,
    )

    generic_tags = dummy_targets.detection_class_tags
    expected_tags = {
        (generic_tags[0].term.name, generic_tags[0].value, 0.15),
    }
    actual_tags = {
        (pt.tag.term.name, pt.tag.value, pt.score) for pt in se_pred.tags
    }
    assert actual_tags == expected_tags


def test_convert_raw_list_to_clip_basic(
    sample_raw_predictions: List[Detection],
    sample_clip: data.Clip,
    dummy_targets: TargetProtocol,
):
    """Test converting a list of RawPredictions to a ClipPrediction."""
    clip_pred = convert_raw_predictions_to_clip_prediction(
        raw_predictions=sample_raw_predictions,
        clip=sample_clip,
        targets=dummy_targets,
        classification_threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
        top_class_only=False,
    )

    assert isinstance(clip_pred, data.ClipPrediction)
    assert clip_pred.clip == sample_clip
    assert len(clip_pred.sound_events) == len(sample_raw_predictions)

    assert clip_pred.sound_events[0].score == (
        sample_raw_predictions[0].detection_score
    )
    assert clip_pred.sound_events[1].score == (
        sample_raw_predictions[1].detection_score
    )
    assert clip_pred.sound_events[2].score == (
        sample_raw_predictions[2].detection_score
    )

    se_pred3_tags = {
        (pt.tag.term.name, pt.tag.value, pt.score)
        for pt in clip_pred.sound_events[2].tags
    }
    generic_tags = dummy_targets.detection_class_tags
    expected_tags3 = {
        (generic_tags[0].term.name, generic_tags[0].value, 0.15),
    }
    assert se_pred3_tags == expected_tags3


def test_convert_raw_list_to_clip_empty(sample_clip, dummy_targets):
    """Test converting an empty list of RawPredictions."""
    clip_pred = convert_raw_predictions_to_clip_prediction(
        raw_predictions=[],
        clip=sample_clip,
        targets=dummy_targets,
    )

    assert isinstance(clip_pred, data.ClipPrediction)
    assert clip_pred.clip == sample_clip
    assert len(clip_pred.sound_events) == 0


def test_convert_raw_list_to_clip_passes_args(
    sample_raw_predictions: List[Detection],
    sample_clip: data.Clip,
    dummy_targets: TargetProtocol,
):
    """Test that arguments like top_class_only are passed through."""

    clip_pred = convert_raw_predictions_to_clip_prediction(
        raw_predictions=sample_raw_predictions,
        clip=sample_clip,
        targets=dummy_targets,
        classification_threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
        top_class_only=True,
    )

    assert len(clip_pred.sound_events) == 3

    se_pred1_tags = {
        (pt.tag.term.name, pt.tag.value, pt.score)
        for pt in clip_pred.sound_events[0].tags
    }
    generic_tags = dummy_targets.detection_class_tags
    expected_tags1 = {
        (generic_tags[0].term.name, generic_tags[0].value, 0.9),
        ("category", "noise", 0.85),
    }
    assert se_pred1_tags == expected_tags1


def test_get_generic_tags_basic(dummy_targets: TargetProtocol):
    """Test creation of generic tags with score."""
    detection_score = 0.75
    generic_tags = dummy_targets.detection_class_tags
    predicted_tags = get_generic_tags(
        detection_score=detection_score, generic_class_tags=generic_tags
    )
    assert len(predicted_tags) == len(generic_tags)
    for predicted_tag in predicted_tags:
        assert isinstance(predicted_tag, data.PredictedTag)
        assert predicted_tag.score == detection_score
        assert predicted_tag.tag in generic_tags


def test_get_prediction_features_basic():
    """Test conversion of feature DataArray to list of Features."""
    feature_data = xr.DataArray(
        [1.1, 2.2, 3.3],
        coords={"feature": ["feat1", "feat2", "feat3"]},
        dims=["feature"],
    )
    features = get_prediction_features(feature_data.values)
    assert len(features) == 3
    for feature, feat_name, feat_value in zip(
        features,
        ["f0", "f1", "f2"],
        [1.1, 2.2, 3.3],
        strict=True,
    ):
        assert isinstance(feature, data.Feature)
        assert feature.term.name == f"batdetect2:{feat_name}"
        assert feature.value == feat_value


def test_get_class_tags_basic(dummy_targets):
    """Test creation of class tags based on scores and decoder."""
    class_scores = xr.DataArray(
        [0.6, 0.2, 0.9],
        coords={"category": ["bat", "noise", "unknown"]},
        dims=["category"],
    )
    predicted_tags = get_class_tags(
        class_scores=class_scores.values,
        targets=dummy_targets,
    )
    assert len(predicted_tags) == 3
    tag_values = [pt.tag.value for pt in predicted_tags]
    tag_scores = [pt.score for pt in predicted_tags]

    assert "Myotis" in tag_values
    assert "noise" in tag_values
    assert "uncertain" in tag_values
    assert 0.6 in tag_scores
    assert 0.2 in tag_scores
    assert 0.9 in tag_scores


def test_get_class_tags_thresholding(dummy_targets):
    """Test class tag creation with a threshold."""
    class_scores = xr.DataArray(
        [0.6, 0.2, 0.9],
        coords={"category": ["bat", "noise", "unknown"]},
        dims=["category"],
    )
    threshold = 0.5
    predicted_tags = get_class_tags(
        class_scores=class_scores.values,
        targets=dummy_targets,
        threshold=threshold,
    )

    assert len(predicted_tags) == 2
    tag_values = [pt.tag.value for pt in predicted_tags]
    assert "Myotis" in tag_values
    assert "noise" not in tag_values
    assert "uncertain" in tag_values


def test_get_class_tags_top_class_only(dummy_targets):
    """Test class tag creation with top_class_only."""
    class_scores = xr.DataArray(
        [0.6, 0.2, 0.9],
        coords={"category": ["bat", "noise", "unknown"]},
        dims=["category"],
    )
    predicted_tags = get_class_tags(
        class_scores=class_scores.values,
        targets=dummy_targets,
        top_class_only=True,
    )

    assert len(predicted_tags) == 1
    assert predicted_tags[0].tag.value == "uncertain"
    assert predicted_tags[0].score == 0.9


def test_get_class_tags_empty(dummy_targets):
    """Test with empty class scores."""
    class_scores = xr.DataArray([], coords={"category": []}, dims=["category"])
    predicted_tags = get_class_tags(
        class_scores=class_scores.values,
        targets=dummy_targets,
    )
    assert len(predicted_tags) == 0
