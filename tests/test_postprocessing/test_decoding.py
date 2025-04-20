from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
import xarray as xr
from soundevent import data

from batdetect2.postprocess.decoding import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    convert_raw_prediction_to_sound_event_prediction,
    convert_raw_predictions_to_clip_prediction,
    convert_xr_dataset_to_raw_prediction,
    get_class_tags,
    get_generic_tags,
    get_prediction_features,
)
from batdetect2.postprocess.types import RawPrediction


@pytest.fixture
def dummy_geometry_builder():
    """A simple GeometryBuilder that creates a BBox around the point."""

    def _builder(
        position: Tuple[float, float],
        dimensions: xr.DataArray,
    ) -> data.BoundingBox:
        time, freq = position
        width = dimensions.sel(dimension="width").item()
        height = dimensions.sel(dimension="height").item()
        return data.BoundingBox(
            coordinates=[
                time - width / 2,
                freq - height / 2,
                time + width / 2,
                freq + height / 2,
            ]
        )

    return _builder


@pytest.fixture
def dummy_sound_event_decoder():
    """A simple SoundEventDecoder mapping names to tags."""
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

    def _decoder(class_name: str) -> List[data.Tag]:
        return tag_map.get(class_name.lower(), [])

    return _decoder


@pytest.fixture
def generic_tags() -> List[data.Tag]:
    """Sample generic tags."""
    return [
        data.Tag(term=data.term_from_key(key="detector"), value="batdetect2")
    ]


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
        "freq": ("detection", expected_freqs),
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
            "score": scores,
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
        "freq": ("detection", np.array([], dtype=np.float64)),
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
def sample_raw_predictions() -> List[RawPrediction]:
    """Manually crafted RawPrediction objects using the actual type."""

    pred1_classes = xr.DataArray(
        [0.43, 0.85], coords={"category": ["bat", "noise"]}, dims=["category"]
    )
    pred1_features = xr.DataArray(
        [7.0, 16.0, 25.0, 34.0],
        coords={"feature": ["f0", "f1", "f2", "f3"]},
        dims=["feature"],
    )
    pred1 = RawPrediction(
        detection_score=0.9,
        start_time=20 - 7 / 2,
        end_time=20 + 7 / 2,
        low_freq=300 - 16 / 2,
        high_freq=300 + 16 / 2,
        class_scores=pred1_classes,
        features=pred1_features,
    )

    pred2_classes = xr.DataArray(
        [0.24, 0.66], coords={"category": ["bat", "noise"]}, dims=["category"]
    )
    pred2_features = xr.DataArray(
        [3.0, 12.0, 21.0, 30.0],
        coords={"feature": ["f0", "f1", "f2", "f3"]},
        dims=["feature"],
    )
    pred2 = RawPrediction(
        detection_score=0.8,
        start_time=10 - 3 / 2,
        end_time=10 + 3 / 2,
        low_freq=200 - 12 / 2,
        high_freq=200 + 12 / 2,
        class_scores=pred2_classes,
        features=pred2_features,
    )

    pred3_classes = xr.DataArray(
        [0.05, 0.02], coords={"category": ["bat", "noise"]}, dims=["category"]
    )
    pred3_features = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0],
        coords={"feature": ["f0", "f1", "f2", "f3"]},
        dims=["feature"],
    )
    pred3 = RawPrediction(
        detection_score=0.15,
        start_time=5.0,
        end_time=6.0,
        low_freq=50.0,
        high_freq=60.0,
        class_scores=pred3_classes,
        features=pred3_features,
    )
    return [pred1, pred2, pred3]


def test_convert_xr_dataset_basic(
    sample_detection_dataset, dummy_geometry_builder
):
    """Test basic conversion of a dataset to RawPrediction list."""
    raw_predictions = convert_xr_dataset_to_raw_prediction(
        sample_detection_dataset, dummy_geometry_builder
    )

    assert isinstance(raw_predictions, list)
    assert len(raw_predictions) == 2

    pred1 = raw_predictions[0]
    assert isinstance(pred1, RawPrediction)
    assert pred1.detection_score == 0.9

    assert pred1.start_time == 20 - 7 / 2
    assert pred1.end_time == 20 + 7 / 2
    assert pred1.low_freq == 300 - 16 / 2
    assert pred1.high_freq == 300 + 16 / 2
    xr.testing.assert_allclose(
        pred1.class_scores,
        sample_detection_dataset["classes"].sel(detection=0),
    )
    xr.testing.assert_allclose(
        pred1.features, sample_detection_dataset["features"].sel(detection=0)
    )

    pred2 = raw_predictions[1]
    assert isinstance(pred2, RawPrediction)
    assert pred2.detection_score == 0.8

    assert pred2.start_time == 10 - 3 / 2
    assert pred2.end_time == 10 + 3 / 2
    assert pred2.low_freq == 200 - 12 / 2
    assert pred2.high_freq == 200 + 12 / 2
    xr.testing.assert_allclose(
        pred2.class_scores,
        sample_detection_dataset["classes"].sel(detection=1),
    )
    xr.testing.assert_allclose(
        pred2.features, sample_detection_dataset["features"].sel(detection=1)
    )


def test_convert_xr_dataset_empty(
    empty_detection_dataset, dummy_geometry_builder
):
    """Test conversion of an empty dataset."""
    raw_predictions = convert_xr_dataset_to_raw_prediction(
        empty_detection_dataset, dummy_geometry_builder
    )
    assert isinstance(raw_predictions, list)
    assert len(raw_predictions) == 0


def test_convert_raw_to_sound_event_basic(
    sample_raw_predictions,
    sample_recording,
    dummy_sound_event_decoder,
    generic_tags,
):
    """Test basic conversion, default threshold, multi-label."""

    raw_pred = sample_raw_predictions[0]

    se_pred = convert_raw_prediction_to_sound_event_prediction(
        raw_prediction=raw_pred,
        recording=sample_recording,
        sound_event_decoder=dummy_sound_event_decoder,
        generic_class_tags=generic_tags,
    )

    assert isinstance(se_pred, data.SoundEventPrediction)
    assert se_pred.score == raw_pred.detection_score

    se = se_pred.sound_event
    assert isinstance(se, data.SoundEvent)
    assert se.recording == sample_recording
    assert isinstance(se.geometry, data.BoundingBox)
    np.testing.assert_allclose(
        se.geometry.coordinates,
        [
            raw_pred.start_time,
            raw_pred.low_freq,
            raw_pred.end_time,
            raw_pred.high_freq,
        ],
    )
    assert len(se.features) == len(raw_pred.features)

    feat_dict = {f.term.name: f.value for f in se.features}
    assert "batdetect2:f0" in feat_dict and isinstance(
        feat_dict["batdetect2:f0"], float
    )
    assert feat_dict["batdetect2:f0"] == 7.0

    expected_tags = {
        (generic_tags[0].term.name, generic_tags[0].value, 0.9),
        ("soundevent:category", "noise", 0.85),
        ("soundevent:species", "Myotis", 0.43),
    }
    actual_tags = {
        (pt.tag.term.name, pt.tag.value, pt.score) for pt in se_pred.tags
    }
    assert actual_tags == expected_tags


def test_convert_raw_to_sound_event_thresholding(
    sample_raw_predictions,
    sample_recording,
    dummy_sound_event_decoder,
    generic_tags,
):
    """Test effect of classification threshold."""
    raw_pred = sample_raw_predictions[0]
    high_threshold = 0.5

    se_pred = convert_raw_prediction_to_sound_event_prediction(
        raw_prediction=raw_pred,
        recording=sample_recording,
        sound_event_decoder=dummy_sound_event_decoder,
        generic_class_tags=generic_tags,
        classification_threshold=high_threshold,
        top_class_only=False,
    )

    expected_tags = {
        (generic_tags[0].term.name, generic_tags[0].value, 0.9),
        ("soundevent:category", "noise", 0.85),
    }
    actual_tags = {
        (pt.tag.term.name, pt.tag.value, pt.score) for pt in se_pred.tags
    }
    assert actual_tags == expected_tags


def test_convert_raw_to_sound_event_no_threshold(
    sample_raw_predictions,
    sample_recording,
    dummy_sound_event_decoder,
    generic_tags,
):
    """Test when classification_threshold is None."""
    raw_pred = sample_raw_predictions[2]

    se_pred = convert_raw_prediction_to_sound_event_prediction(
        raw_prediction=raw_pred,
        recording=sample_recording,
        sound_event_decoder=dummy_sound_event_decoder,
        generic_class_tags=generic_tags,
        classification_threshold=None,
        top_class_only=False,
    )

    expected_tags = {
        (generic_tags[0].term.name, generic_tags[0].value, 0.15),
        ("soundevent:species", "Myotis", 0.05),
        ("soundevent:category", "noise", 0.02),
    }
    actual_tags = {
        (pt.tag.term.name, pt.tag.value, pt.score) for pt in se_pred.tags
    }
    assert actual_tags == expected_tags


def test_convert_raw_to_sound_event_top_class(
    sample_raw_predictions,
    sample_recording,
    dummy_sound_event_decoder,
    generic_tags,
):
    """Test top_class_only=True behavior."""
    raw_pred = sample_raw_predictions[0]

    se_pred = convert_raw_prediction_to_sound_event_prediction(
        raw_prediction=raw_pred,
        recording=sample_recording,
        sound_event_decoder=dummy_sound_event_decoder,
        generic_class_tags=generic_tags,
        classification_threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
        top_class_only=True,
    )

    expected_tags = {
        (generic_tags[0].term.name, generic_tags[0].value, 0.9),
        ("soundevent:category", "noise", 0.85),
    }
    actual_tags = {
        (pt.tag.term.name, pt.tag.value, pt.score) for pt in se_pred.tags
    }
    assert actual_tags == expected_tags


def test_convert_raw_to_sound_event_all_below_threshold(
    sample_raw_predictions,
    sample_recording,
    dummy_sound_event_decoder,
    generic_tags,
):
    """Test when all class scores are below the default threshold."""
    raw_pred = sample_raw_predictions[2]

    se_pred = convert_raw_prediction_to_sound_event_prediction(
        raw_prediction=raw_pred,
        recording=sample_recording,
        sound_event_decoder=dummy_sound_event_decoder,
        generic_class_tags=generic_tags,
        classification_threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
        top_class_only=False,
    )

    expected_tags = {
        (generic_tags[0].term.name, generic_tags[0].value, 0.15),
    }
    actual_tags = {
        (pt.tag.term.name, pt.tag.value, pt.score) for pt in se_pred.tags
    }
    assert actual_tags == expected_tags


def test_convert_raw_list_to_clip_basic(
    sample_raw_predictions,
    sample_clip,
    dummy_sound_event_decoder,
    generic_tags,
):
    """Test converting a list of RawPredictions to a ClipPrediction."""
    clip_pred = convert_raw_predictions_to_clip_prediction(
        raw_predictions=sample_raw_predictions,
        clip=sample_clip,
        sound_event_decoder=dummy_sound_event_decoder,
        generic_class_tags=generic_tags,
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
    expected_tags3 = {
        (generic_tags[0].term.name, generic_tags[0].value, 0.15),
    }
    assert se_pred3_tags == expected_tags3


def test_convert_raw_list_to_clip_empty(
    sample_clip,
    dummy_sound_event_decoder,
    generic_tags,
):
    """Test converting an empty list of RawPredictions."""
    clip_pred = convert_raw_predictions_to_clip_prediction(
        raw_predictions=[],
        clip=sample_clip,
        sound_event_decoder=dummy_sound_event_decoder,
        generic_class_tags=generic_tags,
    )

    assert isinstance(clip_pred, data.ClipPrediction)
    assert clip_pred.clip == sample_clip
    assert len(clip_pred.sound_events) == 0


def test_convert_raw_list_to_clip_passes_args(
    sample_raw_predictions,
    sample_clip,
    dummy_sound_event_decoder,
    generic_tags,
):
    """Test that arguments like top_class_only are passed through."""

    clip_pred = convert_raw_predictions_to_clip_prediction(
        raw_predictions=sample_raw_predictions,
        clip=sample_clip,
        sound_event_decoder=dummy_sound_event_decoder,
        generic_class_tags=generic_tags,
        classification_threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
        top_class_only=True,
    )

    assert len(clip_pred.sound_events) == 3

    se_pred1_tags = {
        (pt.tag.term.name, pt.tag.value, pt.score)
        for pt in clip_pred.sound_events[0].tags
    }
    expected_tags1 = {
        (generic_tags[0].term.name, generic_tags[0].value, 0.9),
        ("soundevent:category", "noise", 0.85),
    }
    assert se_pred1_tags == expected_tags1


def test_get_generic_tags_basic(generic_tags):
    """Test creation of generic tags with score."""
    detection_score = 0.75
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
    features = get_prediction_features(feature_data)
    assert len(features) == 3
    for feature, feat_name, feat_value in zip(
        features, ["feat1", "feat2", "feat3"], [1.1, 2.2, 3.3]
    ):
        assert isinstance(feature, data.Feature)
        assert feature.term.name == f"batdetect2:{feat_name}"
        assert feature.value == feat_value


def test_get_class_tags_basic(dummy_sound_event_decoder):
    """Test creation of class tags based on scores and decoder."""
    class_scores = xr.DataArray(
        [0.6, 0.2, 0.9],
        coords={"category": ["bat", "noise", "unknown"]},
        dims=["category"],
    )
    predicted_tags = get_class_tags(
        class_scores=class_scores,
        sound_event_decoder=dummy_sound_event_decoder,
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


def test_get_class_tags_thresholding(dummy_sound_event_decoder):
    """Test class tag creation with a threshold."""
    class_scores = xr.DataArray(
        [0.6, 0.2, 0.9],
        coords={"category": ["bat", "noise", "unknown"]},
        dims=["category"],
    )
    threshold = 0.5
    predicted_tags = get_class_tags(
        class_scores=class_scores,
        sound_event_decoder=dummy_sound_event_decoder,
        threshold=threshold,
    )

    assert len(predicted_tags) == 2
    tag_values = [pt.tag.value for pt in predicted_tags]
    assert "Myotis" in tag_values
    assert "noise" not in tag_values
    assert "uncertain" in tag_values


def test_get_class_tags_top_class_only(dummy_sound_event_decoder):
    """Test class tag creation with top_class_only."""
    class_scores = xr.DataArray(
        [0.6, 0.2, 0.9],
        coords={"category": ["bat", "noise", "unknown"]},
        dims=["category"],
    )
    predicted_tags = get_class_tags(
        class_scores=class_scores,
        sound_event_decoder=dummy_sound_event_decoder,
        top_class_only=True,
    )

    assert len(predicted_tags) == 1
    assert predicted_tags[0].tag.value == "uncertain"
    assert predicted_tags[0].score == 0.9


def test_get_class_tags_empty(dummy_sound_event_decoder):
    """Test with empty class scores."""
    class_scores = xr.DataArray([], coords={"category": []}, dims=["category"])
    predicted_tags = get_class_tags(
        class_scores=class_scores,
        sound_event_decoder=dummy_sound_event_decoder,
    )
    assert len(predicted_tags) == 0
